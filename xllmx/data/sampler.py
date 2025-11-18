from collections import defaultdict
import logging
from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler

from xllmx.data.dataset import FinetuneConversationDataset

logger = logging.getLogger(__name__)


# todo too slow to be used
def mild_shuffle(items: List, shuffle_factor, engine: np.random.Generator):
    """
    Perform a mild shuffle on the list of items.

    Args:
        engine: random engine
        items (list): The list of items to shuffle.
        shuffle_factor (float): max swap range is computed as len(item) * shuffle_factor.

    Returns:
        list: The mildly shuffled list.
    """

    n = len(items)
    swap_range = int(shuffle_factor * n)
    shuffled_items = [None for _ in items]
    cache = list(range(swap_range))
    for i in range(n):
        if i + swap_range < n:
            cache.append(i + swap_range)
        if len(cache) == 0 or cache[0] != i:  # already swapped
            assert shuffled_items[i] is not None
            continue
        else:
            cache = cache[1:]
            if len(cache) == 0:
                shuffled_items[i] = items[i]
            else:
                cache_idx = engine.integers(low=0, high=len(cache))
                j = cache[cache_idx]
                del cache[cache_idx]
                shuffled_items[i], shuffled_items[j] = items[j], items[i]

    return shuffled_items


class FinetuneDistSampler(Sampler):
    def __init__(
        self,
        dataset: FinetuneConversationDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        batch_size=None,
        acc_grad=1,
        length_clustering=True,
        allow_mixed_task_among_acc=False,
        preprocess=False,
    ):
        """
        Distributed Sampler ensuring data in a batch are of the same type (e.g. text, image-text)
        :param dataset:
        :param num_replicas:
        :param rank:
        :param shuffle:
        :param seed:
        :param batch_size:
        :param acc_grad:
        :param length_clustering:
        :param allow_mixed_task_among_acc:
        """
        # super().__init__()

        if num_replicas is None or rank is None or rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid num_replicas ({num_replicas}) or rank ({rank})")
        assert batch_size is not None

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.acc_grad = acc_grad
        self.length_clustering = length_clustering
        self.allow_mixed_task_among_acc = allow_mixed_task_among_acc
        self.preprocess = preprocess

        self.epoch = 0
        self.start_iter = 0

        global_bsz_acc = batch_size * num_replicas * acc_grad

        if preprocess:
            group_len = defaultdict(int)
            for i, meta in enumerate(dataset.meta_collection):
                group_len[meta["type"]] += int(meta["len"] * meta.get("ratio", 1.0))

            group_len = {key: val // global_bsz_acc * global_bsz_acc for key, val in group_len.items()}

            self.total_size = sum(list(group_len.values()))
        else:
            self.total_size = len(dataset) // global_bsz_acc * global_bsz_acc
        assert self.total_size % num_replicas == 0
        self.num_samples = self.total_size // num_replicas

    def __iter__(self) -> Iterator:
        global_batch_size = self.batch_size * self.num_replicas
        global_bsz_acc = self.batch_size * self.num_replicas * self.acc_grad
        rng = np.random.default_rng(self.seed + self.epoch)

        if self.preprocess:
            group_indices_and_len = defaultdict(list)

            # Initialize the starting index
            start_idx = 0

            # Iterate through the list of dictionaries
            for i, meta in enumerate(self.dataset.meta_collection):
                # Calculate the ending index for the current collection
                end_idx = start_idx + meta["len"]
                indices = list(range(start_idx, end_idx))
                assert len(indices) == len(meta["item_len_list"])
                indices_and_len = [[idx, length] for idx, length in zip(indices, meta["item_len_list"])]
                if meta.get("ratio", 1.0) != 1.0:
                    indices_and_len = list(rng.choice(indices_and_len, int(meta["len"] * meta["ratio"]), replace=False))
                    logger.info(f"meta{i}: sample (ratio = {meta['ratio']}) {len(indices_and_len)} items")
                group_indices_and_len[meta["type"]].extend(indices_and_len)

                # Update the starting index for the next collection
                start_idx = end_idx

            for group_name, indices_and_len in group_indices_and_len.items():
                group_indices_and_len[group_name] = indices_and_len[
                    : len(indices_and_len) // global_bsz_acc * global_bsz_acc
                ]
        else:
            indices = list(range(self.total_size))
        
        if self.shuffle:
            if self.preprocess:
                group_indices = {}
                if self.length_clustering:
                    for group_name, indices_and_len in group_indices_and_len.items():
                        indices_and_len.sort(key=lambda x: x[1])
                        group_indices[group_name] = [_[0] for _ in indices_and_len]
                    
                    # option1: shuffle among neighboring items
                    for group_name, indices in group_indices.items():
                        result = []
                        for pos in range(0, len(indices), global_batch_size * 500):
                            sublist = indices[pos : pos + global_batch_size * 500]
                            rng.shuffle(sublist)
                            result.extend(sublist)
                        group_indices[group_name] = result
                    # option2: mild shuffle
                    #     group_indices[group_name] = mild_shuffle(indices, 0.1, rng)
                    # option3: do nothing
                    # pass
                else:
                    for group_name, indices_and_len in group_indices_and_len.items():
                        rng.shuffle(indices_and_len)
                        group_indices[group_name] = [_[0] for _ in indices_and_len]
                
                del group_indices_and_len

                if self.allow_mixed_task_among_acc:
                    global_batched_indices = [
                        indices[i : i + global_batch_size]
                        for group_name, indices in group_indices.items()
                        for i in range(0, len(indices), global_batch_size)
                    ]
                else:
                    global_batched_indices = []
                    for group_name, indices in group_indices.items():
                        group_batched_indices = [
                            indices[i : i + global_batch_size] for i in range(0, len(indices), global_batch_size)
                        ]
                        rng.shuffle(group_batched_indices)
                        group_batched_indices = [
                            sum(group_batched_indices[i : i + self.acc_grad], start=[])
                            for i in range(0, len(group_batched_indices), self.acc_grad)
                        ]
                        global_batched_indices.extend(group_batched_indices)
                rng.shuffle(global_batched_indices)
                indices = [_ for batch_indices in global_batched_indices for _ in batch_indices]
            else:
                rng.shuffle(indices)
                global_batched_indices = [
                    indices[i : i + global_batch_size]
                    for i in range(0, len(indices), global_batch_size)
                ]
                indices = [_ for batch_indices in global_batched_indices for _ in batch_indices]
        else:
            raise NotImplementedError()

        assert len(indices) == self.total_size

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos : start_pos + self.batch_size]
        # subsample
        assert len(own_indices) == self.num_samples

        if self.start_iter * self.batch_size > len(own_indices):
            own_indices = []
        else:
            own_indices = own_indices[self.start_iter * self.batch_size :]
                
        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, start_iter: int = 0) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            start_iter (int): start iter number.
        """
        self.epoch = epoch
        self.start_iter = start_iter
