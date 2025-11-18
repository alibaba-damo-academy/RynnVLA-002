import copy
import json
import logging
import os
from pathlib import Path
import pickle
from time import sleep
import traceback
import warnings

import h5py
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml

from .item_processor import ItemProcessorBase

logger = logging.getLogger(__name__)


class FinetuneConversationDataset(Dataset):
    def __init__(self, config_path, item_processor: ItemProcessorBase, cache_on_disk=False):

        self.item_processor = item_processor

        logger.info(f"read dataset config from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:")
        logger.info(self.config)

        self.cache_on_disk = cache_on_disk
        if self.cache_on_disk:
            cache_dir = self._get_cache_dir(config_path)
            if dist.get_rank() == 0:
                self._collect_annotations_and_save_to_cache(cache_dir)
            dist.barrier()
            self.meta_collection, self.annotations_collection = self._load_annotations_from_cache(cache_dir)
        else:
            cache_dir = None
            self.meta_collection, self.annotations_collection = self._collect_annotations()

    def __len__(self):
        return sum([_["len"] for _ in self.meta_collection])

    def _collect_annotations(self):
        meta_collection = []
        annotations_collection = []

        for meta in self.config["META"]:
            meta, annotations = self._load_meta(meta)
            meta_collection.append(meta)
            annotations_collection.append(annotations)

        return meta_collection, annotations_collection

    def _load_meta(self, meta):
        if "type" not in meta:
            meta["type"] = "default"

        meta_path, meta_type = meta["path"], meta["type"]
        meta_ext = os.path.splitext(meta_path)[-1]
        if meta_ext == ".json":
            with open(meta_path) as f:
                annotations = json.load(f)
        elif meta_ext == ".jsonl":
            annotations = []
            with open(meta_path) as f:
                for i, line in enumerate(f):
                    try:
                        annotations.append(json.loads(line))
                    except json.decoder.JSONDecodeError as e:
                        logger.error(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}")
                        raise e
        elif meta_ext == ".pkl":
            with open(meta_path, "rb") as f:
                annotations = pickle.load(f)
            assert isinstance(annotations, list)
        elif meta_ext == ".pth":
            annotations = torch.load(meta_path)
            assert isinstance(annotations, list)
        else:
            raise NotImplementedError(
                f'Unknown meta file extension: "{meta_ext}". '
                f"Currently, .json, .jsonl are supported. "
                "If you are using a supported format, please set the file extension so that the proper parsing "
                "routine can be called."
            )
        logger.info(f"{meta_path}, type{meta_type}: len {len(annotations)}")

        meta["len"] = len(annotations)

        meta["item_len_list"] = [self.item_processor.predict_item_token_length(_) for _ in annotations]

        return meta, annotations

    def _collect_annotations_and_save_to_cache(self, cache_dir):
        if (Path(cache_dir) / "data.h5").exists() and (Path(cache_dir) / "ready").exists():
            # off-the-shelf annotation cache exists
            warnings.warn(
                f"Use existing h5 data cache: {Path(cache_dir)}\n"
                f"Note: if the actual data defined by the data config has changed since your last run, "
                f"please delete the cache manually and re-run this experiment, or the data actually used "
                f"will not be updated"
            )
            return

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        meta_collection, annotations_collection = self._collect_annotations()

        # when cache on disk, rank0 saves items to an h5 file
        logger.info(f"start to build data cache to: {Path(cache_dir)}")
        with h5py.File(Path(cache_dir) / "data.h5", "w") as file:
            dt = h5py.vlen_dtype(str)
            for i, annotations in enumerate(annotations_collection):
                serialized_ann = [json.dumps(_) for _ in annotations]
                h5_ann = file.create_dataset(f"ann{i}", (len(serialized_ann),), dtype=dt)
                h5_ann[:] = serialized_ann

            file.create_dataset("meta_collection", data=json.dumps(meta_collection))
        with open(Path(cache_dir) / "ready", "w") as f:
            f.write("ready")
        logger.info(f"data cache built")

    @staticmethod
    def _get_cache_dir(config_path):
        config_identifier = config_path
        disallowed_chars = ["/", "\\", ".", "?", "!"]
        for _ in disallowed_chars:
            config_identifier = config_identifier.replace(_, "-")
        cache_dir = f"/mnt/PLNAS/cenjun/processed_lerobot_data/xllmx_data_cache_2/{config_identifier}"
        return cache_dir

    @staticmethod
    def _load_annotations_from_cache(cache_dir):
        while not (Path(cache_dir) / "ready").exists():
            # cache has not yet been completed by rank 0
            assert dist.get_rank() != 0
            sleep(1)
        cache_file = h5py.File(Path(cache_dir) / "data.h5", "r")
        meta_collection = json.loads(cache_file["meta_collection"].asstr()[()])
        annotations_collection = [cache_file[f"ann{i}"] for i in range(len(meta_collection))]
        return meta_collection, annotations_collection

    def get_item_func(self, meta_idx, idx_in_meta):
        data_item = self.annotations_collection[meta_idx][idx_in_meta]
        if self.cache_on_disk:
            data_item = json.loads(data_item)
        else:
            data_item = copy.deepcopy(data_item)

        return self.item_processor.process_item(data_item, training_mode=True)

    def tie_index_to_meta(self, idx: int):
        # Initialize the starting index
        start_idx = 0

        # Iterate through the list of dictionaries
        for i, meta in enumerate(self.meta_collection):
            # Calculate the ending index for the current collection
            end_idx = start_idx + meta["len"]

            # Check if the given index falls within the current collection
            if start_idx <= idx < end_idx:
                # Calculate the new index within the current collection
                new_index = idx - start_idx
                return i, new_index

            # Update the starting index for the next collection
            start_idx = end_idx

        # If the index is out of range of all collections, raise an error
        raise IndexError("Index out of range")

    def __getitem__(self, index):
        meta_idx, idx_in_meta = self.tie_index_to_meta(index)

        try:
            return self.get_item_func(meta_idx, idx_in_meta)
        except Exception as e:
            logger.info(
                f"Item {index} errored, annotation:\n"
                f"{self.annotations_collection[meta_idx][idx_in_meta]}\n"
                f"Error:\n"
                f"{traceback.format_exc()}"
            )
            if idx_in_meta != 0:
                return self[index - 1]
            else:
                return self[index + self.meta_collection[meta_idx]["len"] - 1]
