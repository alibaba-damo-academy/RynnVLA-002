from collections import defaultdict, deque
import datetime
import logging
import random
import time

from fairscale.nn.model_parallel import initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def random_seed(seed=0):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=1000, fmt=None):
        if fmt is None:
            fmt = "{avg:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            elif isinstance(v, (torch.Tensor, float, int)):
                self.meters[k].update(v.item() if isinstance(v, torch.Tensor) else v)
            elif isinstance(v, list):
                for i, sub_v in enumerate(v):
                    self.meters[f"{k}_{i}"].update(sub_v.item() if isinstance(sub_v, torch.Tensor) else sub_v)
            elif isinstance(v, dict):
                for sub_key, sub_v in v.items():
                    self.meters[f"{k}_{sub_key}"].update(sub_v.item() if isinstance(sub_v, torch.Tensor) else sub_v)
            else:
                raise TypeError(f"Unsupported type {type(v)} for metric {k}")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, start_iter=0, samples_per_iter=None):
        i = start_iter
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        log_msg = [header, "[{0" + "}/{1}]", "{meters}", "time: {time}", "data: {data}"]
        if samples_per_iter is not None:
            log_msg.append("samples/sec: {samples_per_sec:.2f}")
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                try:
                    total_len = len(iterable)
                except:
                    total_len = "unknown"

                msg_kwargs = {
                    "meters": str(self),
                    "time": str(iter_time),
                    "data": str(data_time),
                }
                if samples_per_iter is not None:
                    msg_kwargs["samples_per_sec"] = samples_per_iter / iter_time.avg
                if torch.cuda.is_available():
                    msg_kwargs["memory"] = torch.cuda.max_memory_allocated() / MB

                logger.info(log_msg.format(i, total_len, **msg_kwargs))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {}".format(header, total_time_str))


def add_weight_decay(model, lr, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if name.endswith(".bias") or name.endswith("norm.weight"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
    ]


def broadcast_nonmp_parameters(model):
    if fs_init.get_model_parallel_world_size() == 1:
        return
    logger.info("starting broadcast non-model-parallel parameters within model parallel group")
    memo = set()
    modules = model.named_modules(prefix="", remove_duplicate=True)
    for module_prefix, module in modules:
        members = dict(module._parameters.items())
        for k, v in members.items():
            name = module_prefix + ("." if module_prefix else "") + k
            if v is None or v in memo:
                continue
            if getattr(v, "model_parallel", False):
                logger.info(f"ignore: {name}")
                continue
            memo.add(v)
            dist.broadcast(v, src=fs_init.get_model_parallel_src_rank(), group=fs_init.get_model_parallel_group())
    logger.info("braodcast done")


def mark_mp_params(model: torch.nn.Module):
    from fairscale.nn.model_parallel.layers import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear

    for m in model.modules():
        if isinstance(m, ColumnParallelLinear):
            m.weight.model_parallel = True
            if m.bias is not None:
                m.bias.model_parallel = True

        if isinstance(m, RowParallelLinear):
            m.weight.model_parallel = True

        if isinstance(m, ParallelEmbedding):
            m.weight.model_parallel = True


def print_param_status(model: torch.nn.Module) -> None:
    require_grad_set = []
    no_grad_set = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            require_grad_set.append((name, param))
        else:
            no_grad_set.append((name, param))

    logger.info("Params that require gradient:\n")
    for name, param in require_grad_set:
        model_parallel = getattr(param, "model_parallel", False)
        logger.info(
            f"Param {name}: requires_grad {param.requires_grad}, local_size {param.shape}, model_parallel {model_parallel}, dtype {param.dtype}"
        )

    logger.info("\nParams that do not require gradient:\n")
    for name, param in no_grad_set:
        model_parallel = getattr(param, "model_parallel", False)
        logger.info(
            f"Param {name}: requires_grad {param.requires_grad}, local_size {param.shape}, model_parallel {model_parallel}, dtype {param.dtype}"
        )
