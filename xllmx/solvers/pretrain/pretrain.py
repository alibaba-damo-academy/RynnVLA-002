from abc import ABC, abstractmethod
import argparse
import contextlib
import datetime
import functools
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Optional, Union
import warnings

from fairscale.nn.model_parallel import initialize as fs_init
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import GenerationConfig, TextStreamer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, LogitsWarper
from data.item_processor import FlexARItemProcessor
from data.pre_tokenize_action import ItemProcessor
import transformers

from transformers import AutoProcessor

try:
    from apex.optimizers import FusedAdam as AdamW
except ImportError:
    warnings.warn("cannot import FusedAdam from apex, use torch AdamW instead")
    from torch.optim import AdamW

from xllmx.data.dataset import FinetuneConversationDataset, ItemProcessorBase
from xllmx.data.sampler import FinetuneDistSampler
from xllmx.model.tokenizer import Tokenizer
import xllmx.util as util
import xllmx.util.lr_sched as lr_sched
import xllmx.util.misc as misc
from xllmx.util.tensor_type import promote_param_to_fp32


class PretrainSolverBase(ABC):

    def __init__(self, args):
        self.args = args
        util.dist.init_distributed_mode(args)
        self.logger = self.configure_logger()
        self.logger.info(args)

        assert args.model_parallel_size == 1, (
            "Model parallelism currently not supported, ",
            "so please keep model_parallel_size to 1\n"
            "Note that model parallelism is different from and orthogonal to FSDP"
        )
        fs_init.initialize_model_parallel(args.model_parallel_size)
        self.global_rank = dist.get_rank()
        self.mp_rank = fs_init.get_model_parallel_rank()
        self.mp_world_size = fs_init.get_model_parallel_world_size()
        self.mp_group = fs_init.get_model_parallel_group()
        self.dp_rank = fs_init.get_data_parallel_rank()
        self.dp_world_size = fs_init.get_data_parallel_world_size()
        self.dp_group = fs_init.get_data_parallel_group()

        if self.args.auto_resume and self.args.resume_path is None:
            existing_checkpoints = [_ for _ in os.listdir(self.args.output_dir) if "epoch" in _]
            if len(existing_checkpoints) > 0:

                def ckpt_sort_key(s):
                    # divide ckpt directory names into epoch and iter parts
                    epoch, iteration = util.ckpt.split_ckpt_str_into_epoch_iter(s)
                    if iteration is None:
                        iteration = float("inf")
                    return epoch, iteration

                self.args.resume_path = os.path.join(
                    self.args.output_dir, sorted(existing_checkpoints, key=ckpt_sort_key)[-1]
                )
                self.logger.info(f"auto resume from {self.args.resume_path}")

        if args.output_dir and self.global_rank == 0:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        dist.barrier()

        if args.precision == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.logger.info("work dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        self.logger.info("{}".format(self.args).replace(", ", ",\n"))

        # define the model
        self.mixed_precision_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "tf32": torch.float32,
        }[self.args.precision]

        self.model, self.tokenizer, self.optimizer = self.build_model()

        self.dataset_train, self.sampler_train, self.dataloader_train, self.dataset_val_ind, self.sampler_val_ind, self.dataloader_val_ind, self.dataset_val_ood, self.sampler_val_ood, self.dataloader_val_ood = self.build_data_new()

        self.start_epoch = 0
        self.start_iter = 0
        self.metric_logger_to_resume = None

        if self.args.resume_path and not self.args.eval_only and not self.args.ft:
            self.resume(self.args.resume_path)

        if self.global_rank == 0:
            (Path(args.output_dir) / "tensorboard").mkdir(parents=True, exist_ok=True)
            self.log_writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "tensorboard"))
        else:
            self.log_writer = None
        
        self.item_processor = FlexARItemProcessor(target_size=288)
        self.item_processor_action = ItemProcessor(target_size=256)
        # self.tokenizer_fast = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

        gc.collect()
        torch.cuda.empty_cache()

    def configure_logger(self):
        rank = dist.get_rank()

        logger = logging.getLogger()

        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()  # Console handler
        f_handler = logging.FileHandler(Path(self.args.output_dir) / f"common.log")  # Rank-specific
        f_rank_handler = logging.FileHandler(
            Path(self.args.output_dir) / f"rank-{dist.get_rank()}.log"
        )  # Rank-specific

        # Console and common file handler captures all INFO and above messages
        c_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        f_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        f_rank_handler.setLevel(logging.INFO)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter(f"[rank{rank}:%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        f_rank_handler.setFormatter(formatter)
        # Set the log level based on the rank argument

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        logger.addHandler(f_rank_handler)

        return logger

    @classmethod
    def get_args_parser(cls):
        parser = argparse.ArgumentParser("xllmx Finetuning", add_help=False)

        # Schedule
        parser.add_argument(
            "--batch_size",
            default=4,
            type=int,
            help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
        )
        parser.add_argument(
            "--accum_iter",
            default=4,
            type=int,
            help="Accumulate gradient iterations " "(for increasing the effective batch size under memory constraints)",
        )
        parser.add_argument("--epochs", default=1, type=int)
        parser.add_argument("--warmup_epochs", type=float, default=0.03, help="epoch to warmup LR")

        # Optimizer parameters
        parser.add_argument("--lr", type=float, default=0.00002, help="learning rate (absolute lr)")
        parser.add_argument(
            "--min_lr", type=float, default=0.00, help="lower lr bound for cyclic schedulers that hit 0"
        )
        parser.add_argument("--wd", type=float, default=0.00, help="weight decay (default: 0.00)")
        parser.add_argument("--clip_grad", type=float, default=4.0, help="grad clipping norm")

        parser.add_argument("--init_from", default=None, type=str, help="path to checkpoint for model initialization")

        # Data parameters
        parser.add_argument("--data_config", default="/path/to/data/config/yaml", type=str, help="data config path")
        parser.add_argument("--data_config_train", default="/path/to/data/config/yaml", type=str, help="data config path")
        parser.add_argument("--data_config_val_ind", default="/path/to/data/config/yaml", type=str, help="data config path")
        parser.add_argument("--data_config_val_ood", default="/path/to/data/config/yaml", type=str, help="data config path")
        parser.add_argument(
            "--cache_ann_on_disk",
            action="store_true",
            help="cache the dataset annotations on disk to avoid duplication across ranks. "
            "can save CPU memory, especially with large datasets",
        )
        parser.add_argument(
            "--length_clustering",
            default=True,
            help="gather items with similar length to the same batch",
        )
        parser.add_argument("--disable_length_clustering", action="store_false", dest="length_clustering")
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument(
            "--pin_mem",
            action="store_true",
            help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
        )
        parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
        parser.set_defaults(pin_mem=True)

        # Seed
        parser.add_argument("--seed", default=0, type=int)

        # Control
        parser.add_argument("--output_dir", default="./output_dir", help="path for outputs")
        parser.add_argument("--save_interval", default=1, type=int, help="number of epochs between model saving")
        parser.add_argument(
            "--save_iteration_interval",
            default=5000,
            type=int,
            help="number of iterations between within-epoch model saving",
        )
        parser.add_argument(
            "--only_save_trainable", default=False, action="store_true", help="only save trainable model parameters"
        )
        parser.add_argument(
            "--ckpt_max_keep", default=2, type=int, help="maximum number of checkpoints to keep, <=0 means keep all"
        )
        parser.add_argument("--auto_resume", default=True, help="auto resume from args.output_dir")
        parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
        parser.add_argument("--resume_path", default=None, type=str, help="manually specify resume checkpoint")

        # Parallel
        parser.add_argument("--model_parallel_size", type=int, default=1)
        parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
        parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "tf32"], default="bf16")
        parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"], default="fp32")

        # Checkpointing
        parser.add_argument("--checkpointing", action="store_true", default=False, help="enable gradient checkpointing")
        # parser.add_argument('--quant', action="store_true", default=False,  # todo
        #                     help="enable quantization to speedup and save memory")
        parser.add_argument("--eval_only", type=bool, default=False, help="enable gradient checkpointing")
        parser.add_argument("--ft", type=bool, default=False, help="fintune from the pretrained model or not")
        parser.add_argument("--ablation", type=str, choices=["0", "1", "2", "3", "4", "5"], default="fp32")


        return parser

    def build_model(self) -> (nn.Module, Tokenizer):
        init_from = self.args.resume_path or self.args.init_from
        if init_from is None:
            # starting_point_path = Path(self.args.output_dir) / "starting_point"
            starting_point_path = Path('../ckpts') / 'starting_point'
            if dist.get_rank() == 0:
                if (starting_point_path / "config.json").exists():
                    self.logger.info(f"will use existing starting point at {starting_point_path}")
                    self.logger.info(
                        f"***********************************************************************\n"
                        f"********************************Caution********************************\n"
                        f"Caution: the starting point is created by some previous experiment run \n"
                        f"If the starting point saved by that run is broken, or if the expected  \n"
                        f"starting weights for the model has changed since that run, please manu-\n"
                        f"remove the saved path: \n"
                        f"{starting_point_path} \n"
                        f"and rerun the experiment.\n"
                        f"***********************************************************************\n"
                        f"***********************************************************************\n"
                    )
                else:
                    self.logger.info(f"creating starting-point weights at {starting_point_path}")
                    # self._make_and_save_starting_point(save_path=str(starting_point_path))
            dist.barrier()
            init_from = str(starting_point_path)

        self.logger.info(f"Start instantiating unwrapped model from {init_from}")

        # only rank 0 instantiate, otherwise to meta
        unwrapped_model, tokenizer = self._model_func(init_from)
        if hasattr(unwrapped_model, "get_trainable_params"):
            trainable_params = dict(unwrapped_model.get_trainable_params())
            for key, param in unwrapped_model.named_parameters():
                if key in trainable_params or 'lora' in key:
                    param.requires_grad = True
                    promote_param_to_fp32(param)
                else:
                    param.requires_grad = False
                    promote_param_to_fp32(param)
                    # keep_fp32_keywords = ["norm", "lm_head", "embed_tokens"]
                    # if any([_ in key for _ in keep_fp32_keywords]):
                    #     promote_param_to_fp32(param)
                    # elif param.is_floating_point():
                    #     param.data = param.data.to(self.mixed_precision_dtype)
        else:
            self.logger.warning(
                f"model class {type(unwrapped_model)} does not have `get_trainable_params` method,"
                f"set all params to trainable"
            )
            for key, param in unwrapped_model.named_parameters():
                param.requires_grad = True
                param.requires_grad = True
                promote_param_to_fp32(param)

        self.logger.info("Finish instantiating unwrapped model.")
        self.logger.info(f"Unwrapped model: \n{str(unwrapped_model)}")
        # self.logger.info(f"Model config: \n{unwrapped_model.config.to_dict()}")

        # ----------------
        self.is_peft = getattr(unwrapped_model, "is_peft", False)  # todo
        self.logger.info(f"Model is Peft: {self.is_peft}")
        # ----------------

        misc.mark_mp_params(unwrapped_model)

        # defer this after FSDP
        misc.print_param_status(unwrapped_model)

        train_param_count_local, train_param_count_all = 0, 0
        frozen_param_count_local, frozen_param_count_all = 0, 0
        for name, param in unwrapped_model.named_parameters():
            model_parallel = getattr(param, "model_parallel", False)
            if param.requires_grad:
                if model_parallel:
                    train_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    train_param_count_all += param.numel()
                train_param_count_local += param.numel()
            else:
                if model_parallel:
                    frozen_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    frozen_param_count_all += param.numel()
                frozen_param_count_local += param.numel()

        self.logger.info(
            f"Trainable parameter count : {train_param_count_local} (local rank), {train_param_count_all} (all).\n"
            f"Frozen parameter count : {frozen_param_count_local} (local rank), {frozen_param_count_all} (all)."
        )

        # checkpointing (part1, should be called before FSDP wrapping)
        if self.args.eval_only:
            checkpointing_list = []
        else:
            if self.args.checkpointing:
                # todo more hints for not-implemented
                checkpointing_list = unwrapped_model.get_checkpointing_wrap_module_list()
            else:
                checkpointing_list = []

        # todo pre-sync ignored states
        model = self.setup_fsdp_sync(
            unwrapped_model, self.args.data_parallel, self.args.precision, self.args.grad_precision
        )

        # broadcast non-model-parallel parameters within model parallel group
        misc.broadcast_nonmp_parameters(model)

        # checkpointing (part2, after FSDP wrapping)
        if self.args.checkpointing:
            print("apply gradient checkpointing")
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: submodule in checkpointing_list,
            )

        self.logger.info(f"Wrapped model: \n{str(model)}")

        # Setup optimizer
        opt = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.95))

        return model, tokenizer, opt

    @abstractmethod
    def _model_func(self, init_from: str) -> (nn.Module, Tokenizer | None):  # todo return type get finer # noqa
        raise NotImplementedError(f"{self.__class__} has to implement model_func for model instantiation")

    @abstractmethod
    def _make_and_save_starting_point(self, save_path: str):
        raise NotImplementedError(f"{self.__class__} has not implemented _make_and_save_starting_point()")

    def setup_fsdp_sync(self, model: nn.Module, data_parallel: str, precision: str, grad_precision: Optional[str]) -> FSDP:

        if self.dp_rank == 0:
            param_init_fn = None
        else:
            param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)
        
        model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
            ),
            process_group=fs_init.get_data_parallel_group(),
            sharding_strategy={
                "fsdp": ShardingStrategy.FULL_SHARD,
                "sdp": ShardingStrategy.SHARD_GRAD_OP,
            }[data_parallel],
            mixed_precision=MixedPrecision(
                param_dtype={
                    "fp32": torch.float,
                    "tf32": torch.float,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[precision],
                reduce_dtype={
                    "fp32": torch.float,
                    "tf32": torch.float,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[grad_precision or precision],
            ),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            limit_all_gathers=True,
            use_orig_params=True,
            param_init_fn=param_init_fn

        )
        torch.cuda.synchronize()

        return model

    def build_data(self):
        eff_batch_size = self.args.batch_size * self.args.accum_iter * fs_init.get_data_parallel_world_size()
        self.logger.info("effective batch size: %d" % eff_batch_size)
        dataset_train = self._dataset_func()
        self.logger.info(dataset_train)

        sampler_train = FinetuneDistSampler(
            dataset_train,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
            batch_size=self.args.batch_size,
            acc_grad=self.args.accum_iter,
            seed=self.args.seed,
            length_clustering=self.args.length_clustering,
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler_train,
            collate_fn=lambda batch: tuple(zip(*batch)),
            drop_last=True,
        )

        return dataset_train, sampler_train, dataloader_train
    
    def build_data_new(self):
        eff_batch_size = self.args.batch_size * self.args.accum_iter * fs_init.get_data_parallel_world_size()
        self.logger.info("effective batch size: %d" % eff_batch_size)
        dataset_train, dataset_val_ind, dataset_val_ood = self._dataset_func_new()
        self.logger.info(dataset_train)
        self.logger.info(dataset_val_ind)
        self.logger.info(dataset_val_ood)

        sampler_train = FinetuneDistSampler(
            dataset_train,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
            batch_size=self.args.batch_size,
            acc_grad=self.args.accum_iter,
            seed=self.args.seed,
            length_clustering=self.args.length_clustering,
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler_train,
            collate_fn=lambda batch: tuple(zip(*batch)),
            drop_last=True,
        )
        
        sampler_val_ind = FinetuneDistSampler(
            dataset_val_ind,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
            batch_size=self.args.batch_size,
            acc_grad=self.args.accum_iter,
            seed=self.args.seed,
            length_clustering=self.args.length_clustering,
        )
        dataloader_val_ind = torch.utils.data.DataLoader(
            dataset_val_ind,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler_val_ind,
            collate_fn=lambda batch: tuple(zip(*batch)),
            drop_last=True,
        )
        
        sampler_val_ood = FinetuneDistSampler(
            dataset_val_ood,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
            batch_size=self.args.batch_size,
            acc_grad=self.args.accum_iter,
            seed=self.args.seed,
            length_clustering=self.args.length_clustering,
        )
        dataloader_val_ood = torch.utils.data.DataLoader(
            dataset_val_ood,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler_val_ood,
            collate_fn=lambda batch: tuple(zip(*batch)),
            drop_last=True,
        )

        return dataset_train, sampler_train, dataloader_train, dataset_val_ind, sampler_val_ind, dataloader_val_ind, dataset_val_ood, sampler_val_ood, dataloader_val_ood

    @abstractmethod
    def _item_processor_func(self) -> ItemProcessorBase:
        raise NotImplementedError

    def _dataset_func_new(self):
        item_processor = self._item_processor_func()
        dataset_train = FinetuneConversationDataset(
            self.args.data_config_train, item_processor=item_processor, cache_on_disk=self.args.cache_ann_on_disk
        )
        dataset_val_ind = FinetuneConversationDataset(
            self.args.data_config_val_ind, item_processor=item_processor, cache_on_disk=self.args.cache_ann_on_disk
        )
        dataset_val_ood = FinetuneConversationDataset(
            self.args.data_config_val_ood, item_processor=item_processor, cache_on_disk=self.args.cache_ann_on_disk
        )
        return dataset_train, dataset_val_ind, dataset_val_ood
    
    def _dataset_func(self):
        item_processor = self._item_processor_func()
        dataset = FinetuneConversationDataset(
            self.args.data_config, item_processor=item_processor, cache_on_disk=self.args.cache_ann_on_disk
        )
        return dataset

    def resume(self, resume_path: str):
        """
        Note: model ckpt is not loaded here because _model_func should already have met the resume path as init path
        """

        def _load_optimizer():
            opt_state_world_size = len(
                [x for x in os.listdir(resume_path) if x.startswith("optimizer.") and x.endswith(".pth")]
            )
            assert opt_state_world_size == dist.get_world_size(), (
                f"Resuming from a checkpoint with unmatched world size "
                f"({dist.get_world_size()} vs. {opt_state_world_size}) "
                f"is currently not supported."
            )
            self.logger.info(f"Resuming optimizer states from: {self.args.resume_path}")
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(
                        resume_path,
                        f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
                    ),
                    map_location="cpu",
                )
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.args.lr
                param_group["weight_decay"] = self.args.wd

        _load_optimizer()
        self.logger.info("Optimizer resume complete")

        resume_epoch, resume_iteration = util.ckpt.split_ckpt_str_into_epoch_iter(resume_path.split("/")[-1])

        if resume_iteration is None:
            self.start_epoch = resume_epoch + 1
            self.start_iter = 0
        else:
            self.start_epoch = resume_epoch
            self.start_iter = resume_iteration + 1

        self.logger.info(f"resume to epoch {self.start_epoch} iter {self.start_iter}")

        additional_rank_specific = os.path.join(
            resume_path, f"additional.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth"
        )
        if os.path.exists(additional_rank_specific):
            additional_rank_specific = torch.load(additional_rank_specific, map_location="cpu")
            if "metric_logger" in additional_rank_specific:
                self.metric_logger_to_resume = additional_rank_specific["metric_logger"]
                self.logger.info("metric logger resumed")


    def run_with_eval_awm(self):
        self.logger.info(f"Start training for {self.args.epochs} epochs")
        start_time = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.dataloader_train.sampler.set_epoch(epoch, self.start_iter)  # todo rename set_epoch

            train_stats = self.train_one_epoch_awm(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            if epoch % self.args.save_interval == 0 or epoch + 1 == self.args.epochs:
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    max_keep=self.args.ckpt_max_keep,
                )

            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log_train.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            self.start_iter = 0
            self.metric_logger_to_resume = None
            
            val_stats_ind = self.val_one_epoch_awm_ind(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            log_stats = {**{f"val_{k}": v for k, v in val_stats_ind.items()}, "epoch": epoch}
            
            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log_eval_ind.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
            self.start_iter = 0
            self.metric_logger_to_resume = None
            
            val_stats_ood = self.val_one_epoch_awm_ood(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            log_stats = {**{f"val_{k}": v for k, v in val_stats_ood.items()}, "epoch": epoch}

            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log_eval_ood.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Training time {}".format(total_time_str))
    
    def run_with_eval_awm_w(self):
        self.logger.info(f"Start training for {self.args.epochs} epochs")
        start_time = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.dataloader_train.sampler.set_epoch(epoch, self.start_iter)  # todo rename set_epoch

            train_stats = self.train_one_epoch_awm_w(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            if epoch % self.args.save_interval == 0 or epoch + 1 == self.args.epochs:
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    max_keep=self.args.ckpt_max_keep,
                )

            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log_train.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            self.start_iter = 0
            self.metric_logger_to_resume = None
            
            val_stats_ind = self.val_one_epoch_awm_ind_w(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            log_stats = {**{f"val_{k}": v for k, v in val_stats_ind.items()}, "epoch": epoch}
            
            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log_eval_ind.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
            self.start_iter = 0
            self.metric_logger_to_resume = None
            
            val_stats_ood = self.val_one_epoch_awm_ood_w(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            log_stats = {**{f"val_{k}": v for k, v in val_stats_ood.items()}, "epoch": epoch}

            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log_eval_ood.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Training time {}".format(total_time_str))
    

    def process_lists(self, examples, labels):
        if len(examples) != len(labels):
            raise ValueError("examples and labels must have the same length")

        for i in range(len(examples)):
            if len(examples[i]) != len(labels[i]):
                raise ValueError(f"Mismatch in length at index {i}")

            new_example = []
            new_label = []
            skip = False

            for j in range(len(examples[i])):
                if examples[i][j] == 8197:
                    skip = True
                elif examples[i][j] == 8196:
                    skip = False
                    continue

                if not skip:
                    new_example.append(examples[i][j])
                    new_label.append(labels[i][j])

            examples[i] = new_example
            labels[i] = new_label

        return examples, labels

    
    def train_one_epoch_awm(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        self.model.train(True)
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        accum_iter = self.args.accum_iter
        accum_counter = 0

        self.optimizer.zero_grad()
        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_train,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):
            accum_counter = (accum_counter + 1) % accum_iter
            is_gradient_accumulation_boundary = accum_counter == 0

            examples, labels = batch_data
            if is_gradient_accumulation_boundary or data_iter_step == start_iter:
                lr_sched.adjust_learning_rate_epoch(
                    self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
                )

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, additional_loss_dict, logits, hidden_states, labels_c = self.model(input_ids=examples, labels=labels, output_hidden_states=True, training=True, ablation=self.args.ablation)
            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            effective_loss = loss / accum_iter
            
            accuracies_action, accuracies_image, l1_loss = self.calculate_accuracies(labels_c, logits)
                
            for i in range(len(accuracies_action)):
                metric_logger.update(**{f"acc_action_{i}": accuracies_action[i]})
                metric_logger.update(**{f"l1_loss_action_{i}": l1_loss[i]})
            
            for i in range(len(accuracies_image)):
                metric_logger.update(**{f"acc_image_{i}": accuracies_image[i]})
            

            with (
                self.model.no_sync()
                if self.args.data_parallel in ["sdp", "hsdp"] and not is_gradient_accumulation_boundary
                else contextlib.nullcontext()
            ):
                effective_loss.backward()
            
            if is_gradient_accumulation_boundary:
                grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                metric_logger.update(grad_norm=grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            if dist.get_rank() == 0:
                for metric_name, metric in metric_logger.meters.items():
                    metric_value = metric.value
                    # metric_value = util.dist.all_reduce_mean(metric_value)
                    if log_writer is not None:
                        log_writer.add_scalar(
                            'train_' + metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                        )
            

            # save within epoch
            n_update_per_save = self.args.save_iteration_interval // accum_iter
            if (
                is_gradient_accumulation_boundary and ((data_iter_step + 1) // accum_iter) % n_update_per_save == 0
            ) or (data_iter_step + 1 == accum_iter and epoch == 0):
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    iteration=data_iter_step,
                    additional_rank_specific={
                        "metric_logger": metric_logger,
                    },
                    max_keep=self.args.ckpt_max_keep,
                )            
            # break

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def train_one_epoch_awm_w(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        self.model.train(True)
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        accum_iter = self.args.accum_iter
        accum_counter = 0

        loss_weights = torch.ones(65536)
        loss_weights[3:8195] = 0.02

        self.optimizer.zero_grad()
        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_train,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):
            accum_counter = (accum_counter + 1) % accum_iter
            is_gradient_accumulation_boundary = accum_counter == 0

            examples, labels = batch_data
            if is_gradient_accumulation_boundary or data_iter_step == start_iter:
                lr_sched.adjust_learning_rate_epoch(
                    self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
                )

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, additional_loss_dict, logits, hidden_states, labels_c = self.model(input_ids=examples, labels=labels, output_hidden_states=True, training=True, loss_weights=loss_weights, ablation=self.args.ablation)
            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            effective_loss = loss / accum_iter
            
            accuracies_action, accuracies_image, l1_loss = self.calculate_accuracies(labels_c, logits)
                
            for i in range(len(accuracies_action)):
                metric_logger.update(**{f"acc_action_{i}": accuracies_action[i]})
                metric_logger.update(**{f"l1_loss_action_{i}": l1_loss[i]})
            
            for i in range(len(accuracies_image)):
                metric_logger.update(**{f"acc_image_{i}": accuracies_image[i]})

            with (
                self.model.no_sync()
                if self.args.data_parallel in ["sdp", "hsdp"] and not is_gradient_accumulation_boundary
                else contextlib.nullcontext()
            ):
                effective_loss.backward()

            if is_gradient_accumulation_boundary:
                grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                metric_logger.update(grad_norm=grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            if dist.get_rank() == 0:
                for metric_name, metric in metric_logger.meters.items():
                    metric_value = metric.value
                    # metric_value = util.dist.all_reduce_mean(metric_value)
                    if log_writer is not None:
                        log_writer.add_scalar(
                            'train_' + metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                        )

            # save within epoch
            n_update_per_save = self.args.save_iteration_interval // accum_iter
            if (
                is_gradient_accumulation_boundary and ((data_iter_step + 1) // accum_iter) % n_update_per_save == 0
            ) or (data_iter_step + 1 == accum_iter and epoch == 0):
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    iteration=data_iter_step,
                    additional_rank_specific={
                        "metric_logger": metric_logger,
                    },
                    max_keep=self.args.ckpt_max_keep,
                )
            
            # break

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def is_action_token(self, token):
        return token == 10004 or token == 15004

    def is_image_token(self, token):
        return token == 8196 or token == 8197
    
    
    def calculate_accuracies(self, labels_bs, logits_bs):
        accuracies_action_bs = []
        accuracies_image_bs = []
        l1_loss_bs = []
        for bs in range(labels_bs.shape[0]):
            labels = labels_bs[bs]
            logits = logits_bs[bs]
            i = 0
            accuracies_action = []
            accuracies_image = []
            l1_loss = []
            while i < len(labels):
                correct = 0
                total = 0
                if self.is_action_token(labels[i]):
                    # 找到一组action token
                    start = i
                    i += 1
                    while i < len(labels) and not self.is_action_token(labels[i]):
                        i += 1
                    end = i
                                        
                    # 计算这组action token的准确率
                    pred = torch.argmax(logits[start:end-1], dim=-1)
                    correct = (pred == labels[start+1:end]).sum().item()
                    total = end - start - 1
                                        
                    accuracies_action.append(correct/total)
                                        
                    conti_action = torch.tensor(self.decode_token_ids_to_actions(pred))
                    gt_conti_action = torch.tensor(self.decode_token_ids_to_actions(labels[start+1:end]))
                    action_l1_loss = torch.nn.functional.l1_loss(conti_action, gt_conti_action)
                    l1_loss.append(action_l1_loss)
                    
                    i += 1
                    
                elif self.is_image_token(labels[i]):
                    # 找到一组image token
                    start = i
                    i += 1
                    while i < len(labels) and not self.is_image_token(labels[i]):
                        i += 1
                    end = i
                                    
                    # 计算这组image token的准确率
                    pred = torch.argmax(logits[start:end-1], dim=-1)
                    correct = (pred == labels[start+1:end]).sum().item()
                    total = end - start - 1
                                    
                    accuracies_image.append(correct/total)
                    
                    i += 1
                    
                else:
                    i += 1
        
            accuracies_action_bs.append(accuracies_action)
            accuracies_image_bs.append(accuracies_image)
            l1_loss_bs.append(l1_loss)
        # print(accuracies_action_bs, accuracies_image_bs, l1_loss_bs)
        return self.calculate_position_averages(accuracies_action_bs), self.calculate_position_averages(accuracies_image_bs), self.calculate_position_averages(l1_loss_bs)
    
    
    def calculate_position_averages(self, data):
        max_length = 10
        sums = [0] * max_length
        counts = [0] * max_length
        
        for sublist in data:
            for i, value in enumerate(sublist):
                sums[max_length - len(sublist) + i] += value
                counts[max_length - len(sublist) + i] += 1
        
        averages = [sum / count if count > 0 else None for sum, count in zip(sums, counts)]
        return averages
    
    
    def val_one_epoch_awm_ind(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        # self.model.train(False)
        self.model.eval()
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_val_ind,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):

            examples, labels = batch_data

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, additional_loss_dict, logits, hidden_states, labels_c = self.model(input_ids=examples, labels=labels, output_hidden_states=True, training=True, ablation=self.args.ablation)            
            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
            metric_logger.update(grad_norm=grad_norm)
            torch.cuda.synchronize()
            
            # if self.global_rank == 0:
            #     print('examples: ', examples, 'labels: ', labels, 'c_loss: ', c_loss, 'add_loss: ', add_loss, 'grad_norm: ', grad_norm)
                        
            accuracies_action, accuracies_image, l1_loss = self.calculate_accuracies(labels_c, logits)
                
            for i in range(len(accuracies_action)):
                metric_logger.update(**{f"acc_action_{i}": accuracies_action[i]})
                metric_logger.update(**{f"l1_loss_action_{i}": l1_loss[i]})
            
            for i in range(len(accuracies_image)):
                metric_logger.update(**{f"acc_image_{i}": accuracies_image[i]})

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            if dist.get_rank() == 0:
                for metric_name, metric in metric_logger.meters.items():
                    metric_value = metric.value
                    # metric_value = util.dist.all_reduce_mean(metric_value)
                    if log_writer is not None:
                        log_writer.add_scalar(
                            'train_' + metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                        )

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def val_one_epoch_awm_ind_w(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        # self.model.train(False)
        self.model.eval()
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        loss_weights = torch.ones(65536)
        loss_weights[3:8195] = 0.02

        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_val_ind,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):

            examples, labels = batch_data

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, additional_loss_dict, logits, hidden_states, labels_c = self.model(input_ids=examples, labels=labels, output_hidden_states=True, training=True, loss_weights=loss_weights, ablation=self.args.ablation)            
            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
            metric_logger.update(grad_norm=grad_norm)
            torch.cuda.synchronize()
            
            # if self.global_rank == 0:
            #     print('examples: ', examples, 'labels: ', labels, 'c_loss: ', c_loss, 'add_loss: ', add_loss, 'grad_norm: ', grad_norm)
                        
            accuracies_action, accuracies_image, l1_loss = self.calculate_accuracies(labels_c, logits)
                
            for i in range(len(accuracies_action)):
                metric_logger.update(**{f"acc_action_{i}": accuracies_action[i]})
                metric_logger.update(**{f"l1_loss_action_{i}": l1_loss[i]})
            
            for i in range(len(accuracies_image)):
                metric_logger.update(**{f"acc_image_{i}": accuracies_image[i]})

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            if dist.get_rank() == 0:
                for metric_name, metric in metric_logger.meters.items():
                    metric_value = metric.value
                    # metric_value = util.dist.all_reduce_mean(metric_value)
                    if log_writer is not None:
                        log_writer.add_scalar(
                            'train_' + metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                        )

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    
    def val_one_epoch_awm_ood(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        # self.model.train(False)
        self.model.eval()
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_val_ood,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):

            examples, labels = batch_data

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, additional_loss_dict, logits, hidden_states, labels_c = self.model(input_ids=examples, labels=labels, output_hidden_states=True, training=True, ablation=self.args.ablation)
            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
            metric_logger.update(grad_norm=grad_norm)
            torch.cuda.synchronize()
            
            # if self.global_rank == 0:
            #     print('examples: ', examples, 'labels: ', labels, 'c_loss: ', c_loss, 'add_loss: ', add_loss, 'grad_norm: ', grad_norm)
                        
            accuracies_action, accuracies_image, l1_loss = self.calculate_accuracies(labels_c, logits)
                
            for i in range(len(accuracies_action)):
                metric_logger.update(**{f"acc_action_{i}": accuracies_action[i]})
                metric_logger.update(**{f"l1_loss_action_{i}": l1_loss[i]})
            
            for i in range(len(accuracies_image)):
                metric_logger.update(**{f"acc_image_{i}": accuracies_image[i]})

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            if dist.get_rank() == 0:
                for metric_name, metric in metric_logger.meters.items():
                    metric_value = metric.value
                    # metric_value = util.dist.all_reduce_mean(metric_value)
                    if log_writer is not None:
                        log_writer.add_scalar(
                            'train_' + metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                        )

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def val_one_epoch_awm_ood_w(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        # self.model.train(False)
        self.model.eval()
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        loss_weights = torch.ones(65536)
        loss_weights[3:8195] = 0.02

        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_val_ood,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):

            examples, labels = batch_data

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, additional_loss_dict, logits, hidden_states, labels_c = self.model(input_ids=examples, labels=labels, output_hidden_states=True, training=True, loss_weights=loss_weights, ablation=self.args.ablation)
            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
            metric_logger.update(grad_norm=grad_norm)
            torch.cuda.synchronize()
            
            # if self.global_rank == 0:
            #     print('examples: ', examples, 'labels: ', labels, 'c_loss: ', c_loss, 'add_loss: ', add_loss, 'grad_norm: ', grad_norm)
                        
            accuracies_action, accuracies_image, l1_loss = self.calculate_accuracies(labels_c, logits)
                
            for i in range(len(accuracies_action)):
                metric_logger.update(**{f"acc_action_{i}": accuracies_action[i]})
                metric_logger.update(**{f"l1_loss_action_{i}": l1_loss[i]})
            
            for i in range(len(accuracies_image)):
                metric_logger.update(**{f"acc_image_{i}": accuracies_image[i]})

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            if dist.get_rank() == 0:
                for metric_name, metric in metric_logger.meters.items():
                    metric_value = metric.value
                    # metric_value = util.dist.all_reduce_mean(metric_value)
                    if log_writer is not None:
                        log_writer.add_scalar(
                            'train_' + metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                        )

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def val_one_epoch_g(
        self,
        epoch=0,
        start_iter=0,
        log_writer=None,
        metric_logger=None,
    ):
        # self.model.train(False)
        self.model.eval()
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_val_ind,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):

            examples, labels = batch_data

            examples, res_new = self.split_sublists(examples)
            # print(labels.shape)
            # import pdb; pdb.set_trace()
            for i in range(len(labels)):
                g_imgs = []
                all_imgs = self.extract_subsequences(labels[i])
                for img in all_imgs:
                    g_img = self.item_processor_action.decode_image(img)
                    g_imgs.append(g_img)
                self.save_imgs(g_imgs)
                import pdb; pdb.set_trace()

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                generation_config = GenerationConfig(
                    max_new_tokens=8192,
                    max_length=self.model.config.max_position_embeddings,
                    temperature=1,
                    top_k=None,
                    do_sample=False,
                    eos_token_id=[8710],
                )

                logits_processor = self.create_logits_processor()

                generation_result = None
                past_key_values = None
                
                for input_id_t in res_new:
                    if generation_result is not None:
                        input_id_t = torch.cat((generation_result,input_id_t), dim=-1)
                    # generation_result, past_key_values = self.model.generate_img(
                    #     input_id_t, generation_config, past_key_values
                    # )
                    generation_result, _ = self.model.generate_img(
                        input_id_t, generation_config
                    )
                    # break

            for i in range(generation_result.shape[0]):
                g_imgs = []
                all_imgs = self.extract_subsequences(generation_result[i].cpu().tolist())
                for img in all_imgs:
                    g_img = self.item_processor_action.decode_image(img)
                    g_imgs.append(g_img)
                self.save_imgs(g_imgs)
                import pdb; pdb.set_trace()
            torch.cuda.synchronize()
    
    def save_imgs(self, g_imgs):
        # 指定保存图像的文件夹路径
        output_folder = "output_dis_bair/g_imgs_t"
        os.makedirs(output_folder, exist_ok=True)  # 创建文件夹（如果不存在）

        # 保存每张图像到文件夹
        for i, img in enumerate(g_imgs):
            img_path = os.path.join(output_folder, f"image_{i+1:03d}.png")  # 文件名格式：image_001.png
            img.save(img_path)
            print(f"Saved: {img_path}")

        # 将所有图像合并为一个动图
        gif_path = os.path.join(output_folder, "animation.gif")
        g_imgs[0].save(
            gif_path,
            save_all=True,              # 保存所有帧
            append_images=g_imgs[1:],   # 附加剩余的图像
            duration=500,               # 每帧之间的延迟时间（毫秒）
            loop=0                      # 循环次数（0 表示无限循环）
        )
        print(f"Saved GIF: {gif_path}")
            
    
    def split_sublists(self, data):
        result = []
        
        for sublist in data:
            # 初始化临时变量
            new_sublists = []
            start_index = 0
            
            # 找到第一个以 15004 结尾的部分
            try:
                first_15004_index = sublist.index(8710)
                new_sublists.append(sublist[:first_15004_index + 1])
                start_index = first_15004_index + 1
            except ValueError:
                # 如果没有找到 15004，直接跳过该子列表
                continue
            
            # 找到后续以 10004 开头到 15004 结尾的部分
            while start_index < len(sublist):
                try:
                    # 找到下一个 10004 的位置
                    next_10004_index = sublist.index(10004, start_index)
                    # 找到对应的 15004 的位置
                    next_15004_index = sublist.index(8710, next_10004_index)
                    # 提取这部分并添加到结果中
                    new_sublists.append(sublist[next_10004_index:next_15004_index + 1])
                    # 更新起始索引
                    start_index = next_15004_index + 1
                except ValueError:
                    # 如果找不到 10004 或 15004，结束循环
                    break
            
            # 将当前子列表的所有拆分结果添加到最终结果中
            result.append(new_sublists)
        
        res_new = []
        for i in range(len(result[0])):
            list_t = []
            for j in range(len(result)):
                list_t.append(result[j][i])
            res_new.append(torch.tensor(list_t, device=self.model.device))
        
        return result, res_new
    
    def extract_subsequences(self, lst):
        result = []  # 用于存储符合条件的子序列
        start_index = None  # 记录子序列的起始位置

        for i, num in enumerate(lst):
            if num == 8197:  # 找到子序列的起始位置
                start_index = i
            elif num == 8196 and start_index is not None:  # 找到子序列的结束位置
                result.append(lst[start_index:i + 1])  # 提取子序列并加入结果
                start_index = None  # 重置起始位置

        return result
    

    def decode_token_ids_to_actions(self, dis_action):
        bins = np.linspace(-1, 1, 256)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        discretized_actions = dis_action - 1 - 10004
        discretized_actions = np.clip(discretized_actions.cpu().numpy() - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
        return bin_centers[discretized_actions]
    
    
    def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10):
        logits_processor = LogitsProcessorList()

        cfg_processor = LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(
            guidance_scale=cfg,
            model=self.model,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            patch_size=32,
        )

        candidate_processor = MultiModalLogitsProcessor(
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            patch_size=32,
            voc_size=65536,
        )

        topk_processor = InterleavedTopKLogitsWarper(
            image_top_k=image_top_k,
            text_top_k=text_top_k,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        )

        logits_processor.append(cfg_processor)
        logits_processor.append(candidate_processor)
        logits_processor.append(topk_processor)

        return logits_processor

class LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        image_start_token_id,
        image_end_token_id,
        image_next_line_token_id,
        patch_size,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context_backup = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": transformers.DynamicCache() if use_cache else None,
            "first_pass": True,
        }
        self.unconditional_context = None

        self.nums_image_start_tokens = None

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

    def get_unconditional_logits(self, input_ids, image_start_token_id_index):

        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, image_start_token_id_index:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if num_image_start_tokens == num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            self.unconditional_context = None
            return scores

        elif num_image_start_tokens == num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            if new_token_num >= 2:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2

                if self.unconditional_context is None:
                    self.unconditional_context = copy.deepcopy(self.unconditional_context_backup)

                if self.guidance_scale == 1.0:
                    return scores

                unconditional_logits = self.get_unconditional_logits(input_ids, self.image_start_token_id_index)[:, -1]

                scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
                return scores_processed

        else:
            print("Something wrong in the decoding process.")

        return scores


class MultiModalLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        patch_size=None,
        voc_size=None,
    ):
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

        self.vocab_list = [i for i in range(voc_size)]
        self.image_token_list = [i for i in range(4, 8195 + 1)]
        self.suppress_tokens = torch.tensor(
            [x for x in self.vocab_list if x not in self.image_token_list], device="cuda"
        )

        self.vocab_tensor = torch.arange(voc_size, device="cuda")
        self.suppress_token_mask = torch.isin(self.vocab_tensor, self.suppress_tokens)
        self.new_line_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_next_line_token_id], device="cuda")
        )
        self.eos_image_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_end_token_id], device="cuda")
        )

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # print(self.num_image_start_tokens, self.num_image_end_tokens)

        if self.num_image_start_tokens == self.num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            return scores

        elif self.num_image_start_tokens == self.num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0]
                print(self.image_start_token_id_index)
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()

            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            # print(f"num new tokens: {new_token_num}")
            if new_token_num >= 2:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    # print(f"h_grids: {h_grids}, w_grids: {w_grids}")
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2
                    print(f"h_latent_dim: {self.h_latent_dim}, w_latent_dim: {self.w_latent_dim}")

                tokens = input_ids[0][self.image_start_token_id_index + 3 :]
                if (len(tokens) + 1) % (self.w_latent_dim + 1) == 0:
                    new_line_constrained_scores = torch.full_like(scores, -math.inf)
                    new_line_constrained_scores[:, self.image_next_line_token_id] = 0
                    print(f"new line: {len(tokens)+1}")
                    return new_line_constrained_scores
                elif (len(tokens) + 1) == (self.w_latent_dim + 1) * self.h_latent_dim + 1:
                    eos_image_constrained_scores = torch.full_like(scores, -math.inf)
                    eos_image_constrained_scores[:, self.image_end_token_id] = 0
                    print(f"eos image: {len(tokens)+1}")
                    return eos_image_constrained_scores
                elif (len(tokens) + 1) % (self.w_latent_dim + 1) != 0:
                    image_constrained_scores = torch.where(self.suppress_token_mask, -float("inf"), scores)
                    return image_constrained_scores
        else:
            print("Something wrong in the decoding process.")

        return scores


class InterleavedTopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements. Often used together
    with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].
    """

    def __init__(
        self,
        image_top_k: int,
        text_top_k: int,
        image_start_token_id=None,
        image_end_token_id=None,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(text_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`text_top_k` has to be a strictly positive integer, but is {text_top_k}")
        if not isinstance(image_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`image_top_k` has to be a strictly positive integer, but is {image_top_k}")

        self.image_top_k = max(image_top_k, min_tokens_to_keep)
        self.text_top_k = max(text_top_k, min_tokens_to_keep)
        self.filter_value = filter_value

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if self.num_image_start_tokens == self.num_image_end_tokens + 1:
            top_k = min(self.image_top_k, scores.size(-1))
        else:
            top_k = min(self.text_top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
