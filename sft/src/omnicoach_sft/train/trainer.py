"""Load and collate one batch per tensor-parallel group."""

from __future__ import annotations

import os
import resource
import time
from typing import Any

import pyarrow as pa
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.tensor_parallel import broadcast_data
from swift.megatron.trainers import MegatronTrainer
from swift.utils import get_logger

logger = get_logger()


class _IdleDataLoader:
    """Placeholder on TP ranks that receive batches from their group leader."""

    def __iter__(self) -> _IdleDataLoader:
        return self

    def __next__(self) -> None:
        return None


def _init_media_worker(_: int) -> None:
    """Keep native libraries inside each DataLoader worker single-threaded."""

    num_threads = max(1, int(os.environ.get("OMNI_MEDIA_WORKER_THREADS", "1")))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(num_threads)
    torch.set_num_threads(num_threads)
    pa.set_cpu_count(num_threads)
    pa.set_io_thread_count(num_threads)


def _describe_batch(batch: dict[str, Any]) -> tuple[list[tuple[str, torch.dtype]], dict[str, Any]]:
    tensor_specs: list[tuple[str, torch.dtype]] = []
    metadata: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_specs.append((key, value.dtype))
        else:
            metadata[key] = value
    return tensor_specs, metadata


def broadcast_tp_batch(batch: dict[str, Any] | None) -> dict[str, Any]:
    """Broadcast a collated batch from TP rank zero without pickling tensor payloads."""

    tp_group = mpu.get_tensor_model_parallel_group()
    is_source = mpu.get_tensor_model_parallel_rank() == 0
    source_global_rank = mpu.get_tensor_model_parallel_src_rank()

    if is_source:
        if not isinstance(batch, dict):
            raise TypeError(f"TP data source expected a dict batch, got {type(batch).__name__}")
        tensor_specs, metadata = _describe_batch(batch)
        descriptor: list[Any] = [(tensor_specs, metadata)]
    else:
        batch = {}
        descriptor = [None]
    dist.broadcast_object_list(descriptor, src=source_global_rank, group=tp_group)
    tensor_specs, metadata = descriptor[0]

    dtype_order: list[torch.dtype] = []
    for _, dtype in tensor_specs:
        if dtype not in dtype_order:
            dtype_order.append(dtype)

    result = dict(metadata)
    for dtype in dtype_order:
        keys = [key for key, key_dtype in tensor_specs if key_dtype == dtype]
        result.update(broadcast_data(keys, batch, dtype, tp_group=tp_group))
    return result


class OmniMegatronTrainer(MegatronTrainer):
    """Megatron trainer that removes duplicate TP-rank media preprocessing."""

    def __init__(self, args, template, *, enable_tp_data_dispatch: bool) -> None:
        self.enable_tp_data_dispatch = bool(
            enable_tp_data_dispatch
            and not args.streaming
            and int(args.tensor_model_parallel_size) > 1
        )
        self.slow_data_log_seconds = float(os.environ.get("OMNI_SLOW_DATA_LOG_SECONDS", "30"))
        super().__init__(args, template)

    @property
    def _is_data_loader_source(self) -> bool:
        return not self.enable_tp_data_dispatch or mpu.get_tensor_model_parallel_rank() == 0

    def _create_dataloader(self, dataset, batch_sampler):
        if not self._is_data_loader_source:
            return _IdleDataLoader()

        dataloader = super()._create_dataloader(dataset, batch_sampler)
        if self.args.dataloader_num_workers > 0:
            dataloader.worker_init_fn = _init_media_worker
        if self.enable_tp_data_dispatch:
            logger.info(
                "TP data dispatch enabled: only tensor-parallel rank 0 creates "
                "DataLoader workers and media tensors are broadcast after collation."
            )
        return dataloader

    def get_batch(self, data_iterator, vp_stage=None):
        if not self.enable_tp_data_dispatch:
            return super().get_batch(data_iterator, vp_stage)
        started = time.perf_counter()
        raw_batch = next(data_iterator) if self._is_data_loader_source else None
        loaded = time.perf_counter()
        batch = self._prepare_batch(broadcast_tp_batch(raw_batch), vp_stage)
        finished = time.perf_counter()
        if self._is_data_loader_source and finished - started >= self.slow_data_log_seconds:
            max_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            try:
                open_fds = len(os.listdir("/proc/self/fd"))
            except OSError:
                open_fds = -1
            logger.warning(
                f"Slow TP batch at iteration {self.state.iteration}: "
                f"dataloader={loaded - started:.3f}s, "
                f"broadcast_and_prepare={finished - loaded:.3f}s, "
                f"rank_max_rss={max_rss_gib:.2f}GiB, rank_open_fds={open_fds}"
            )
        return batch

    def _collect_config_info(self) -> dict[str, str]:
        return {"seq2seq_mode": "sft" if self.template.use_chat_template else "pt"}
