"""Megatron SFT entrypoint for contracted Agent shared-media data with fast packing lengths."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from megatron.core import mpu

# Swift's torchrun launches this file by path, so src/ may not yet be on sys.path.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swift.dataset import DatasetLoader
from swift.megatron.arguments import MegatronSftArguments
from swift.megatron.pipelines.train.sft import MegatronSft
from swift.utils import get_logger, is_master, safe_ddp_context

from omnicoach_sft.media.parquet_pool import ParquetFilePool
from omnicoach_sft.train.dataset import (
    AgentSharedMediaDataset,
    AgentSharedMediaRegistry,
    expand_agent_manifests,
    expand_agent_shared_turn_parts,
    load_agent_parquet_parts,
    prepare_agent_length_dataset,
    resolve_agent_shared_source_root,
    validate_agent_training_artifact,
)
from omnicoach_sft.train.lengths import load_pack_length_profile, prepare_source_lengths
from omnicoach_sft.train.loss import install_exact_agent_suffix_loss_scale
from omnicoach_sft.train.trainer import OmniMegatronTrainer

logger = get_logger()
# Set this to a shared filesystem to reuse length arrays across runs.
_CACHE_DIR_ENV = "OMNI_PACK_LENGTH_CACHE_DIR"


@dataclass
class OmniMegatronSftArguments(MegatronSftArguments):
    """Project-local Megatron arguments missing from the pinned Swift release."""

    omni_pack_length_profile: str | None = None
    overlap_cpu_optimizer_d2h_h2d: bool = False
    omni_tp_data_dispatch: bool = True

    def _init_megatron_args(self):
        # BaseArguments already resolved model_revision into model_dir. The
        # pinned Swift Megatron initializer repeats metadata loading without
        # forwarding that revision; reuse the resolved directory for that call.
        model = self.model
        model_dir = getattr(self, "model_dir", None)
        if self.model_revision and model_dir:
            self.model = model_dir
        try:
            return super()._init_megatron_args()
        finally:
            self.model = model


@dataclass(frozen=True)
class LoadedSource:
    """One source plus the resolved contract source path."""

    dataset: Any
    contract_source: str


def _write_overflow_report(output_dir: str, records: list[dict[str, Any]]) -> None:
    if not is_master():
        return
    path = Path(output_dir) / "pack_length_overflow.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


class FastMegatronSft(MegatronSft):
    args_class = OmniMegatronSftArguments

    def _parse_args(self, args):
        args = super()._parse_args(args)
        if not args.omni_pack_length_profile:
            raise ValueError("--omni_pack_length_profile is required; training reads only contracted Agent Parquet sources")
        self._pack_length_profile = load_pack_length_profile(args.omni_pack_length_profile)
        if self._pack_length_profile.model_family == "qwen2_5_omni":
            # Swift reads this when constructing its model, after argument parsing.
            if os.environ.get("ENABLE_AUDIO_OUTPUT", "0").lower() not in {"0", "false"}:
                raise ValueError("Qwen2.5-Omni trace SFT requires ENABLE_AUDIO_OUTPUT=0")
            os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
        return args

    def __init__(self, args=None) -> None:
        max_open_files = max(1, int(os.environ.get("OMNI_MEDIA_PARQUET_MAX_OPEN_FILES", "32")))
        media_file_pool = ParquetFilePool(max_open_files)
        self._agent_shared_media_registry = AgentSharedMediaRegistry(file_pool=media_file_pool)
        self._has_agent_shared_sources = False
        self._pack_length_overflow_records: list[dict[str, Any]] = []
        super().__init__(args)
        if getattr(self.template, "version", None) == "omni_v2_5":
            from omnicoach_sft.train.qwen25_omni import configure_qwen25_trace_template

            configure_qwen25_trace_template(self.template)
            logger.info("Qwen2.5-Omni trace policy: dataset-owned system prompt and complete audio")
        install_exact_agent_suffix_loss_scale(self.template)
        self._validate_dataset_owned_system_prompt()
        if self._pack_length_profile.cache_dir is None:
            raw_cache_dir = os.environ.get(_CACHE_DIR_ENV)
            if raw_cache_dir:
                cache_dir = Path(raw_cache_dir).expanduser().resolve()
            else:
                cache_dir = Path(tempfile.gettempdir()) / "omni_sft_pack_length"
            if is_master():
                logger.info(f"Pack-length cache: {cache_dir}")
            self._pack_length_profile = replace(self._pack_length_profile, cache_dir=cache_dir)

    def prepare_trainer(self):
        if self.args.task_type != "causal_lm":
            return super().prepare_trainer()
        return OmniMegatronTrainer(
            self.args,
            self.template,
            enable_tp_data_dispatch=(self._has_agent_shared_sources and self.args.omni_tp_data_dispatch),
        )

    def _validate_dataset_owned_system_prompt(self) -> None:
        if getattr(self.args, "system", None) is not None:
            raise ValueError(
                "system prompt is owned by the versioned dataset contract; remove --system/system from training args"
            )
        default_system = getattr(self.template.template_meta, "default_system", None)
        if default_system is not None:
            raise ValueError(
                "expected the dataset-owned Swift template to have default_system=None; "
                "refusing a template that could inject a system prompt into rows that omit one"
            )

    def _validate_fast_path(self) -> None:
        args = self.args
        if args.streaming:
            raise ValueError("fast pack-length estimation does not support streaming datasets")
        if args.split_dataset_ratio != 0:
            raise ValueError("fast pack-length estimation currently requires split_dataset_ratio=0")
        if args.dataset_shuffle:
            raise ValueError("fast pack-length estimation currently requires dataset_shuffle=false")
        if getattr(args, "interleave_prob", None):
            raise ValueError("fast pack-length estimation does not support interleaved datasets")

        profile = self._pack_length_profile
        expected_version = "omni_v2_5" if profile.model_family == "qwen2_5_omni" else "omni_v3"
        if getattr(self.template, "version", None) != expected_version:
            raise ValueError(f"pack-length model_family={profile.model_family} does not match the Swift template")

    def _load_source(self, source: str, *, dataset_kwargs: dict[str, Any]) -> LoadedSource:
        agent_shared_root = resolve_agent_shared_source_root(source)
        if agent_shared_root is None:
            raise ValueError(f"not an Agent shared-media source root: {source}")
        self._has_agent_shared_sources = True
        # Megatron groups are initialized by the argument adapter. TP peers
        # only need task statistics for packing; the batch source owns media.
        tp_dispatch = (
            self.args.task_type == "causal_lm"
            and self.args.omni_tp_data_dispatch
            and self.args.tensor_model_parallel_size > 1
        )
        if not tp_dispatch or mpu.get_tensor_model_parallel_rank() == 0:
            self._agent_shared_media_registry.store_for_root(agent_shared_root)

        if dataset_kwargs.get("streaming"):
            raise ValueError("contracted Agent Parquet loading does not support streaming")
        agent_parts = expand_agent_shared_turn_parts(agent_shared_root)
        num_proc = int(dataset_kwargs.get("num_proc") or 1)
        logger.info(
            f"Loading {len(agent_parts)} Agent Parquet shards as one train split "
            f"with num_proc={min(max(1, num_proc), len(agent_parts))}: {source}"
        )
        # Match Swift's local-path loader, but give HF all Parquet paths in
        # one data_files collection instead of invoking Swift once per path.
        # safe_ddp_context makes one local master populate each node-local
        # cache before the remaining ranks open it.
        with safe_ddp_context(None, True):
            dataset = load_agent_parquet_parts(
                agent_parts,
                num_proc=num_proc,
            )
        if dataset_kwargs.get("columns"):
            raise ValueError("contracted Agent Parquet owns its column names; custom columns mapping is unsupported")
        validate_agent_training_artifact(dataset, agent_shared_root)
        return LoadedSource(dataset, str(agent_shared_root))

    def _prepare_sources(self, sources: Sequence[str], *, dataset_kwargs: dict[str, Any]) -> list[Any]:
        args = self.args
        profile = self._pack_length_profile
        packing_length = int(args.packing_length or args.max_length)
        prepared = []
        for source_index, source in enumerate(sources):
            started = time.monotonic()
            if is_master():
                logger.info(f"SFT startup: source {source_index + 1}/{len(sources)} loading: {source}")
            loaded = self._load_source(source, dataset_kwargs=dataset_kwargs)
            loaded_at = time.monotonic()
            channel = "agent" if "tools" in loaded.dataset.column_names else "base"
            # Normalize while the source still has no row-selection indices.
            # Doing this after overflow filtering would make Dataset.add_column
            # flatten the indexed view and serialize all embedded media again.
            dataset = prepare_agent_length_dataset(
                loaded.dataset,
                source=loaded.contract_source,
                channel=channel,
            )
            normalized_at = time.monotonic()
            dataset, overflow_records = prepare_source_lengths(
                dataset,
                source=loaded.contract_source,
                template=self.template,
                profile=profile,
                model_identity=str(args.model),
                packing_length=packing_length,
                num_proc=args.dataset_num_proc,
                row_resolver=None,
            )
            prepared.append(dataset)
            self._pack_length_overflow_records.extend(overflow_records)
            if is_master():
                finished = time.monotonic()
                logger.info(
                    f"SFT startup: source {source_index + 1}/{len(sources)} ready: "
                    f"load/index={loaded_at - started:.1f}s, normalize={normalized_at - loaded_at:.1f}s, "
                    f"lengths/sync={finished - normalized_at:.1f}s, total={finished - started:.1f}s"
                )
        return prepared

    def _get_dataset(self):
        self._validate_fast_path()
        args = self.args
        train_sources = expand_agent_manifests(args.dataset or [])
        val_sources = expand_agent_manifests(args.val_dataset or [])
        dataset_kwargs = args.get_dataset_kwargs()
        dataset_kwargs["remove_unused_columns"] = False
        train_dataset = DatasetLoader.concat_datasets(
            self._prepare_sources(train_sources, dataset_kwargs=dataset_kwargs)
        )
        validation_kwargs = {key: value for key, value in dataset_kwargs.items() if key != "interleave_prob"}
        val_dataset = DatasetLoader.concat_datasets(
            self._prepare_sources(val_sources, dataset_kwargs=validation_kwargs)
        )
        _write_overflow_report(args.output_dir, self._pack_length_overflow_records)
        return train_dataset, val_dataset

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        self._save_val_dataset(val_dataset)
        for name, dataset in (("train", train_dataset), ("validation", val_dataset)):
            if dataset is not None and "lengths" not in dataset.column_names:
                raise ValueError(f"fast pack-length path did not add a lengths column to the {name} dataset")
        if self._has_agent_shared_sources:
            train_dataset = (
                AgentSharedMediaDataset(train_dataset, self._agent_shared_media_registry)
                if train_dataset is not None
                else None
            )
            val_dataset = (
                AgentSharedMediaDataset(val_dataset, self._agent_shared_media_registry)
                if val_dataset is not None
                else None
            )
        return train_dataset, val_dataset


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    FastMegatronSft().main()


if __name__ == "__main__":
    main()
