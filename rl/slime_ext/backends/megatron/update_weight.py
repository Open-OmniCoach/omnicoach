"""Override for ``slime.backends.megatron_utils.update_weight.common``.

Rebased onto upstream Slime ``79aa41a8`` (preserving ``partition_sections``,
MCore 0.19+ ``mtp_model_layer``, and the ``tp_size == 1`` zero-copy fast path)
while adding Qwen3-Omni multimodal tower handling, optional ``middle_path``
nesting, and ``local_experts.<idx>`` naming.
"""

import inspect
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist

try:
    from megatron.core import mpu
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
except ImportError:  # pragma: no cover - allows pure-CPU unit testing without Megatron
    mpu = None
    get_transformer_layer_offset = None

from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix
from slime.utils.types import ParamInfo

_applied = False

_MULTIMODAL_TOWER_PREFIXES = (
    "thinker.visual.",
    "thinker.vision_model.",
    "thinker.audio_model.",
    "thinker.video_model.",
)


def _is_multimodal_tower_param(name: str) -> bool:
    name = strip_param_name_prefix(name)
    return name.startswith(_MULTIMODAL_TOWER_PREFIXES)


def _merge_param_partitions(name, partitions, partition_dim, partition_sections=None):
    if partition_sections is not None:
        tp_size = len(partitions)
        if any(size % tp_size for size in partition_sections):
            raise ValueError("Projection sections must be divisible by TP")
        local_sections = tuple(size // tp_size for size in partition_sections)
        split = [part.split(local_sections, dim=partition_dim) for part in partitions]
        result = torch.cat([part for section in zip(*split, strict=True) for part in section], dim=partition_dim)
        result.partition_sections = partition_sections
        return result
    if ("linear_fc1.weight" in name or "linear_fc1.bias" in name) and not _is_multimodal_tower_param(name):
        split = [part.chunk(2, dim=0) for part in partitions]
        partitions = [part[0] for part in split] + [part[1] for part in split]
    # Megatron grouped MoE exposes its FC2 partition dimension as zero.
    if "linear_fc2.weight" in name and partition_dim == 0 and not _is_multimodal_tower_param(name):
        partition_dim = 1
    return torch.cat(partitions, dim=partition_dim)


def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    """All-gather TP-sharded param to full tensor with tp_size==1 fast path."""
    if "expert_bias" in name:
        return param

    if _is_multimodal_tower_param(name) and not hasattr(param, "tensor_model_parallel"):
        return param.data

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    data = param.data
    if (sections := getattr(param, "partition_sections", None)) is not None:
        data.partition_sections = sections
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return data

    if ".experts." in name:
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    assert param.partition_stride == 1 or (
        param.partition_stride == 2 and "linear_fc1" in name
    ), "partition_stride != 1 is not supported"
    return _merge_param_partitions(name, param_partitions, partition_dim, getattr(param, "partition_sections", None))


def all_gather_params_async(param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]]) -> list[torch.Tensor]:
    """Parallel TP all-gather for multiple params with tp_size==1 zero-copy fast path."""
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None))
        elif (
            not getattr(param, "tensor_model_parallel", False)
            or getattr(param, "parallel_mode", None) == "duplicated"
        ):
            gather_tasks.append((info, param.data, None, None, None))
        else:
            if ".experts." in info.name:
                tp_size = mpu.get_expert_tensor_parallel_world_size()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()

            if tp_size == 1:
                gather_tasks.append((info, param.data, None, None, None))
                continue

            if ".experts." in info.name:
                tp_group = mpu.get_expert_tensor_parallel_group()
            else:
                tp_group = mpu.get_tensor_model_parallel_group()

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim))
            handles.append(handle)

    for handle in handles:
        handle.wait()

    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        if handle is None:
            param = direct_param
        else:
            assert partition_dim is not None, "partition_stride != 1 is not supported"
            param = _merge_param_partitions(
                info.name,
                param_partitions,
                partition_dim,
                info.attrs.get("partition_sections") if hasattr(info, "attrs") and info.attrs else None,
            )

        sections = info.attrs.get("partition_sections") if hasattr(info, "attrs") and info.attrs else None
        if sections is not None:
            param.partition_sections = sections

        gathered_params.append(param)

    return gathered_params


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def _maybe_get_cpu_backup(x: torch.Tensor) -> torch.Tensor:
    from torch_memory_saver import torch_memory_saver

    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x, zero_copy=True)) is not None:
        return cpu_tensor

    return x


def _named_params_and_buffers_vanilla(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():
            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace,
    model: Sequence[torch.nn.Module],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (global_name, param/buffer) with PP/EP-consistent names.

    Supports optional nested prefix (``middle_path``), ``local_experts.<idx>``,
    and upstream MCore 0.19+ ``mtp_model_layer`` expert paths.
    """
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    expert_offset = ep_rank * args.num_experts // ep_size if getattr(args, "num_experts", None) else 0

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for is_buffer, named_tensors in (
            (False, model_module.named_parameters),
            (True, model_module.named_buffers),
        ):
            for name, tensor in named_tensors():
                if is_buffer and "expert_bias" not in name:
                    continue
                if not name.startswith("module.module."):
                    name = "module." + name

                layer_kind = "decoder"
                match = re.match(r"module\.(?:module\.)?(?:(.*)\.)?decoder\.layers\.(\d+)\.(.+)", name)
                if match is None and not is_buffer:
                    layer_kind = "mtp"
                    match = re.match(r"module\.(?:module\.)?(?:(.*)\.)?mtp\.layers\.(\d+)\.(.+)", name)
                if match is None:
                    yield name, tensor
                    continue

                middle_path, layer_idx, rest = match.groups()
                middle_path = f"{middle_path}." if middle_path else ""
                if layer_kind == "decoder":
                    layer_idx = int(layer_idx) + layer_offset

                if layer_kind == "mtp":
                    # Supports both GPT MTP (transformer_layer.mlp.experts.*) and
                    # MCore 0.19+ hybrid MTP (mtp_model_layer.layers.{k}.mlp.experts.*).
                    mtp_expert = re.fullmatch(r"(.*?)mlp\.experts\.(.+)\.(weight|bias)(\d+)", rest)
                    if mtp_expert is None:
                        yield name, tensor
                        continue
                    layer_path, projection, param_type, expert_idx = mtp_expert.groups()
                    rest = f"{layer_path}mlp.experts.{projection}.{param_type}{int(expert_idx) + expert_offset}"
                    yield f"module.module.{middle_path}mtp.layers.{layer_idx}.{rest}", tensor
                    continue

                if not is_buffer:
                    expert = re.match(r"mlp\.experts\.(.+)\.(weight|bias)(\d+)", rest)
                    if expert:
                        projection, param_type, expert_idx = expert.groups()
                        rest = f"mlp.experts.{projection}.{param_type}{int(expert_idx) + expert_offset}"
                    else:
                        local_expert = re.match(r"mlp\.experts\.local_experts\.(\d+)\.(.+)", rest)
                        if local_expert:
                            expert_idx, projection = local_expert.groups()
                            rest = f"mlp.experts.local_experts.{int(expert_idx) + expert_offset}.{projection}"

                yield f"module.module.{middle_path}{layer_kind}.layers.{layer_idx}.{rest}", tensor


_REBIND_TARGETS = {
    "all_gather_param": all_gather_param,
    "all_gather_params_async": all_gather_params_async,
    "named_params_and_buffers": named_params_and_buffers,
}

# Sibling modules that do `from .common import <name>` at import time.
_CONSUMER_MODULES = (
    "slime.backends.megatron_utils.actor",
    "slime.backends.megatron_utils.hf_checkpoint_saver",
    "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
    "slime.backends.megatron_utils.update_weight.update_weight_from_distributed_delta",
    "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct",
)


def patch():
    global _applied
    if _applied:
        return
    import sys

    import slime.backends.megatron_utils.update_weight.common as mod

    for attr, fn in _REBIND_TARGETS.items():
        setattr(mod, attr, fn)
    mod._MULTIMODAL_TOWER_PREFIXES = _MULTIMODAL_TOWER_PREFIXES
    mod._is_multimodal_tower_param = _is_multimodal_tower_param
    mod._merge_param_partitions = _merge_param_partitions
    mod._named_params_and_buffers_global = _named_params_and_buffers_global
    mod._named_params_and_buffers_vanilla = _named_params_and_buffers_vanilla
    mod._maybe_get_cpu_backup = _maybe_get_cpu_backup

    for consumer_name in _CONSUMER_MODULES:
        consumer = sys.modules.get(consumer_name)
        if consumer is None:
            continue
        for attr, fn in _REBIND_TARGETS.items():
            if hasattr(consumer, attr):
                setattr(consumer, attr, fn)

    _applied = True
