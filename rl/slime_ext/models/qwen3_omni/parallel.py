from __future__ import annotations

from bisect import bisect_left

import torch


def mark_replicated_parameters(module: torch.nn.Module) -> None:
    """Mark an HF tower as replicated across Megatron TP ranks."""

    for parameter in module.parameters():
        parameter.tensor_model_parallel = False
        parameter.partition_dim = -1
        parameter.partition_stride = 1
        parameter.average_gradients_across_tp_domain = True


def vision_cp_partition(grid_thw: torch.Tensor, cp_size: int):
    """Partition contiguous whole frames by patch count, preserving media order.

    Follows Megatron-Bridge Qwen3-VL's vision_dp_when_cp: expand temporal grids,
    encode disjoint frames, gather final and every DeepStack feature in order.
    """
    frames = [(1, h, w) for t, h, w in grid_thw.tolist() for _ in range(t)]
    offsets = [0]
    for _, h, w in frames:
        offsets.append(offsets[-1] + h * w)
    cuts = [0, *(bisect_left(offsets, offsets[-1] * rank // cp_size) for rank in range(1, cp_size)), len(frames)]
    return grid_thw.new_tensor(frames).reshape(-1, 3), cuts, offsets


def normalize_visual_outputs(outputs: object) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Normalize HF Qwen3-Omni vision encoder outputs across Transformers versions."""
    def _as_feature_list(features: object) -> list[torch.Tensor]:
        if features is None:
            return []
        if isinstance(features, torch.Tensor):
            return [features]
        return list(features)

    if hasattr(outputs, "pooler_output") or hasattr(outputs, "last_hidden_state"):
        hidden_states = getattr(outputs, "pooler_output", None)
        if hidden_states is None:
            hidden_states = outputs.last_hidden_state
        deepstack_features = getattr(outputs, "deepstack_features", None)
        if deepstack_features is None:
            deepstack_features = getattr(outputs, "deepstack_feature_lists", None)
        if deepstack_features is None:
            deepstack_features = getattr(outputs, "hidden_states", None)
        return hidden_states, _as_feature_list(deepstack_features)

    if not isinstance(outputs, (tuple, list)) or len(outputs) == 0:
        raise TypeError(f"Unexpected vision encoder output type: {type(outputs)!r}")

    hidden_states = outputs[0]
    if len(outputs) == 2 and isinstance(outputs[1], (tuple, list)):
        deepstack_features = outputs[1]
    else:
        deepstack_features = outputs[1:]
    return hidden_states, _as_feature_list(deepstack_features)


def encode_frozen_vision_cp(visual, pixel_values: torch.Tensor, grid_thw: torch.Tensor, cp_group):
    """Distribute a frozen vision tower over the decoder's existing CP group."""
    if torch.is_grad_enabled() and any(parameter.requires_grad for parameter in visual.parameters()):
        raise ValueError("--vision-cp requires a frozen vision encoder, including its mergers")
    size = torch.distributed.get_world_size(cp_group)
    rank = torch.distributed.get_rank(cp_group)
    frames, cuts, offsets = vision_cp_partition(grid_thw, size)
    begin, end = cuts[rank : rank + 2]
    feature_counts = [
        (offsets[b] - offsets[a]) // visual.spatial_merge_unit for a, b in zip(cuts, cuts[1:], strict=False)
    ]
    with torch.no_grad():
        if begin < end:
            output = visual(pixel_values[offsets[begin] : offsets[end]], grid_thw=frames[begin:end])
            hidden_states, deepstack_features = normalize_visual_outputs(output)
            local = torch.stack([hidden_states, *deepstack_features], dim=1).contiguous()
        else:
            local = pixel_values.new_empty(
                (0, 1 + len(visual.deepstack_visual_indexes), visual.config.out_hidden_size)
            )
        # Empty ranks still participate. No backward collective is needed for a frozen tower.
        gathered = [local.new_empty((count, *local.shape[1:])) for count in feature_counts]
        torch.distributed.all_gather(gathered, local, group=cp_group)
        features = torch.cat(gathered, dim=0).unbind(dim=1)
    return features[0], list(features[1:])


def split_data_for_context_parallel(value: torch.Tensor, cp_size: int, cp_rank: int) -> torch.Tensor:
    """Apply Megatron's balanced zigzag CP split to a sequence-first tensor."""

    if cp_size == 1:
        return value
    if value.shape[0] % (2 * cp_size):
        raise ValueError(f"sequence length {value.shape[0]} is not divisible by 2 * CP ({2 * cp_size})")

    chunks = value.view(2 * cp_size, value.shape[0] // (2 * cp_size), *value.shape[1:])
    indices = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=value.device)
    return chunks.index_select(0, indices).flatten(0, 1)


def unpack_cp_full_tokens(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack real tokens, excluding each sample's CP padding and the tail bin."""
    max_length = max(lengths, default=0)
    unpacked = input_ids.new_zeros((len(lengths), max_length))
    for row, (start, length) in enumerate(zip(cu_seqlens.tolist(), lengths, strict=False)):
        unpacked[row, :length] = input_ids[0, start : start + length]
    mask = torch.arange(max_length, device=input_ids.device).unsqueeze(0) < torch.tensor(
        lengths, device=input_ids.device
    ).unsqueeze(1)
    return unpacked, mask


def pack_cp_position_ids(
    position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lengths: list[int],
    total_length: int,
) -> torch.Tensor:
    """Keep full packed MRoPE positions; native THD RoPE selects each CP shard."""
    result = position_ids.new_zeros((position_ids.shape[0], 1, total_length))
    boundaries = cu_seqlens.tolist()
    for row, length in enumerate(lengths):
        start, end = boundaries[row : row + 2]
        result[:, 0, start : start + length] = position_ids[:, row, :length]
        if length and end > start + length:
            result[:, 0, start + length : end] = position_ids[:, row, length - 1 : length]
    if lengths and boundaries[len(lengths)] < total_length:
        end = boundaries[len(lengths)]
        result[:, :, end:] = result[:, :, end - 1 : end]
    return result


def select_packed_cp_deepstack(
    mask: torch.Tensor | None,
    embeddings: list[torch.Tensor] | None,
    local_indices: torch.Tensor,
) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
    """Select visual rows using the same per-sample CP mapping as embeddings."""
    if mask is None or embeddings is None:
        return mask, embeddings
    local_mask = mask.index_select(-1, local_indices)
    feature_rows = mask.reshape(-1).long().cumsum(0) - 1
    local_feature_rows = feature_rows.index_select(0, local_indices)[local_mask.reshape(-1)]
    return local_mask, [features.index_select(0, local_feature_rows) for features in embeddings]


def shard_deepstack_for_model_parallel(
    mask: torch.Tensor | None,
    embeddings: list[torch.Tensor] | None,
    *,
    tp_size: int,
    tp_rank: int,
    cp_size: int,
    cp_rank: int,
    sequence_parallel: bool,
) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
    """Keep DeepStack features aligned with the local CP/SP token shard."""

    if mask is None or embeddings is None:
        return mask, embeddings
    if not sequence_parallel:
        tp_size, tp_rank = 1, 0

    part_count = tp_size * (2 * cp_size if cp_size > 1 else 1)
    if part_count == 1:
        return mask, embeddings
    if mask.shape[-1] % part_count:
        raise ValueError(f"sequence length {mask.shape[-1]} is not divisible by {part_count} model-parallel parts")

    if cp_size > 1:
        first = list(range(cp_rank * tp_size, (cp_rank + 1) * tp_size))
        mirror_rank = 2 * cp_size - cp_rank - 1
        local_parts = first + list(range(mirror_rank * tp_size, (mirror_rank + 1) * tp_size))
    else:
        local_parts = list(range(tp_size))
    parts_per_tp = len(local_parts) // tp_size
    local_parts = local_parts[tp_rank * parts_per_tp : (tp_rank + 1) * parts_per_tp]

    mask_parts = mask.chunk(part_count, dim=-1)
    feature_lengths = torch.stack([part.sum(-1) for part in mask_parts], dim=-1)
    feature_offsets = [0, *feature_lengths.flatten().cumsum(0).tolist()]
    feature_slices = []
    for batch_index in range(mask.shape[0]):
        for part_index in local_parts:
            offset_index = batch_index * part_count + part_index
            feature_slices.append(slice(feature_offsets[offset_index], feature_offsets[offset_index + 1]))

    local_embeddings = [torch.cat([features[index] for index in feature_slices]) for features in embeddings]
    local_mask = torch.cat([mask_parts[index] for index in local_parts], dim=-1)
    return local_mask, local_embeddings
