"""Packed-sequence helpers used by slime_ext Megatron patches."""

from __future__ import annotations

import torch

REAL_CU_SEQLENS_Q_ATTR = "_slime_ext_real_cu_seqlens_q"


def attach_real_cu_seqlens_q(packed_seq_params: object, cu_seqlens: torch.Tensor) -> None:
    setattr(packed_seq_params, REAL_CU_SEQLENS_Q_ATTR, cu_seqlens)


def get_real_cu_seqlens_q(packed_seq_params: object) -> torch.Tensor:
    cu_seqlens = getattr(packed_seq_params, REAL_CU_SEQLENS_Q_ATTR, None)
    if cu_seqlens is None:
        raise ValueError(
            "Qwen3-Omni packed forward requires real sample cu_seqlens from slime_ext get_batch; "
            f"missing {REAL_CU_SEQLENS_Q_ATTR} on packed_seq_params."
        )
    return cu_seqlens


def packed_thd_to_padded_batch(
    packed_values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pad_value: int | float = 0,
) -> torch.Tensor:
    if packed_values.dim() < 2 or packed_values.shape[0] != 1:
        raise ValueError(f"Expected THD values shaped [1, T, ...], got {tuple(packed_values.shape)}")

    offsets = [int(x) for x in cu_seqlens.tolist()]
    seqlens = [offsets[i + 1] - offsets[i] for i in range(max(len(offsets) - 1, 0))]
    batch_size = len(seqlens)
    max_seq_len = max(seqlens, default=0)
    result = packed_values.new_full((batch_size, max_seq_len, *packed_values.shape[2:]), pad_value)

    for i, seqlen in enumerate(seqlens):
        start = offsets[i]
        result[i, :seqlen] = packed_values[0, start : start + seqlen]
    return result


def padded_batch_to_packed_thd(unpacked_values: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    if unpacked_values.dim() < 2:
        raise ValueError(f"Expected padded batch values shaped [B, S, ...], got {tuple(unpacked_values.shape)}")

    offsets = [int(x) for x in cu_seqlens.tolist()]
    total_len = offsets[-1] if offsets else 0
    result = unpacked_values.new_zeros((1, total_len, *unpacked_values.shape[2:]))

    for i in range(max(len(offsets) - 1, 0)):
        start = offsets[i]
        end = offsets[i + 1]
        result[0, start:end] = unpacked_values[i, : end - start]
    return result


def packed_attention_mask(cu_seqlens: torch.Tensor, max_seq_len: int | None = None) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    if max_seq_len is None:
        max_seq_len = max((int(x) for x in seqlens.tolist()), default=0)
    positions = torch.arange(max_seq_len, device=cu_seqlens.device).unsqueeze(0)
    return positions < seqlens.unsqueeze(1)


def pad_position_ids_to_length(position_ids: torch.Tensor, total_length: int) -> torch.Tensor:
    current_length = position_ids.shape[-1]
    if current_length == total_length:
        return position_ids
    if current_length > total_length:
        raise ValueError(f"position_ids length {current_length} exceeds input length {total_length}")
    if current_length == 0:
        pad = position_ids.new_zeros((*position_ids.shape[:-1], total_length))
    else:
        pad = position_ids[..., -1:].expand(*position_ids.shape[:-1], total_length - current_length)
    return torch.cat([position_ids, pad], dim=-1)


def pack_padded_batch_position_ids(
    position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_length: int,
) -> torch.Tensor:
    packed = padded_batch_to_packed_thd(position_ids.permute(1, 2, 0), cu_seqlens).permute(2, 0, 1)
    return pad_position_ids_to_length(packed, total_length)


def count_audio_token_runs(
    input_ids: torch.Tensor,
    audio_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    counts = []
    zero = input_ids.new_zeros(1, dtype=torch.long)
    for i, row in enumerate(input_ids):
        if attention_mask is not None:
            row = row[attention_mask[i].to(row.device).bool()]
        mask = (row == audio_token_id).to(dtype=torch.long)
        padded = torch.cat([zero, mask, zero])
        diff = padded[1:] - padded[:-1]
        starts = torch.nonzero(diff == 1, as_tuple=False).flatten()
        ends = torch.nonzero(diff == -1, as_tuple=False).flatten()
        counts.append(ends - starts)

    if not counts:
        return torch.empty(0, device=input_ids.device, dtype=torch.long)
    return torch.cat(counts, dim=0).to(dtype=torch.long)
