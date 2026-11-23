"""Response-only packing, masking, log-prob, and entropy helpers for Megatron loss.

Gathers response rows into one compact ``[sum(R), V_local]`` tensor before
running Slime's vocab-parallel logprob/entropy kernel. Includes empty-index
CUDA 64-bit backward protection and lazy ``TensorRef`` top-p support.
"""

from __future__ import annotations

import logging
import sys
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

try:
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import get_logits_and_tokens_offset_with_cp
except ImportError:  # pragma: no cover - allows pure-CPU unit testing without Megatron
    mpu = None
    get_logits_and_tokens_offset_with_cp = None

_IMPORTED_MPU = mpu

logger = logging.getLogger(__name__)

_logged_active_shape = False
_original_get_log_probs_and_entropy: Callable[..., Any] | None = None
_upstream_loss = None


def _current_mpu():
    facade = sys.modules.get("slime_ext.backends.megatron.loss")
    if facade is not None and getattr(facade, "mpu", _IMPORTED_MPU) is not _IMPORTED_MPU:
        return getattr(facade, "mpu", None)
    return mpu


def bind_upstream(loss_mod: Any, original_get_log_probs_and_entropy: Callable[..., Any]) -> None:
    """Bind upstream Slime loss module references during ``loss.patch()``."""
    global _upstream_loss, _original_get_log_probs_and_entropy
    _upstream_loss = loss_mod
    _original_get_log_probs_and_entropy = original_get_log_probs_and_entropy


def _response_only_enabled(args: Namespace) -> bool:
    return bool(getattr(args, "response_only_log_probs", False))


def _response_only_lm_head_enabled(args: Namespace) -> bool:
    return bool(getattr(args, "response_only_lm_head", False))


def _split_rows(values: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
    """Split the leading dimension while preserving explicit zero-length rows."""
    result = []
    offset = 0
    for length in lengths:
        result.append(values[offset : offset + length])
        offset += length
    if offset != values.size(0):
        raise ValueError(f"Response split consumed {offset} rows, but tensor has {values.size(0)}")
    return result


def _response_row_indices_and_targets(
    logits: torch.Tensor,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (len(unconcat_tokens) == len(total_lengths) == len(response_lengths)):
        raise ValueError(
            "unconcat_tokens, total_lengths, and response_lengths must have equal lengths: "
            f"{len(unconcat_tokens)}, {len(total_lengths)}, {len(response_lengths)}"
        )

    row_indices = []
    targets = []
    sequence_offset = 0
    for sample_index, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=True)
    ):
        if tokens.ndim != 1 or tokens.numel() != total_length:
            raise ValueError(
                f"Sample {sample_index} token shape {tuple(tokens.shape)} does not match total_length={total_length}"
            )
        if response_length < 0 or response_length > total_length:
            raise ValueError(
                f"Sample {sample_index} has invalid response_length={response_length} for total_length={total_length}"
            )

        prompt_length = total_length - response_length
        if response_length > 0 and prompt_length < 1:
            raise ValueError(f"Sample {sample_index} needs at least one prompt token for next-token prediction")

        if response_length > 0:
            start = sequence_offset + prompt_length - 1
            end = sequence_offset + total_length - 1
            row_indices.append(torch.arange(start, end, device=logits.device, dtype=torch.long))
            targets.append(tokens[-response_length:].to(device=logits.device, dtype=torch.long))
        sequence_offset += total_length

    if sequence_offset > logits.size(0):
        raise ValueError(
            f"Packed samples require {sequence_offset} rows, but logits only contain {logits.size(0)}"
        )

    if row_indices:
        return torch.cat(row_indices, dim=0), torch.cat(targets, dim=0)
    return (
        torch.empty((0,), dtype=torch.long, device=logits.device),
        torch.empty((0,), dtype=torch.long, device=logits.device),
    )


def _flat_response_targets(
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    device: torch.device,
) -> torch.Tensor:
    if not (len(unconcat_tokens) == len(total_lengths) == len(response_lengths)):
        raise ValueError(
            "unconcat_tokens, total_lengths, and response_lengths must have equal lengths: "
            f"{len(unconcat_tokens)}, {len(total_lengths)}, {len(response_lengths)}"
        )
    targets = []
    for sample_index, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=True)
    ):
        if tokens.ndim != 1 or tokens.numel() != total_length:
            raise ValueError(
                f"Sample {sample_index} token shape {tuple(tokens.shape)} does not match total_length={total_length}"
            )
        if response_length < 0 or response_length > total_length:
            raise ValueError(
                f"Sample {sample_index} has invalid response_length={response_length} for total_length={total_length}"
            )
        if response_length > 0 and total_length == response_length:
            raise ValueError(f"Sample {sample_index} needs at least one prompt token for next-token prediction")
        if response_length:
            targets.append(tokens[-response_length:].to(device=device, dtype=torch.long))
    if targets:
        return torch.cat(targets, dim=0)
    return torch.empty((0,), dtype=torch.long, device=device)


def _cp_response_rows_and_targets(
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[list[int]]]:
    """Map packed CP rows to targets and original response support positions."""
    row_indices, targets, local_lengths, response_positions = [], [], [], []
    sequence_offset = 0
    for tokens, total_length, response_length in zip(
        unconcat_tokens, total_lengths, response_lengths, strict=True
    ):
        chunk_size, chunks, logit_offsets, token_offsets = get_logits_and_tokens_offset_with_cp(
            total_length, response_length
        )
        prompt_length = total_length - response_length
        positions = []
        for half, ((logit_start, logit_end), (token_start, token_end)) in enumerate(
            zip(logit_offsets, token_offsets, strict=True)
        ):
            if token_end == token_start:
                continue
            start = sequence_offset + half * chunk_size + logit_start - chunks[half][0]
            row_indices.append(
                torch.arange(start, start + logit_end - logit_start, device=device, dtype=torch.long)
            )
            targets.append(tokens[token_start:token_end].to(device=device, dtype=torch.long))
            positions.extend(range(token_start - prompt_length, token_end - prompt_length))
        local_lengths.append(len(positions))
        response_positions.append(positions)
        sequence_offset += 2 * chunk_size
    empty = torch.empty((0,), dtype=torch.long, device=device)
    return (
        torch.cat(row_indices) if row_indices else empty,
        torch.cat(targets) if targets else empty,
        local_lengths,
        response_positions,
    )


def _contiguous_position_spans(positions: list[int]) -> list[tuple[int, int, int]]:
    """Group response positions into contiguous ``(local_start, resp_start, length)`` runs."""
    if not positions:
        return []
    spans: list[tuple[int, int, int]] = []
    local_start = 0
    resp_start = positions[0]
    run_len = 1
    for local_idx in range(1, len(positions)):
        if positions[local_idx] == resp_start + run_len:
            run_len += 1
        else:
            spans.append((local_start, resp_start, run_len))
            local_start = local_idx
            resp_start = positions[local_idx]
            run_len = 1
    spans.append((local_start, resp_start, run_len))
    return spans


def _build_response_topp_keep_mask(
    response_lengths: list[int],
    vocab_local: int,
    device: torch.device,
    top_p_token_ids: list[Any],
    top_p_token_offsets: list[Any],
    response_positions: list[list[int]] | None = None,
) -> torch.Tensor:
    """Build the native replay mask with contiguous-span reads (TensorRef-safe)."""
    active_mpu = _current_mpu()
    tp_rank = active_mpu.get_tensor_model_parallel_rank() if active_mpu is not None else 0
    vocab_start = tp_rank * vocab_local
    vocab_end = vocab_start + vocab_local
    keep = torch.ones((sum(response_lengths), vocab_local), dtype=torch.bool, device=device)
    rows_chunks: list[torch.Tensor] = []
    cols_chunks: list[torch.Tensor] = []

    response_base = 0
    for sample_index, response_length in enumerate(response_lengths):
        if sample_index < len(top_p_token_ids) and sample_index < len(top_p_token_offsets):
            ids = top_p_token_ids[sample_index]
            offsets = top_p_token_offsets[sample_index]
            if hasattr(offsets, "load"):
                offsets = offsets.load()
            offsets_list = offsets.tolist() if torch.is_tensor(offsets) else list(offsets)
            max_recorded = max(len(offsets_list) - 1, 0)
            if response_positions is None:
                valid_len = min(response_length, max_recorded)
                spans = [(0, 0, valid_len)] if valid_len > 0 else []
            else:
                valid_positions = [p for p in response_positions[sample_index] if p < max_recorded]
                spans = _contiguous_position_spans(valid_positions)

            for local_start, resp_start, span_len in spans:
                resp_end = resp_start + span_len
                keep[response_base + local_start : response_base + local_start + span_len].fill_(False)
                base_offset = offsets_list[resp_start]
                end_offset = offsets_list[resp_end]
                if end_offset <= base_offset:
                    continue
                # Slice once per contiguous span so lazy TensorRef reads stay bounded.
                span_ids = torch.as_tensor(ids[base_offset:end_offset], dtype=torch.long, device="cpu")
                if span_ids.numel() == 0:
                    continue
                counts = torch.diff(torch.tensor(offsets_list[resp_start : resp_end + 1], dtype=torch.long))
                span_rows = torch.repeat_interleave(
                    torch.arange(span_len, dtype=torch.long), counts
                )
                in_vocab = (span_ids >= vocab_start) & (span_ids < vocab_end)
                if bool(in_vocab.any().item()):
                    rows_chunks.append(span_rows[in_vocab] + (response_base + local_start))
                    cols_chunks.append(span_ids[in_vocab] - vocab_start)
        response_base += response_length

    if rows_chunks:
        all_rows = torch.cat(rows_chunks, dim=0).to(device)
        all_cols = torch.cat(cols_chunks, dim=0).to(device)
        keep[all_rows, all_cols] = True
    return keep


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    top_p_token_ids: list[Any] | None = None,
    top_p_token_offsets: list[Any] | None = None,
):
    """Run one vocabulary loss pass over all response rows in the microbatch."""
    global _logged_active_shape

    if not _response_only_enabled(args):
        if _original_get_log_probs_and_entropy is None:
            raise RuntimeError("slime_ext loss patch has not been initialized")
        return _original_get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            non_loss_data=non_loss_data,
            top_p_token_ids=top_p_token_ids,
            top_p_token_offsets=top_p_token_offsets,
        )

    assert non_loss_data
    active_mpu = _current_mpu()
    cp_size = active_mpu.get_context_parallel_world_size() if active_mpu is not None else 1
    if getattr(args, "allgather_cp", False):
        raise NotImplementedError("--response-only-log-probs supports packed two-chunk CP, not allgather CP")
    if logits.dtype != torch.float32 or logits.ndim != 3 or logits.size(0) != 1:
        raise ValueError(f"Expected float32 logits shaped [1, T, V_local], got {logits.dtype} {tuple(logits.shape)}")

    full_logits = logits.squeeze(0)
    local_response_lengths = response_lengths
    response_positions = None
    if cp_size > 1:
        row_indices, compact_targets, local_response_lengths, response_positions = _cp_response_rows_and_targets(
            unconcat_tokens, total_lengths, response_lengths, full_logits.device
        )
        if _response_only_lm_head_enabled(args):
            expected_rows = sum(local_response_lengths)
            if full_logits.size(0) != expected_rows:
                raise ValueError(
                    f"Response-only LM head produced {full_logits.size(0)} rows, expected {expected_rows} CP-local rows"
                )
            compact_logits = full_logits
        else:
            # Guard empty row_indices against PyTorch 64-bit CUDA index_select
            # backward zero-sized grid crashes (matches upstream loss.py).
            compact_logits = full_logits.index_select(0, row_indices) if row_indices.numel() else full_logits[:0]
    elif _response_only_lm_head_enabled(args):
        compact_targets = _flat_response_targets(
            unconcat_tokens,
            total_lengths,
            response_lengths,
            full_logits.device,
        )
        expected_rows = sum(response_lengths)
        if full_logits.size(0) != expected_rows:
            raise ValueError(
                f"Response-only LM head produced {full_logits.size(0)} rows, expected {expected_rows}"
            )
        compact_logits = full_logits
    else:
        row_indices, compact_targets = _response_row_indices_and_targets(
            full_logits,
            unconcat_tokens,
            total_lengths,
            response_lengths,
        )
        # Guard empty row_indices against PyTorch 64-bit CUDA index_select backward crash.
        compact_logits = full_logits.index_select(0, row_indices) if row_indices.numel() else full_logits[:0]

    rollout_temperature = getattr(args, "rollout_temperature", 1.0)
    if rollout_temperature != 1.0:
        compact_logits = compact_logits / rollout_temperature
    compact_logits = compact_logits.contiguous()

    top_p_keep_mask = None
    if top_p_token_ids is not None and top_p_token_offsets is not None:
        top_p_keep_mask = _build_response_topp_keep_mask(
            local_response_lengths,
            compact_logits.size(-1),
            compact_logits.device,
            top_p_token_ids,
            top_p_token_offsets,
            response_positions,
        )

    with_entropy_grad = with_entropy and getattr(args, "entropy_coef", 0.0) != 0
    if compact_logits.size(0) == 0:
        # Preserve a zero-gradient edge to the full logits. Upstream gets this
        # edge by slicing its full-row result; its generic empty-kernel branch
        # alone would return a detached tensor.
        log_probs = compact_logits.sum(dim=-1)
        entropy = compact_logits.sum(dim=-1) if with_entropy_grad else compact_logits.new_zeros((0,))
    else:
        log_probs, entropy = _upstream_loss.calculate_log_probs_and_entropy(
            compact_logits,
            compact_targets,
            active_mpu.get_tensor_model_parallel_group(),
            with_entropy=with_entropy,
            with_entropy_grad=with_entropy_grad,
            chunk_size=args.log_probs_chunk_size,
            log_prob_keep_mask=top_p_keep_mask,
        )
        log_probs = log_probs.squeeze(-1)

    if not _logged_active_shape:
        full_rows = sum(total_lengths) if _response_only_lm_head_enabled(args) else full_logits.size(0)
        response_rows = compact_logits.size(0)
        logger.info(
            "response-only log-probs active: full_rows=%d response_rows=%d response_fraction=%.6f",
            full_rows,
            response_rows,
            response_rows / max(full_rows, 1),
        )
        _logged_active_shape = True

    result = {"log_probs": _split_rows(log_probs, local_response_lengths)}
    if with_entropy:
        if entropy is None:
            raise RuntimeError("Entropy was requested but the vocabulary kernel returned None")
        result["entropy"] = _split_rows(entropy, local_response_lengths)
    return torch.empty((0,), device=logits.device), result


__all__ = [
    "_build_response_topp_keep_mask",
    "_response_only_enabled",
    "_response_only_lm_head_enabled",
    "bind_upstream",
    "get_log_probs_and_entropy",
]
