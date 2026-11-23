"""Pack SGLang's exact nucleus IDs without per-row host synchronization."""

import logging
import sys

import torch

_applied = False
_original = None
_top_p_filter_rows = None
_top_p_keep_mask_sorted = None


def _ensure_sglang_helpers():
    global _original, _top_p_filter_rows, _top_p_keep_mask_sorted
    if _original is None or _top_p_filter_rows is None or _top_p_keep_mask_sorted is None:
        from sglang.srt.layers.utils import top_p as upstream_top_p

        if _original is None:
            _original = upstream_top_p.get_top_p_token_ids_from_probs
        _top_p_filter_rows = upstream_top_p._top_p_filter_rows
        _top_p_keep_mask_sorted = upstream_top_p._top_p_keep_mask_sorted
    return _original, _top_p_filter_rows, _top_p_keep_mask_sorted


def get_top_p_token_ids_from_probs(
    probs,
    top_ks,
    top_ps,
    min_ps,
    need_top_p_sampling,
    need_min_p_sampling,
    request_mask,
):
    original, filter_rows, keep_mask_sorted = _ensure_sglang_helpers()

    if probs.shape[0] == 1:
        return original(
            probs, top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling, request_mask
        )

    rows = filter_rows(
        top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling, request_mask
    )
    keep, indices = keep_mask_sorted(
        probs, top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling
    )
    keep &= rows[:, None]
    # Boolean selection preserves row-major order. Do its dynamic allocation
    # once for the entire batch, then split using one transfer of row lengths.
    packed = indices[keep].to(torch.int32)
    counts = keep.sum(dim=-1)
    lengths = torch.where(rows, counts, -1).tolist()
    if all(length < 0 for length in lengths):
        return None
    parts = packed.split([max(length, 0) for length in lengths])
    return [part if length >= 0 else None for part, length in zip(parts, lengths, strict=True)]


def patch():
    global _applied
    if _applied:
        return

    from sglang.srt.layers import sampler
    from sglang.srt.layers.utils import top_p

    # Capture upstream references before replacing top_p attributes.
    _ensure_sglang_helpers()
    top_p.get_top_p_token_ids_from_probs = get_top_p_token_ids_from_probs
    sampler.get_top_p_token_ids_from_probs = get_top_p_token_ids_from_probs
    logprob_proc = sys.modules.get("sglang.srt.layers.logprob_processor")
    if logprob_proc is None:
        try:
            from sglang.srt.layers import logprob_processor as logprob_proc
        except ImportError:
            logprob_proc = None
    if logprob_proc is not None and hasattr(logprob_proc, "get_top_p_token_ids_from_probs"):
        logprob_proc.get_top_p_token_ids_from_probs = get_top_p_token_ids_from_probs
    _applied = True
    logging.getLogger(__name__).info("Batched top-p ID packing enabled; nucleus selection unchanged")
