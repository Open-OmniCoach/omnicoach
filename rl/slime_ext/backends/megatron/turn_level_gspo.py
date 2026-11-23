"""Turn-Level GSPO (TG-GSPO) geometric-mean importance ratio and loss helpers.

Computes per-turn geometric-mean log-ratios using ``turn_ids`` (``-1`` for
masked/non-assistant tokens, ``0..K-1`` for assistant turns), broadcasts
per-turn ``rho_k = exp(log_rho_k)`` over each turn's tokens, applies standard
PPO clipping per turn, and reduces with optional ``turn_weights`` (or
``loss_masks``) under both CP=1 and CP>1. Standard sequence-level GSPO and
GRPO remain untouched when disabled.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

try:
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import (
        all_gather_with_cp,
        slice_log_prob_with_cp,
    )
except ImportError:  # pragma: no cover - allows pure-CPU unit testing without Megatron
    mpu = None
    all_gather_with_cp = None
    slice_log_prob_with_cp = None

_original_policy_loss_function: Callable[..., Any] | None = None
_original_compute_advantages_and_returns: Callable[..., Any] | None = None
_upstream_loss = None


def bind_upstream(
    loss_mod: Any,
    original_policy_loss_function: Callable[..., Any],
    original_compute_advantages_and_returns: Callable[..., Any],
) -> None:
    """Bind upstream Slime loss module references during ``loss.patch()``."""
    global _upstream_loss, _original_policy_loss_function, _original_compute_advantages_and_returns
    _upstream_loss = loss_mod
    _original_policy_loss_function = original_policy_loss_function
    _original_compute_advantages_and_returns = original_compute_advantages_and_returns


def _turn_level_gspo_enabled(args: Namespace) -> bool:
    return bool(getattr(args, "use_turn_level_gspo", False))


def get_response_cp_offsets(
    total_length: int,
    response_length: int,
    cp_size: int,
    cp_rank: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return response-space ``((s0, e0), (s1, e1))`` zigzag CP slices.

    Matches ``slime.backends.megatron_utils.cp_utils.slice_log_prob_with_cp``
    exactly without requiring Megatron MPU initialization.
    """
    if cp_size < 1 or not (0 <= cp_rank < cp_size):
        raise ValueError(f"Invalid CP config: cp_rank={cp_rank}, cp_size={cp_size}")
    if cp_size == 1:
        return (0, response_length), (0, 0)
    prompt_length = total_length - response_length
    chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = (
        (2 * cp_size - cp_rank - 1) * chunk_size,
        (2 * cp_size - cp_rank) * chunk_size,
    )
    spans: list[tuple[int, int]] = []
    for chunk in (chunk_0, chunk_1):
        lo = max(chunk[0], prompt_length - 1)
        hi = min(chunk[1], total_length - 1)
        if lo < hi:
            spans.append((lo - (prompt_length - 1), hi - (prompt_length - 1)))
        else:
            spans.append((0, 0))
    return spans[0], spans[1]


def slice_response_tensor_with_cp(
    tensor: torch.Tensor,
    total_length: int,
    response_length: int,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Slice a full ``[response_length, ...]`` tensor to this CP rank's zigzag rows."""
    if len(tensor) != response_length:
        raise ValueError(
            f"Response tensor length {len(tensor)} != response_length={response_length} "
            f"(total_length={total_length})"
        )
    if cp_size == 1:
        return tensor
    (s0, e0), (s1, e1) = get_response_cp_offsets(total_length, response_length, cp_size, cp_rank)
    return torch.cat([tensor[s0:e0], tensor[s1:e1]], dim=0)


def _default_slice_fn() -> Callable[[torch.Tensor, int, int], torch.Tensor]:
    if mpu is not None and slice_log_prob_with_cp is not None:
        return slice_log_prob_with_cp
    return lambda tensor, _total_len, _resp_len: tensor


def compute_turn_level_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    turn_ids: list[torch.Tensor],
    total_lengths: list[int] | None = None,
    response_lengths: list[int] | None = None,
    slice_fn: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute Turn-Level GSPO ``ppo_kl = -log_rho_k`` broadcast per valid turn.

    For each sample and each valid turn ``k >= 0``:
        ``log_rho_k = sum_{t in turn k}(valid_t * (logp_t - old_logp_t)) / max(1, sum_{t in turn k} valid_t)``
    where ``valid_t = (loss_mask_t > 0) & (turn_ids_t == k)``.

    Returns a concatenated 1D tensor aligned to ``local_log_probs`` (works under
    both ``CP = 1`` and ``CP > 1``; gradients flow differentiably into
    ``full_log_probs``).
    """
    if not (
        len(full_log_probs)
        == len(full_old_log_probs)
        == len(local_log_probs)
        == len(loss_masks)
        == len(turn_ids)
    ):
        raise ValueError("Turn-Level GSPO inputs must have identical sample counts")

    slicer = slice_fn or _default_slice_fn()
    ppo_kl_list: list[torch.Tensor] = []

    for idx, (full_lp, full_old_lp, local_lp, loss_mask, sample_turn_ids) in enumerate(
        zip(full_log_probs, full_old_log_probs, local_log_probs, loss_masks, turn_ids, strict=True)
    ):
        resp_len = full_lp.shape[0] if response_lengths is None else response_lengths[idx]
        total_len = resp_len + 1 if total_lengths is None else total_lengths[idx]

        loss_mask_t = loss_mask.to(device=full_lp.device)
        turn_ids_t = sample_turn_ids.to(device=full_lp.device, dtype=torch.long)
        if not (
            full_lp.shape == full_old_lp.shape == loss_mask_t.shape == turn_ids_t.shape
            and full_lp.ndim == 1
        ):
            raise ValueError(
                f"Sample {idx} shape mismatch: full_lp={tuple(full_lp.shape)}, "
                f"full_old_lp={tuple(full_old_lp.shape)}, loss_mask={tuple(loss_mask_t.shape)}, "
                f"turn_ids={tuple(turn_ids_t.shape)}"
            )

        valid_mask = (loss_mask_t > 0) & (turn_ids_t >= 0)
        token_kl = full_old_lp - full_lp  # ppo_kl convention: ratio = exp(-ppo_kl) = exp(new - old)

        if bool(valid_mask.any().item()):
            num_turns = int(turn_ids_t[valid_mask].max().item()) + 1
            safe_turn_ids = turn_ids_t.clamp(min=0)
            mask_f = valid_mask.to(dtype=full_lp.dtype)
            turn_kl_sums = full_lp.new_zeros((num_turns,)).scatter_add(0, safe_turn_ids, token_kl * mask_f)
            turn_counts = full_lp.new_zeros((num_turns,)).scatter_add(0, safe_turn_ids, mask_f)
            turn_mean_kl = turn_kl_sums / torch.clamp_min(turn_counts, 1.0)
            sample_kl_full = torch.where(
                valid_mask,
                turn_mean_kl[safe_turn_ids],
                full_lp.new_zeros(full_lp.shape),
            )
        else:
            # Keep a zero-gradient edge to full_lp when a sample has no valid turns.
            sample_kl_full = full_lp * 0.0

        if slice_fn is not None or sample_kl_full.shape[0] != local_lp.shape[0]:
            sample_kl_local = slicer(sample_kl_full, total_len, resp_len)
        else:
            sample_kl_local = sample_kl_full

        if sample_kl_local.shape != local_lp.shape:
            raise ValueError(
                f"Sample {idx} sliced turn KL shape {tuple(sample_kl_local.shape)} "
                f"!= local_log_prob shape {tuple(local_lp.shape)}"
            )
        ppo_kl_list.append(sample_kl_local)

    if not ppo_kl_list:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(ppo_kl_list, dim=0)


def get_turn_level_gspo_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    turn_ids: list[torch.Tensor],
    turn_weights: list[torch.Tensor] | None = None,
    sample_denoms: list[torch.Tensor] | torch.Tensor | None = None,
    calculate_per_token_loss: bool = False,
    slice_fn: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
    normalize_turn_weights: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a CP-consistent loss reducer honoring ``turn_ids`` and optional ``turn_weights``.

    Contract for ``turn_weights``:
    - By default (``normalize_turn_weights=False``), ``turn_weights`` are
      pre-normalized across each rollout/trajectory so that active weights sum
      to ``1.0`` for a packed full-trajectory sample, or sum to ``omega_k`` on
      each expanded per-turn sample with ``sum_k omega_k == 1.0`` across sibling
      turn samples of the same rollout. Per-sample division by
      ``weights_full.sum()`` is NOT applied (which would cancel ``omega_k`` and
      multiply multi-turn trajectories by their turn count).
    - Under ``calculate_per_token_loss=True``, when ``sample_denoms`` (e.g.
      ``rollout_mask_sums`` = total active tokens ``M_r`` of the rollout) is
      supplied, each sample is scaled by ``M_r`` so total weight matches
      ``num_tokens`` while preserving exact relative turn shares ``omega_k``.
    - Set ``normalize_turn_weights=True`` only for unnormalized per-sample weights.
    """
    slicer = slice_fn or _default_slice_fn()
    local_weights: list[torch.Tensor] = []
    local_lengths: list[int] = []
    denoms: list[torch.Tensor] = []

    for idx, (total_len, resp_len, loss_mask, sample_turn_ids) in enumerate(
        zip(total_lengths, response_lengths, loss_masks, turn_ids, strict=True)
    ):
        device = loss_mask.device
        turn_ids_t = sample_turn_ids.to(device=device, dtype=torch.long)
        valid = (loss_mask > 0) & (turn_ids_t >= 0)

        if turn_weights is not None:
            weights_full = turn_weights[idx].to(device=device, dtype=torch.float32) * valid.to(torch.float32)
            if calculate_per_token_loss:
                if normalize_turn_weights:
                    weights_full = weights_full / torch.clamp_min(weights_full.sum(), 1e-8)
                if sample_denoms is not None:
                    raw_scale = sample_denoms[idx]
                    scale = torch.clamp_min(
                        raw_scale if torch.is_tensor(raw_scale) else weights_full.new_tensor(float(raw_scale)),
                        1.0,
                    )
                    weights_full = weights_full * scale
                denom = weights_full.new_tensor(1.0)
            else:
                if normalize_turn_weights:
                    denom = torch.clamp_min(weights_full.sum(), 1e-8)
                else:
                    denom = weights_full.new_tensor(1.0)
        else:
            weights_full = loss_mask.to(dtype=torch.float32) * (turn_ids_t >= 0).to(torch.float32)
            if sample_denoms is not None:
                raw_denom = sample_denoms[idx]
                denom = torch.clamp_min(
                    raw_denom if torch.is_tensor(raw_denom) else weights_full.new_tensor(float(raw_denom)),
                    1.0,
                )
            else:
                denom = torch.clamp_min(weights_full.sum(), 1.0)

        weights_local = slicer(weights_full, total_len, resp_len)
        local_weights.append(weights_local)
        local_lengths.append(int(weights_local.size(0)))
        denoms.append(denom)

    if calculate_per_token_loss:
        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            if not local_weights:
                return x.sum() * 0.0
            return sum(
                (x_i * w_i.to(device=x.device, dtype=x.dtype)).sum()
                for x_i, w_i in zip(x.split(local_lengths, dim=0), local_weights, strict=True)
            )
        return sum_of_token

    def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
        if not local_weights:
            return x.sum() * 0.0
        return sum(
            (x_i * w_i.to(device=x.device, dtype=x.dtype)).sum() / denom.to(device=x.device, dtype=x.dtype)
            for x_i, w_i, denom in zip(x.split(local_lengths, dim=0), local_weights, denoms, strict=True)
        )

    return sum_of_sample_mean


def compute_turn_level_gspo_loss(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    turn_ids: list[torch.Tensor],
    *,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
    turn_weights: list[torch.Tensor] | None = None,
    total_lengths: list[int] | None = None,
    response_lengths: list[int] | None = None,
    sample_denoms: list[torch.Tensor] | torch.Tensor | None = None,
    calculate_per_token_loss: bool = False,
    slice_fn: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
    normalize_turn_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """End-to-end Turn-Level GSPO loss helper (works on CPU PyTorch without Megatron).

    Returns ``(reduced_pg_loss, per_token_pg_loss, per_token_clipfrac, per_token_ppo_kl)``.
    """
    resp_lens = [lp.shape[0] for lp in full_log_probs] if response_lengths is None else response_lengths
    tot_lens = [r + 1 for r in resp_lens] if total_lengths is None else total_lengths

    ppo_kl = compute_turn_level_gspo_kl(
        full_log_probs=full_log_probs,
        full_old_log_probs=full_old_log_probs,
        local_log_probs=local_log_probs,
        loss_masks=loss_masks,
        turn_ids=turn_ids,
        total_lengths=tot_lens,
        response_lengths=resp_lens,
        slice_fn=slice_fn,
    )
    adv_cat = torch.cat(advantages, dim=0)
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * adv_cat
    pg_losses2 = -ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_high) * adv_cat
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        if eps_clip_c <= 1.0:
            raise ValueError(f"eps_clip_c must be > 1.0, got {eps_clip_c}")
        pg_losses3 = -eps_clip_c * adv_cat
        clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
        per_token_pg_loss = torch.where(adv_cat < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        per_token_pg_loss = clip_pg_losses1

    reducer = get_turn_level_gspo_reducer(
        total_lengths=tot_lens,
        response_lengths=resp_lens,
        loss_masks=loss_masks,
        turn_ids=turn_ids,
        turn_weights=turn_weights,
        sample_denoms=sample_denoms,
        calculate_per_token_loss=calculate_per_token_loss,
        slice_fn=slice_fn,
        normalize_turn_weights=normalize_turn_weights,
    )
    reduced_loss = reducer(per_token_pg_loss)
    return reduced_loss, per_token_pg_loss, clipfrac, ppo_kl


def compute_advantages_and_returns(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Use ``token_advantages`` directly when present; else delegate upstream."""
    token_advs = rollout_data.get("token_advantages")
    if token_advs is None or getattr(args, "custom_advantage_function_path", None) is not None:
        if _original_compute_advantages_and_returns is None:
            raise RuntimeError("slime_ext loss patch has not been initialized")
        return _original_compute_advantages_and_returns(args, rollout_data)

    if mpu is not None and not mpu.is_pipeline_last_stage():
        return

    rollout_log_probs = rollout_data.get("rollout_log_probs")
    log_probs = rollout_log_probs if getattr(args, "use_rollout_logprobs", False) else rollout_data.get("log_probs")
    ref_log_probs = rollout_data.get("ref_log_probs")
    loss_masks = rollout_data.get("loss_masks")
    response_lengths = rollout_data.get("response_lengths")
    total_lengths = rollout_data.get("total_lengths")

    if getattr(args, "kl_coef", 0) == 0 or not log_probs:
        xs = log_probs or rollout_log_probs or token_advs
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            _upstream_loss.compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]
    rollout_data["kl"] = kl

    device = kl[0].device if kl else token_advs[0].device
    advantages = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in token_advs]
    returns = [a for a in advantages]

    if getattr(args, "use_opd", False):
        _upstream_loss.apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    if getattr(args, "normalize_advantages", False):
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size() if mpu is not None else 1
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            cp_rank = mpu.get_context_parallel_rank()
            all_masks = torch.cat(
                [
                    slice_response_tensor_with_cp(m, t_len, r_len, cp_size, cp_rank)
                    for m, t_len, r_len in zip(loss_masks, total_lengths, response_lengths, strict=True)
                ]
            )
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        whitened_flat = _upstream_loss.distributed_masked_whiten(
            all_advs,
            all_masks,
            process_group=dp_cp_group,
            shift_mean=True,
        )
        advantages = list(torch.split(whitened_flat, [chunk.size(0) for chunk in advantages]))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def policy_loss_function(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch Turn-Level GSPO when ``--use-turn-level-gspo`` is set; else upstream."""
    if not _turn_level_gspo_enabled(args):
        if _original_policy_loss_function is None:
            raise RuntimeError("slime_ext loss patch has not been initialized")
        return _original_policy_loss_function(args, batch, logits, sum_of_sample_mean)

    turn_ids = batch.get("turn_ids")
    if turn_ids is None:
        raise ValueError("--use-turn-level-gspo requires 'turn_ids' in batch")
    turn_weights = batch.get("turn_weights")

    adv_list = batch.get("advantages")
    if adv_list is None:
        adv_list = batch.get("token_advantages")
    if adv_list is None:
        raise ValueError("--use-turn-level-gspo requires 'advantages' or 'token_advantages' in batch")

    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = _upstream_loss.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        **_upstream_loss.get_rollout_top_p_logprob_kwargs(args, batch),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    _upstream_loss._maybe_capture_log_probs(batch, log_probs)
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [lp.detach() for lp in log_probs]
    train_log_probs_for_tis = batch.get("log_probs") or [lp.detach() for lp in log_probs]

    full_log_probs = [
        all_gather_with_cp(lp, total_len, resp_len)
        for lp, total_len, resp_len in zip(log_probs, total_lengths, response_lengths, strict=True)
    ]
    full_old_log_probs = [
        all_gather_with_cp(old_lp, total_len, resp_len)
        for old_lp, total_len, resp_len in zip(old_log_probs, total_lengths, response_lengths, strict=True)
    ]

    if args.use_opsm:
        opsm_mask, opsm_clipfrac = _upstream_loss.compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=adv_list,
            loss_masks=batch["loss_masks"],
        )

    ppo_kl = compute_turn_level_gspo_kl(
        full_log_probs=full_log_probs,
        full_old_log_probs=full_old_log_probs,
        local_log_probs=log_probs,
        loss_masks=batch["loss_masks"],
        turn_ids=turn_ids,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        slice_fn=slice_log_prob_with_cp,
    )
    old_log_probs_cat = torch.cat(old_log_probs, dim=0)
    log_probs_cat = torch.cat(log_probs, dim=0)
    advantages_cat = torch.cat(adv_list, dim=0)

    pg_loss_per_token, pg_clipfrac_per_token = _upstream_loss.compute_policy_loss(
        ppo_kl,
        advantages_cat,
        args.eps_clip,
        args.eps_clip_high,
        eps_clip_c=args.eps_clip_c,
    )

    if args.use_opsm:
        pg_loss_per_token = pg_loss_per_token * opsm_mask

    loss_masks_for_pg = batch["loss_masks"]
    if args.get_mismatch_metrics or args.use_tis:
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean
        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"
        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss_per_token,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }
        tis_func = (
            _upstream_loss.load_function(args.custom_tis_function_path)
            if args.custom_tis_function_path is not None
            else _upstream_loss.vanilla_tis_function
        )
        pg_loss_per_token, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)
        loss_masks_for_pg = modified_response_masks

    pg_loss_reducer = get_turn_level_gspo_reducer(
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        loss_masks=loss_masks_for_pg,
        turn_ids=turn_ids,
        turn_weights=turn_weights,
        sample_denoms=batch.get("rollout_mask_sums"),
        calculate_per_token_loss=args.calculate_per_token_loss,
        slice_fn=slice_log_prob_with_cp,
    )

    pg_loss = pg_loss_reducer(pg_loss_per_token)
    pg_clipfrac = pg_loss_reducer(pg_clipfrac_per_token)
    ppo_kl_reduced = pg_loss_reducer(ppo_kl)

    entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
    entropy_loss = sum_of_sample_mean(entropy)
    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        importance_ratio = torch.exp(log_probs_cat - old_log_probs_cat) if args.use_unbiased_kl else None
        kl = _upstream_loss.compute_approx_kl(
            log_probs_cat,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if log_probs_cat.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        log_probs_to_compare = log_probs_cat if args.use_rollout_logprobs else old_log_probs_cat
        train_rollout_logprob_abs_diff = sum_of_sample_mean((log_probs_to_compare - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl_reduced.clone().detach(),
    }
    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()
    if args.get_mismatch_metrics or args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        for metric_key, metric_value in tis_metrics.items():
            reported_loss[metric_key] = sum_of_sample_mean_for_mismatch_metrics(metric_value)
    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac
    if batch.get("opd_reverse_kl"):
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


__all__ = [
    "_turn_level_gspo_enabled",
    "bind_upstream",
    "compute_advantages_and_returns",
    "compute_turn_level_gspo_kl",
    "compute_turn_level_gspo_loss",
    "get_response_cp_offsets",
    "get_turn_level_gspo_reducer",
    "policy_loss_function",
    "slice_response_tensor_with_cp",
]
