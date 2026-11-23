"""Shared Omni hooks for CPU mRoPE and frozen-encoder feature reuse.

TE 2.18 native THD CP reorder and MCore 0.20 native vectorized THD RoPE
handle packed CP reordering and RoPE directly without Python monkey-patches.
"""

import os
from functools import wraps

from . import cpu_mrope
from .encoder_cache import VALID_ENCODER_FEATURE_MODES, EncoderFeatures

_buffer_reserved = False
_encoder_features = None
_rollout_id = -1
_in_train = False
_seen_log_prob_prefixes: set[str] = set()
_flush_installed = False


def _configured_deepep_num_sms() -> int | None:
    """Return an explicitly configured positive DeepEP SM count, or None if unset."""
    raw = os.environ.get("SLIME_OMNI_DEEPEP_NUM_SMS")
    if raw is None or not raw.strip():
        return None
    try:
        num_sms = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"SLIME_OMNI_DEEPEP_NUM_SMS must be a positive integer, got {raw!r}"
        ) from exc
    return num_sms if num_sms > 0 else None


def _reserve_deepep(args, model):
    """Opt-in DeepEP SM reservation when SLIME_OMNI_DEEPEP_NUM_SMS > 0 and managers exist."""
    global _buffer_reserved
    if _buffer_reserved:
        return

    num_sms = _configured_deepep_num_sms()
    if num_sms is None:
        _buffer_reserved = True
        return

    managers = [
        module.token_dispatcher._comm_manager
        for part in model
        for module in part.modules()
        if getattr(getattr(module, "token_dispatcher", None), "_comm_manager", None) is not None
    ]
    if not managers:
        _buffer_reserved = True
        return

    from megatron.core.transformer.moe import fused_a2a

    groups = {id(manager.group): manager.group for manager in managers}
    fused_a2a.set_deepep_num_sms(num_sms)
    for group in groups.values():
        fused_a2a.get_buffer(group, args.hidden_size * 2)
    _buffer_reserved = True


def _begin(args, model, phase):
    global _encoder_features
    _reserve_deepep(args, model)
    if os.environ.get("SLIME_OMNI_CPU_MROPE", "1") == "1":
        cpu_mrope.install()

    mode = os.environ.get("SLIME_OMNI_ENCODER_FEATURE_MODE", "cache").strip().lower()
    if mode not in VALID_ENCODER_FEATURE_MODES:
        raise ValueError(
            "SLIME_OMNI_ENCODER_FEATURE_MODE must be one of "
            f"{sorted(VALID_ENCODER_FEATURE_MODES)}, got {mode!r}"
        )
    if mode in {"cache", "measure"}:
        if _encoder_features is None:
            _encoder_features = EncoderFeatures(
                model,
                mode,
                float(os.environ.get("SLIME_OMNI_ENCODER_CACHE_GIB", "12")),
                log_dir=os.environ.get("SLIME_OMNI_ENCODER_FEATURE_LOG_DIR"),
            )
        _encoder_features.begin(phase)


def before_log_prob(args, model, store_prefix):
    """Advance rollout_id reliably even when ref_log_probs (--use-kl-loss) is disabled."""
    global _rollout_id, _in_train
    if _rollout_id < 0:
        _rollout_id = int(getattr(args, "start_rollout_id", 0) or 0)
        _in_train = False
        _seen_log_prob_prefixes.clear()
    elif _in_train or store_prefix in _seen_log_prob_prefixes:
        _rollout_id += 1
        _in_train = False
        _seen_log_prob_prefixes.clear()
    _seen_log_prob_prefixes.add(store_prefix)
    _begin(args, model, {"phase": f"{store_prefix}log_probs", "rollout_id": _rollout_id})


def before_train(args, rollout_id, step_id, model, optimizer, scheduler):
    global _flush_installed, _rollout_id, _in_train
    _rollout_id = rollout_id
    _in_train = True
    _seen_log_prob_prefixes.clear()
    _begin(args, model, {"phase": "actor_train", "rollout_id": rollout_id, "step_id": step_id})
    if not _flush_installed:
        original = optimizer.prepare_grads

        @wraps(original)
        def prepare_grads():
            result = original()
            if _encoder_features is not None:
                _encoder_features.flush()
            return result

        optimizer.prepare_grads = prepare_grads
        _flush_installed = True
