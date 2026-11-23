"""Thin facade wiring response-only loss compaction and Turn-Level GSPO into Slime."""

from __future__ import annotations

import sys

from . import response_only_loss, turn_level_gspo
from .response_only_loss import (
    _build_response_topp_keep_mask,
    _response_only_enabled,
    _response_only_lm_head_enabled,
    get_log_probs_and_entropy,
    mpu,
)
from .turn_level_gspo import (
    _turn_level_gspo_enabled,
    compute_advantages_and_returns,
    compute_turn_level_gspo_kl,
    compute_turn_level_gspo_loss,
    get_response_cp_offsets,
    get_turn_level_gspo_reducer,
    policy_loss_function,
    slice_response_tensor_with_cp,
)

_applied = False


def patch() -> None:
    """Install the opt-in wrappers and repair upstream eager imports."""
    global _applied
    if _applied:
        return

    import slime.backends.megatron_utils.loss as loss_mod

    response_only_loss.bind_upstream(loss_mod, loss_mod.get_log_probs_and_entropy)
    turn_level_gspo.bind_upstream(
        loss_mod,
        loss_mod.policy_loss_function,
        loss_mod.compute_advantages_and_returns,
    )

    loss_mod.get_log_probs_and_entropy = get_log_probs_and_entropy
    loss_mod.policy_loss_function = policy_loss_function
    loss_mod.compute_advantages_and_returns = compute_advantages_and_returns

    for module_name in (
        "slime.backends.megatron_utils.actor",
        "slime.backends.megatron_utils.server.logprob_utils",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            if hasattr(module, "get_log_probs_and_entropy"):
                module.get_log_probs_and_entropy = get_log_probs_and_entropy
            if hasattr(module, "compute_advantages_and_returns"):
                module.compute_advantages_and_returns = compute_advantages_and_returns

    _applied = True


__all__ = [
    "_build_response_topp_keep_mask",
    "_response_only_enabled",
    "_response_only_lm_head_enabled",
    "_turn_level_gspo_enabled",
    "compute_advantages_and_returns",
    "compute_turn_level_gspo_kl",
    "compute_turn_level_gspo_loss",
    "get_log_probs_and_entropy",
    "get_response_cp_offsets",
    "get_turn_level_gspo_reducer",
    "mpu",
    "patch",
    "policy_loss_function",
    "slice_response_tensor_with_cp",
]
