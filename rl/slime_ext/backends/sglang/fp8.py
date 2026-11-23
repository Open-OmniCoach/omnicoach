"""Opt-in SGLang FP8 compatibility for prefixless Omni multimedia linears.

The pinned SGLang Qwen3-Omni implementation constructs a small number of
multimedia ``LinearBase`` modules with an empty component prefix.  Encoding an
empty string in ``ignored_layers`` is unsafe because this SGLang version
normalizes it to ``model.`` and consequently disables FP8 for the complete
Thinker decoder.  Keep only those prefixless linears unquantized at runtime;
all explicitly named decoder projections and experts retain the checkpoint's
FP8 method.

This patch is deliberately opt-in: set
``SLIME_EXT_QWEN3OMNI_FP8_PREFIXLESS_BF16=1`` before any SGLang model is
constructed when serving FP8 Qwen3-Omni checkpoints.  BF16 runs leave it
disabled and keep their default runtime behavior unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any

_STATE_ATTR = "_slime_ext_qwen3omni_fp8_prefixless_bf16_state"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PatchState:
    original_get_quant_method: Any
    installed_get_quant_method: Any


def patch_prefixless_multimedia_bf16() -> None:
    """Keep exactly the empty-prefix Omni multimedia linears in BF16."""

    import sglang.srt.layers.quantization.fp8 as fp8_module
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    existing = getattr(fp8_module, _STATE_ATTR, None)
    if existing is not None:
        if not isinstance(existing, _PatchState):
            raise RuntimeError("Qwen3-Omni FP8 prefixless patch state was replaced")
        if fp8_module.Fp8Config.get_quant_method is not existing.installed_get_quant_method:
            raise RuntimeError("Qwen3-Omni FP8 prefixless patch binding drifted")
        return

    original_get_quant_method = fp8_module.Fp8Config.get_quant_method

    @wraps(original_get_quant_method)
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and prefix == "":
            return UnquantizedLinearMethod()
        return original_get_quant_method(self, layer, prefix)

    state = _PatchState(
        original_get_quant_method=original_get_quant_method,
        installed_get_quant_method=get_quant_method,
    )
    fp8_module.Fp8Config.get_quant_method = get_quant_method
    setattr(fp8_module, _STATE_ATTR, state)
    _LOGGER.warning(
        "SLIME_EXT_QWEN3OMNI_FP8_PREFIXLESS_BF16=installed decoder_fp8=enabled"
    )


__all__ = ["patch_prefixless_multimedia_bf16"]
