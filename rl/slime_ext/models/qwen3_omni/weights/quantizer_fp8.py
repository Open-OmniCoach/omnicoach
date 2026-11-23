"""Qwen3-Omni online FP8 quantization for Megatron-to-HF sync.

Upstream Slime's FP8 dispatcher recognizes decoder names rooted at
``module.module.decoder.layers``.  Omni's MCore actor nests the same decoder
under ``thinker.language_model`` and also uses ``local_experts`` parameter
names.  Without this model-local dispatch, BF16 weights are sent to an FP8
SGLang model without matching scales, corrupting the first online update.
"""

from __future__ import annotations

import logging
import re

from .naming import strip_module_prefixes

_DECODER_PATTERN = re.compile(
    r"thinker\.language_model\.decoder\.layers\.(\d+)\.(.+)"
)
_LOCAL_EXPERT_PATTERN = re.compile(
    r"mlp\.experts\.local_experts\.\d+\.linear_fc[12]\.weight"
)
_FLAT_EXPERT_PATTERN = re.compile(r"mlp\.experts\.(.+)\.weight\d+")
_SHARED_EXPERT_PARAMS = {
    "mlp.shared_experts.linear_fc1.weight",
    "mlp.shared_experts.linear_fc2.weight",
}
_ATTENTION_FP8_PARAMS = {
    "self_attention.linear_proj.weight",
    "self_attention.linear_qkv.weight",
}
_DENSE_MLP_FP8_PARAMS = {
    "mlp.linear_fc1.weight",
    "mlp.linear_fc2.weight",
}
_LOGGER = logging.getLogger(__name__)
_logged_activation = False


def _attention_is_ignored(quantization_config) -> bool:
    """Map the HF ``self_attn`` ignore scope onto MCore attention names.

    SGLang reads Hugging Face module paths from ``ignored_layers`` while the
    online converter receives Megatron names such as ``self_attention``.  The
    two sides must make the same precision decision or a BF16 online update can
    overwrite an FP8 buffer (or vice versa).
    """

    ignored_layers = quantization_config.get("ignored_layers")
    if ignored_layers is None:
        ignored_layers = quantization_config.get("modules_to_not_convert", [])
    if isinstance(ignored_layers, str):
        ignored_layers = [ignored_layers]
    for value in ignored_layers or []:
        parts = str(value).removeprefix("model.").strip(".").split(".")
        if "self_attn" in parts or "self_attention" in parts:
            return True
    return False


def quantize_qwen3omni_params_fp8(
    args,
    megatron_name,
    converted_named_params,
    quantization_config,
    transform_ue8m0=True,
):
    """Quantize exactly the Omni Thinker decoder families served as FP8."""

    global _logged_activation

    if quantization_config["quant_method"] != "fp8":
        return converted_named_params
    if quantization_config.get("fmt", "e4m3") != "e4m3":
        raise AssertionError(
            f"Unsupported FP8 format: {quantization_config.get('fmt')}"
        )
    if quantization_config["activation_scheme"] != "dynamic":
        raise AssertionError(
            "Qwen3-Omni online FP8 requires dynamic activation scaling"
        )

    match = _DECODER_PATTERN.fullmatch(strip_module_prefixes(megatron_name))
    if match is None:
        return converted_named_params
    _layer_idx, rest = match.groups()

    quantize = (
        (rest in _ATTENTION_FP8_PARAMS and not _attention_is_ignored(quantization_config))
        or rest in _DENSE_MLP_FP8_PARAMS
        or rest in _SHARED_EXPERT_PARAMS
        or _LOCAL_EXPERT_PATTERN.fullmatch(rest) is not None
        or _FLAT_EXPERT_PATTERN.fullmatch(rest) is not None
    )
    if not quantize:
        return converted_named_params

    if not _logged_activation:
        _LOGGER.warning(
            "SLIME_EXT_QWEN3OMNI_ONLINE_FP8=enabled "
            "nested_decoder=true local_experts=true attention=%s block=%s",
            "bf16" if _attention_is_ignored(quantization_config) else "fp8",
            quantization_config.get("weight_block_size"),
        )
        _logged_activation = True

    from slime.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import (
        _quantize_param,
    )

    weight_block_size = quantization_config.get("weight_block_size")
    force_ue8m0_scale = getattr(args, "force_fp8_ue8m0_scale", False)
    quantized = []
    for converted_name, param in converted_named_params:
        # Megatron TE checkpoints may expose auxiliary BF16 scales alongside a
        # weight.  SGLang needs newly computed block scales for the weight, not
        # those source-side auxiliaries.
        if converted_name.endswith("_scale"):
            continue
        quantized.extend(
            _quantize_param(
                converted_name,
                param,
                weight_block_size,
                transform_ue8m0,
                force_ue8m0_scale=force_ue8m0_scale,
            )
        )
    return quantized


__all__ = ["quantize_qwen3omni_params_fp8"]
