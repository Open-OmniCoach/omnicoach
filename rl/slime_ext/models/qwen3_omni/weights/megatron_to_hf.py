import re

import torch

from .naming import strip_module_prefixes

_LAYER_NAMES = {
    "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
    "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
    "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
    "mlp.router.weight": "mlp.gate.weight",
    "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
    "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
    "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
}


def _remove_vocab_padding(args, param):
    vocab_size = getattr(args, "vocab_size", None)
    if vocab_size is None:
        return param
    return param[:vocab_size]


def _convert_mlp_projection(base, projection, param):
    if projection == "linear_fc1":
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [(f"{base}.gate_proj.weight", gate_weight), (f"{base}.up_proj.weight", up_weight)]
    return [(f"{base}.down_proj.weight", param)]


def convert_qwen3omni_to_hf(args, name, param):
    name = strip_module_prefixes(name)

    if name.startswith("thinker.visual."):
        return [(name, param)]
    if name.startswith("thinker.vision_model."):
        return [("thinker.visual." + name.removeprefix("thinker.vision_model."), param)]
    if name.startswith("thinker.audio_model."):
        return [("thinker.audio_tower." + name.removeprefix("thinker.audio_model."), param)]
    if name.startswith("thinker.video_model."):
        return [("thinker.video_tower." + name.removeprefix("thinker.video_model."), param)]

    if name == "thinker.language_model.embedding.word_embeddings.weight":
        return [("thinker.model.embed_tokens.weight", _remove_vocab_padding(args, param))]
    if name == "thinker.language_model.output_layer.weight":
        return [("thinker.lm_head.weight", _remove_vocab_padding(args, param))]
    if name == "thinker.language_model.decoder.final_layernorm.weight":
        return [("thinker.model.norm.weight", param)]

    head_dim = getattr(args, "kv_channels", None)
    if head_dim is None:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"thinker\.language_model\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        base = f"thinker.model.layers.{layer_idx}"

        local_expert = re.match(r"mlp\.experts\.local_experts\.(\d+)\.(linear_fc[12])\.weight", rest)
        flat_expert = re.match(r"mlp\.experts\.(.+)\.weight(\d+)", rest)
        if local_expert or flat_expert:
            if local_expert:
                expert_idx, projection = local_expert.groups()
            else:
                projection, expert_idx = flat_expert.groups()
            if projection not in {"linear_fc1", "linear_fc2"}:
                raise ValueError(f"Unknown expert parameter name: {name}")
            return _convert_mlp_projection(f"{base}.mlp.experts.{expert_idx}", projection, param)

        shared_expert = re.match(r"mlp\.shared_experts\.(.+)", rest)
        if shared_expert:
            projection = shared_expert[1]
            if projection == "gate_weight":
                return [(f"{base}.mlp.shared_expert_gate.weight", param)]
            if projection not in {"linear_fc1.weight", "linear_fc2.weight"}:
                raise ValueError(f"Unknown shared expert parameter name: {name}")
            return _convert_mlp_projection(
                f"{base}.mlp.shared_expert", projection.removesuffix(".weight"), param
            )

        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.self_attn.o_proj.weight", param)]
        if rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, [value_num_per_group, 1, 1], dim=1)
            return [
                (f"{base}.self_attn.q_proj.weight", q_param.reshape(-1, args.hidden_size)),
                (f"{base}.self_attn.k_proj.weight", k_param.reshape(-1, args.hidden_size)),
                (f"{base}.self_attn.v_proj.weight", v_param.reshape(-1, args.hidden_size)),
            ]
        if rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                [value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            return [
                (f"{base}.self_attn.q_proj.bias", q_bias.contiguous().flatten()),
                (f"{base}.self_attn.k_proj.bias", k_bias.contiguous().flatten()),
                (f"{base}.self_attn.v_proj.bias", v_bias.contiguous().flatten()),
            ]
        if rest in {"mlp.linear_fc1.weight", "mlp.linear_fc2.weight"}:
            return _convert_mlp_projection(f"{base}.mlp", rest.split(".")[1], param)
        mapped = _LAYER_NAMES.get(rest)
        if mapped is not None:
            return [(f"{base}.{mapped}", param)]

    raise ValueError(f"Unknown parameter name: {name}")


_applied = False


def patch():
    """Register Qwen3-Omni Megatron-to-HF conversion and online FP8 quantization."""
    global _applied
    if _applied:
        return

    import slime.backends.megatron_utils.megatron_to_hf as mod

    from slime_ext.models.qwen3_omni.weights.quantizer_fp8 import (
        quantize_qwen3omni_params_fp8,
    )

    _upstream_core = mod._convert_to_hf_core
    _upstream_quantize_params = mod.quantize_params

    def _convert_to_hf_core(args, model_name, name, param):
        if "qwen3omni" in model_name or "qwen3_omni" in model_name or "qwen3-omni" in model_name:
            return convert_qwen3omni_to_hf(args, name, param)
        return _upstream_core(args, model_name, name, param)

    def _quantize_params(
        args,
        megatron_name,
        converted_named_params,
        quantization_config,
        transform_ue8m0=True,
    ):
        normalized_name = strip_module_prefixes(megatron_name)
        if (
            quantization_config is not None
            and quantization_config.get("quant_method") == "fp8"
            and normalized_name.startswith("thinker.language_model.")
        ):
            return quantize_qwen3omni_params_fp8(
                args,
                megatron_name,
                converted_named_params,
                quantization_config,
                transform_ue8m0,
            )
        return _upstream_quantize_params(
            args,
            megatron_name,
            converted_named_params,
            quantization_config,
            transform_ue8m0,
        )

    mod._convert_to_hf_core = _convert_to_hf_core
    mod.quantize_params = _quantize_params
    mod.convert_qwen3omni_to_hf = convert_qwen3omni_to_hf
    mod.quantize_qwen3omni_params_fp8 = quantize_qwen3omni_params_fp8
    _applied = True
