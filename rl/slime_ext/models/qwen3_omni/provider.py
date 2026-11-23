from __future__ import annotations

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from transformers import AutoConfig


def _apply_omni_config(config, args, thinker_config) -> None:
    text_config = thinker_config.text_config
    rope = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", None) or {}

    config.vocab_size = args.padded_vocab_size
    config.language_max_sequence_length = args.max_position_embeddings
    config.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
    config.share_embeddings_and_output_weights = not args.untie_embeddings_and_output_weights
    config.rotary_percent = args.rotary_percent
    config.rotary_base = args.rotary_base
    config.mrope_section = list(rope.get("mrope_section", args.mrope_section))
    # MCore 0.20 requests return_raw_freqs=True when apply_rope_fusion is enabled;
    # keep it disabled to match upstream Qwen3-VL / Qwen3.5-VL providers.
    config.apply_rope_fusion = False

    for name, default in (
        ("image_token_id", 151655),
        ("video_token_id", 151656),
        ("audio_token_id", 151675),
        ("vision_start_token_id", 151652),
        ("audio_start_token_id", 151669),
        ("position_id_per_seconds", 13),
    ):
        setattr(config, name, getattr(thinker_config, name, default))

    config.spatial_merge_size = getattr(thinker_config.vision_config, "spatial_merge_size", 2)
    config.multimodal_attn_impl = getattr(args, "multimodal_attn_impl", "auto")
    config.vit_gradient_checkpointing = getattr(args, "vit_gradient_checkpointing", False)
    config.response_only_lm_head = getattr(args, "response_only_lm_head", False)
    config.vision_batch_invariant_positions = getattr(args, "vision_batch_invariant_positions", False)
    config.vision_cp = getattr(args, "vision_cp", False)
    config.audio_fixed_conv_batch = getattr(args, "audio_fixed_conv_batch", False)


def get_qwen3_omni_model_provider(args, config, vp_stage):
    """Return Slime's native Qwen3-Omni model provider."""

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    if getattr(hf_config, "model_type", None) != "qwen3_omni_moe":
        raise ValueError(f"{args.hf_checkpoint} is not a Qwen3-Omni checkpoint")

    thinker_config = hf_config.thinker_config
    _apply_omni_config(config, args, thinker_config)
    layer_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=args.transformer_impl == "transformer_engine",
        vp_stage=vp_stage,
    )

    def model_provider(pre_process=True, post_process=True, vp_stage=None):
        from .model import Qwen3OmniModel

        process_groups = ProcessGroupCollection.use_mpu_process_groups()
        return Qwen3OmniModel(
            language_transformer_config=config,
            language_transformer_layer_spec=layer_spec,
            thinker_transformer_config=thinker_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=process_groups,
        )

    return model_provider
