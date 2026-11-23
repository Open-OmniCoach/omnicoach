from types import SimpleNamespace

from slime_ext.models.qwen3_omni import model as model_module
from slime_ext.models.qwen3_omni import provider


def test_native_provider_populates_omni_config(monkeypatch):
    thinker_config = SimpleNamespace(
        text_config=SimpleNamespace(rope_parameters={"mrope_section": [24, 20, 20]}),
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    hf_config = SimpleNamespace(model_type="qwen3_omni_moe", thinker_config=thinker_config)
    args = SimpleNamespace(
        hf_checkpoint="/qwen3-omni",
        padded_vocab_size=152064,
        max_position_embeddings=65536,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=True,
        rotary_percent=1.0,
        rotary_base=1_000_000,
        mrope_section=[1, 1, 1],
        multimodal_attn_impl="auto",
        vit_gradient_checkpointing=False,
        transformer_impl="transformer_engine",
    )
    config = SimpleNamespace()
    requested_groups = []
    process_groups = object()

    monkeypatch.setattr(provider.AutoConfig, "from_pretrained", lambda *args, **kwargs: hf_config)
    monkeypatch.setattr(provider, "get_gpt_decoder_block_spec", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        provider.ProcessGroupCollection,
        "use_mpu_process_groups",
        lambda groups=None: requested_groups.append(groups) or process_groups,
    )
    monkeypatch.setattr(model_module, "Qwen3OmniModel", lambda **kwargs: kwargs)

    model_provider = provider.get_qwen3_omni_model_provider(args, config, None)
    model = model_provider()

    assert callable(model_provider)
    assert config.vocab_size == 152064
    assert config.mrope_section == [24, 20, 20]
    assert config.audio_token_id == 151675
    assert config.apply_rope_fusion is False
    assert requested_groups == [None]
    assert model["pg_collection"] is process_groups
