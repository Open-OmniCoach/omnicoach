"""Qwen3-Omni HF <-> Megatron weight conversion and FP8 quantization."""


def patch() -> None:
    from . import hf_to_megatron, megatron_to_hf

    hf_to_megatron.patch()
    megatron_to_hf.patch()


__all__ = ["patch"]
