"""Shared Megatron/HF parameter naming helpers for Qwen3-Omni."""

from __future__ import annotations


def strip_module_prefixes(name: str) -> str:
    """Strip any number of leading Megatron DDP/FP16 ``module.`` wrappers."""
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name


__all__ = ["strip_module_prefixes"]
