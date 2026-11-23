"""Native Qwen3-Omni model package for Slime."""

__all__ = ["Qwen3OmniModel"]


def __getattr__(name: str):
    if name == "Qwen3OmniModel":
        from .model import Qwen3OmniModel

        return Qwen3OmniModel
    raise AttributeError(name)
