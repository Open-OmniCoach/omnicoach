from __future__ import annotations

import re

from slime.backends.megatron_utils.hf_to_megatron.qwen import qwen_moe_hf_tensor

from .naming import strip_module_prefixes


class _ThinkerReader:
    """Present the nested Omni thinker as a regular Qwen checkpoint."""

    def __init__(self, reader):
        self.reader = reader

    def __contains__(self, name: str):
        return f"thinker.{name}" in self.reader

    def get_tensor(self, name: str):
        return self.reader.get_tensor(f"thinker.{name}")


_TOWER_PREFIXES = {
    "thinker.visual.": "thinker.visual.",
    "thinker.audio_model.": "thinker.audio_tower.",
}


def _normalize_local_expert(name: str) -> str:
    match = re.fullmatch(r"(.+\.mlp\.experts)\.local_experts\.(\d+)\.(linear_fc[12])\.(weight|bias)", name)
    if match is None:
        return name
    prefix, expert, projection, kind = match.groups()
    return f"{prefix}.{projection}.{kind}{expert}"


def qwen3_omni_hf_tensor(name: str, reader, config):
    """Return the full HF tensor for one Qwen3-Omni MCore parameter."""

    name = strip_module_prefixes(name)
    for mcore_prefix, hf_prefix in _TOWER_PREFIXES.items():
        if name.startswith(mcore_prefix):
            return reader.get_tensor(hf_prefix + name.removeprefix(mcore_prefix))

    if not name.startswith("thinker.language_model."):
        raise KeyError(f"Unsupported Qwen3-Omni Megatron parameter {name!r}")

    language_name = _normalize_local_expert(name.removeprefix("thinker."))
    return qwen_moe_hf_tensor(
        language_name,
        _ThinkerReader(reader),
        config.thinker_config.text_config,
    )


_applied = False


def patch():
    """Register Qwen3-Omni with Slime's direct HF weight loader."""
    global _applied
    if _applied:
        return

    import slime.backends.megatron_utils.hf_to_megatron as mod

    mod._LOADERS.setdefault("qwen3_omni_moe", qwen3_omni_hf_tensor)
    _applied = True
