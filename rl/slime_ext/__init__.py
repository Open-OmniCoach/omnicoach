"""Omni and active-perception extensions for Slime, activated by :func:`apply`."""

import logging
import os

logger = logging.getLogger(__name__)

_applied = False


def apply():
    """Idempotently apply every slime_ext patch and registration.

    Safe to call repeatedly and from any process (driver or Ray worker). Each
    sub-patch is also individually idempotent.
    """
    global _applied
    if _applied:
        return
    # Upstream imports can re-enter through the autoload hook.
    _applied = True

    try:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

        # Pre-warmed Ray workers need Slime's sibling packages on sys.path;
        # changing their PYTHONPATH environment alone is insufficient.
        import sys
        from pathlib import Path

        import slime

        slime_repo = str(Path(slime.__file__).parents[1])
        if slime_repo not in sys.path:
            sys.path.insert(0, slime_repo)

        from slime_ext.backends.megatron import (
            actor,
            arguments,
            data,
            loss,
            rolling_checkpoint,
            update_weight,
        )
        from slime_ext.models.qwen3_omni.weights import (
            hf_to_megatron,
            megatron_to_hf,
        )

        for module in (
            arguments,
            data,
            loss,
            update_weight,
            hf_to_megatron,
            megatron_to_hf,
            actor,
            rolling_checkpoint,
        ):
            module.patch()

        if os.environ.get("SLIME_EXT_BATCHED_TOP_P_IDS") == "1":
            from slime_ext.backends.sglang import top_p

            top_p.patch()

        if os.environ.get("SLIME_EXT_QWEN3OMNI_FP8_PREFIXLESS_BF16") == "1":
            from slime_ext.backends.sglang.fp8 import (
                patch_prefixless_multimedia_bf16,
            )

            patch_prefixless_multimedia_bf16()
    except Exception:
        _applied = False
        raise

    logger.info("slime_ext patches applied.")
