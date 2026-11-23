"""Thin Agent GSPO rollout entrypoint delegating to ``slime_ext.agent`` with no-op hooks.

Uses only ``slime_ext.agent.rollout``; zero cross-example imports.
"""

from __future__ import annotations

from typing import Any

from slime_ext.agent.rollout import RolloutHooks, generate_episode

DEFAULT_GSPO_HOOKS = RolloutHooks()


async def generate(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    **kwargs: Any,
) -> Any:
    """Slime ``--custom-generate-function-path`` for pure outcome Agent GSPO."""
    return await generate_episode(
        args,
        sample,
        sampling_params,
        evaluation=evaluation,
        hooks=DEFAULT_GSPO_HOOKS,
        **kwargs,
    )


__all__ = ["DEFAULT_GSPO_HOOKS", "generate"]
