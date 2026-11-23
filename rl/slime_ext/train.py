"""Unified CLI training entrypoint: ``python -m slime_ext.train``.

Supports optional Ray cluster initialization before dispatching to upstream
Slime's ``train.py``:
- ``--ray-address <addr>`` (or env ``SLIME_RAY_ADDRESS``)
- ``--ray-runtime-env-json <json>`` (or env ``SLIME_RAY_RUNTIME_ENV_JSON``)
- ``--ray-env-vars-json <json>`` (or env ``SLIME_RAY_ENV_VARS_JSON``)

Wrapper Ray flags are stripped from ``sys.argv`` prior to running Slime's parser.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from slime_ext._entrypoint import run

_RAY_FLAGS = ("--ray-address", "--ray-runtime-env-json", "--ray-env-vars-json")


def _extract_ray_options(argv: Sequence[str]) -> tuple[dict[str, str | None], list[str]]:
    """Strip wrapper Ray flags from argv while preserving all Slime CLI args."""
    extracted: dict[str, str | None] = {
        "ray_address": None,
        "ray_runtime_env_json": None,
        "ray_env_vars_json": None,
    }
    remaining: list[str] = []
    args = list(argv)
    i = 0
    while i < len(args):
        token = args[i]
        matched = False
        for flag in _RAY_FLAGS:
            key = flag.lstrip("-").replace("-", "_")
            if token == flag:
                if i + 1 >= len(args):
                    raise ValueError(f"{flag} requires a value")
                extracted[key] = args[i + 1]
                i += 2
                matched = True
                break
            prefix = f"{flag}="
            if token.startswith(prefix):
                extracted[key] = token[len(prefix) :]
                i += 1
                matched = True
                break
        if not matched:
            remaining.append(token)
            i += 1
    return extracted, remaining


def _build_runtime_env(
    runtime_env_json: str | None,
    env_vars_json: str | None,
) -> dict[str, Any] | None:
    runtime_env: dict[str, Any] | None = None
    if runtime_env_json:
        parsed = json.loads(runtime_env_json)
        if not isinstance(parsed, dict):
            raise ValueError("--ray-runtime-env-json must decode to a JSON object")
        runtime_env = parsed
    if env_vars_json:
        parsed_env = json.loads(env_vars_json)
        if not isinstance(parsed_env, dict):
            raise ValueError("--ray-env-vars-json must decode to a JSON object")
        runtime_env = dict(runtime_env or {})
        merged_env = dict(runtime_env.get("env_vars") or {})
        merged_env.update(parsed_env)
        runtime_env["env_vars"] = merged_env
    return runtime_env


def _maybe_init_ray(argv: Sequence[str] | None = None) -> list[str]:
    """Parse optional Ray options from argv/env, initialize Ray if requested, and return remaining argv."""
    source_argv = list(sys.argv[1:] if argv is None else argv)
    opts, remaining = _extract_ray_options(source_argv)

    address = opts["ray_address"] or os.environ.get("SLIME_RAY_ADDRESS")
    runtime_env_json = opts["ray_runtime_env_json"] or os.environ.get("SLIME_RAY_RUNTIME_ENV_JSON")
    env_vars_json = opts["ray_env_vars_json"] or os.environ.get("SLIME_RAY_ENV_VARS_JSON")
    runtime_env = _build_runtime_env(runtime_env_json, env_vars_json)

    if address or runtime_env is not None:
        import ray

        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {}
            if address:
                init_kwargs["address"] = address
            if runtime_env is not None:
                init_kwargs["runtime_env"] = runtime_env
            ray.init(**init_kwargs)

    return remaining


def main(argv: Sequence[str] | None = None) -> None:
    remaining = _maybe_init_ray(argv)
    sys.argv = [sys.argv[0], *remaining]
    run("train")


if __name__ == "__main__":
    main()
