"""Resolve and validate Example 1 (qwen3_omni_rl) configuration."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

VALID_ESTIMATORS = frozenset({"grpo", "gspo", "cispo"})
VALID_REWARD_TYPES = frozenset({"auto", "mcq", "av_match", "temporal_iou"})
VALID_ABLATIONS = frozenset({"none", "no-audio", "no-image", "text-only"})


def resolve(path: str | Path, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = dict(environment or {})
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or "qwen3_omni_rl" not in config:
        raise ValueError("config must contain top-level 'qwen3_omni_rl' mapping")
    rl = config["qwen3_omni_rl"]

    if "ADVANTAGE_ESTIMATOR" in env:
        rl["advantage_estimator"] = env["ADVANTAGE_ESTIMATOR"].strip().lower()
    if "REWARD_TYPE" in env:
        rl["reward_type"] = env["REWARD_TYPE"].strip().lower()
    if "ABLATION_MODE" in env:
        rl["ablation_mode"] = env["ABLATION_MODE"].strip().lower()

    if rl["advantage_estimator"] not in VALID_ESTIMATORS:
        raise ValueError(
            f"advantage_estimator must be one of {sorted(VALID_ESTIMATORS)}, got {rl['advantage_estimator']!r}"
        )
    if rl["reward_type"] not in VALID_REWARD_TYPES:
        raise ValueError(
            f"reward_type must be one of {sorted(VALID_REWARD_TYPES)}, got {rl['reward_type']!r}"
        )
    if rl["ablation_mode"] not in VALID_ABLATIONS:
        raise ValueError(
            f"ablation_mode must be one of {sorted(VALID_ABLATIONS)}, got {rl['ablation_mode']!r}"
        )

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    print(yaml.safe_dump(resolve(args.base, os.environ), sort_keys=False), end="")


if __name__ == "__main__":
    main()
