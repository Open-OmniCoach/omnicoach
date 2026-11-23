"""Resolve and validate Example 2 (qwen3_omni_agent_gspo) configuration.

Maintains strict recipe isolation: outcome-only agent configuration without process-credit sections.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from examples._common.agent_recipe import resolve_agent_config


def resolve(path: str | Path, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    return resolve_agent_config(path, environment, forbidden_agent_keys=("credit",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    print(yaml.safe_dump(resolve(args.base, os.environ), sort_keys=False), end="")


if __name__ == "__main__":
    main()
