"""Prepare task JSONL for Agent GSPO (does not require evidence_targets)."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples._common.agent_recipe import prepare_agent_jsonl


def prepare_jsonl(source: Path, output: Path, *, check_files: bool = True) -> int:
    """Validate agent task JSONL and normalize relative media paths."""
    return prepare_agent_jsonl(source, output, check_files=check_files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = prepare_jsonl(args.input, args.output)
    print(f"Prepared {count} Agent-GSPO prompts: {args.output}")


if __name__ == "__main__":
    main()
