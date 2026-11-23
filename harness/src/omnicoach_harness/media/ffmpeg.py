"""`require_binary` and `run`, the subprocess wrapper used for every FFmpeg call."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandResult:
    command: tuple[str, ...]
    stdout: str
    stderr: str


def require_binary(name: str) -> str:
    value = shutil.which(name)
    if value is None:
        raise RuntimeError(f"required system binary is unavailable: {name}")
    return value


def run(command: Sequence[str], *, timeout_seconds: float) -> CommandResult:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    result = CommandResult(
        command=tuple(str(item) for item in command),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if completed.returncode != 0:
        tail = completed.stderr[-4000:]
        raise RuntimeError(f"media command failed ({completed.returncode}): {tail}")
    return result
