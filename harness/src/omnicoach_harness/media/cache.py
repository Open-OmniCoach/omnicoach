"""Keys, directories and locks for media operations cached on disk."""

from __future__ import annotations

import base64
import errno
import fcntl
import json
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


def safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized[:160] or "item"


def operation_key(name: str, arguments: Mapping[str, Any]) -> str:
    payload = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{safe_id(name)}_{encoded}"


def ensure_operation_dir(root: str | Path, *, name: str, arguments: Mapping[str, Any]) -> Path:
    key = operation_key(name, arguments)
    chunks = [key[index : index + 180] for index in range(0, len(key), 180)]
    path = Path(root).joinpath(*chunks)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def operation_lock(path: str | Path) -> Iterator[None]:
    """Serialize one cached media operation across threads and local processes."""

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    key = str(root.resolve())
    with _LOCKS_GUARD:
        thread_lock = _LOCKS.setdefault(key, threading.RLock())
    with thread_lock, (root / ".operation.lock").open("a+b") as handle:
        process_lock = True
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
            process_lock = False
        try:
            yield
        finally:
            if process_lock:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
