"""`IndexedActorStore`, an append-only store that journals offsets for fast resume."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .store import JsonlResultStore


class IndexedActorStore(JsonlResultStore):
    def __init__(self, path: str | Path, *, key_field: str) -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path = self.path.with_suffix(".resume.jsonl")
        self._lock = threading.Lock()
        self._offsets: dict[str, int] = {}
        self.statuses: dict[str, str] = {}
        end = 0
        if self.index_path.exists():
            valid_bytes = 0
            with self.index_path.open("rb") as f:
                for line in f:
                    if not line.endswith(b"\n"):
                        break
                    row = json.loads(line)
                    self._offsets[row["key"]] = row["offset"]
                    self.statuses[row["key"]] = row["status"]
                    end = row["end"]
                    valid_bytes = f.tell()
            if valid_bytes < self.index_path.stat().st_size:
                with self.index_path.open("r+b") as f:
                    f.truncate(valid_bytes)  # Rebuild only this derived journal's torn tail.
        size = self.path.stat().st_size if self.path.exists() else 0
        if end > size:
            raise ValueError("append-only actor output was truncated; preserve it and use a successor")
        if size > end:
            with self.path.open("rb", buffering=8 * 1024**2) as f:
                f.seek(end)
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        raise ValueError("incomplete actor output tail; preserve for explicit recovery")
                    row = json.loads(line)
                    key = str(row[key_field])
                    self._offsets[key] = offset
                    self._record_unlocked(key, row["episode"]["status"], f.tell())

    def _record_unlocked(self, key: str, status: str, end: int) -> None:
        value = {"key": key, "offset": self._offsets[key], "end": end, "status": status}
        with self.index_path.open("ab") as f:
            f.write((json.dumps(value, separators=(",", ":")) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
        self.statuses[key] = status

    def append(self, row: Any) -> None:
        # Hold the store lock across both the primary JSONL append and the resume
        # journal write so concurrent appenders cannot interleave offsets.
        with self._lock:
            self._append_unlocked(row)
            key = str(row[self.key_field])
            self._record_unlocked(key, row["episode"]["status"], self.path.stat().st_size)

    def replace_many(self, rows: Any, *, archive_path: str | Path) -> None:
        del archive_path
        if list(rows):
            raise ValueError("indexed actor output is append-only; retry in a successor")
