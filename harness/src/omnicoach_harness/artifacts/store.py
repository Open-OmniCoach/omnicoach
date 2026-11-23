"""`JsonlResultStore` plus the JSON read/write helpers it is built on."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .schema import json_safe


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield value


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


class JsonlResultStore:
    """Thread-safe append store keyed by one explicit persisted field."""

    def __init__(self, path: str | Path, *, key_field: str) -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._offsets: dict[str, int] = {}
        if self.path.is_file():
            with self.path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{self.path}:{offset} must contain a JSON object")
                    if (key := row.get(key_field)) is not None:
                        self._offsets[str(key)] = offset

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._offsets

    def _read(self, key: str) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[key])
            return json.loads(handle.readline())

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read(key) if key in self._offsets else None

    def _append_unlocked(self, row: Mapping[str, Any]) -> None:
        key = str(row.get(self.key_field) or "")
        if not key:
            raise ValueError(f"result row requires {self.key_field}")
        if key in self._offsets:
            raise ValueError(f"duplicate result key: {key}")
        encoded = json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a+b") as handle:
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            offset = handle.tell()
            handle.write((encoded + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        self._offsets[key] = offset

    def append(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            self._append_unlocked(row)

    def replace(self, row: Mapping[str, Any], *, archive_path: str | Path) -> None:
        """Atomically replace one keyed row while preserving its prior value in JSONL history."""

        self.replace_many((row,), archive_path=archive_path)

    def replace_many(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        archive_path: str | Path,
    ) -> None:
        """Atomically replace keyed rows with one archive append and one primary rewrite."""

        replacements: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row.get(self.key_field) or "")
            if not key:
                raise ValueError(f"result row requires {self.key_field}")
            if key in replacements:
                raise ValueError(f"duplicate replacement key: {key}")
            replacements[key] = dict(row)
        if not replacements:
            return

        with self._lock:
            missing = [key for key in replacements if key not in self._offsets]
            if missing:
                raise ValueError(f"cannot replace missing result key: {missing[0]}")

            archive = Path(archive_path)
            archive.parent.mkdir(parents=True, exist_ok=True)
            history_index = sum(1 for _ in iter_jsonl(archive)) if archive.is_file() else 0
            with archive.open("a", encoding="utf-8") as handle:
                for offset, key in enumerate(replacements, start=1):
                    archived = {
                        "history_key": f"{key}#attempt-{history_index + offset}",
                        "superseded_key": key,
                        "artifact": self._read(key),
                    }
                    handle.write(
                        json.dumps(json_safe(archived), ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())

            temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            new_offsets: dict[str, int] = {}
            with self.path.open("rb") as source, temp.open("wb") as handle:
                for existing_key, offset in self._offsets.items():
                    new_offsets[existing_key] = handle.tell()
                    if existing_key in replacements:
                        line = (json.dumps(json_safe(replacements[existing_key]), ensure_ascii=False,
                                           separators=(",", ":")) + "\n").encode("utf-8")
                    else:
                        source.seek(offset)
                        line = source.readline().rstrip(b"\n") + b"\n"
                    handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            self._offsets = new_offsets
