"""Materialize Evalkit cache URIs into a worker-local Agent build cache."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from omnicoach_sft.io import atomic_write_bytes


class EvalkitCacheMaterializer:
    def __init__(self, *, cache_root: str | Path, cache_dataset: str, output_root: str | Path) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.cache_dataset = cache_dataset
        self.output_root = Path(output_root).resolve()

    def resolve(self, uri: str, *, expected_kind: str) -> str:
        if not uri.startswith("evalkit-cache://"):
            return uri
        parsed = urlparse(uri)
        pieces = [unquote(parsed.netloc), *(unquote(piece) for piece in parsed.path.split("/") if piece)]
        if len(pieces) != 5:
            raise ValueError(f"invalid evalkit cache URI: {uri}")
        dataset, cache_id, media_id, bundle_kind, payload_kind = pieces
        if dataset != self.cache_dataset or payload_kind != expected_kind:
            raise ValueError(f"evalkit cache URI contract mismatch: {uri}")
        entry = (self.cache_root / dataset / cache_id / media_id).resolve()
        namespace = (self.cache_root / dataset).resolve()
        if entry != namespace and namespace not in entry.parents:
            raise ValueError(f"evalkit cache URI escapes its namespace: {uri}")
        manifest = self._manifest(entry / "manifest.json")
        if str(manifest.get("cache_id") or "") != cache_id or str(manifest.get("media_id") or "") != media_id:
            raise ValueError(f"evalkit cache identity mismatch: {entry}")
        query = parse_qs(parsed.query)
        if expected_kind == "frame":
            return self._frame(uri, entry, manifest, bundle_kind, query)
        if expected_kind == "audio":
            return self._audio(uri, entry, manifest, query)
        raise ValueError(f"unsupported evalkit cache payload: {expected_kind!r}")

    def _frame(
        self,
        uri: str,
        entry: Path,
        manifest: dict[str, Any],
        bundle_kind: str,
        query: dict[str, list[str]],
    ) -> str:
        timestamp = self._single_float(query, "timestamp", uri)
        manifest_key = "initial" if bundle_kind == "initial" else "visual"
        record = manifest.get(manifest_key)
        if not isinstance(record, dict) or not record.get("file"):
            raise ValueError(f"cache manifest has no {manifest_key} payload: {entry}")
        match = next(
            (item for item in record.get("index") or () if abs(float(item[0]) - timestamp) <= 1e-6),
            None,
        )
        if match is None:
            raise ValueError(f"frame timestamp {timestamp} is absent from cache: {entry}")
        source = entry / str(record["file"])
        offset, length = int(match[1]), int(match[2])
        payload = self._payload(source)[offset : offset + length]
        if len(payload) != length:
            raise ValueError(f"short read from evalkit MJPEG cache: {source}")
        digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
        target = self.output_root / "images" / f"{digest}.jpg"
        atomic_write_bytes(target, payload, parents=True, skip_if_size_matches=True)
        return str(target)

    def _audio(
        self,
        uri: str,
        entry: Path,
        manifest: dict[str, Any],
        query: dict[str, list[str]],
    ) -> str:
        record = manifest.get("audio")
        if not isinstance(record, dict) or not record.get("file"):
            raise ValueError(f"cache manifest has no audio payload: {entry}")
        source = entry / str(record["file"])
        if "start" not in query and "end" not in query:
            return str(source.resolve())
        start = self._single_float(query, "start", uri)
        end = self._single_float(query, "end", uri)
        if end <= start:
            raise ValueError(f"invalid evalkit audio span [{start}, {end}]: {uri}")
        target = self.output_root / "audios" / f"{hashlib.sha256(uri.encode()).hexdigest()}.wav"
        if target.is_file() and target.stat().st_size > 44:
            return str(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with wave.open(str(source), "rb") as reader:
                rate = reader.getframerate()
                start_frame = min(reader.getnframes(), max(0, round(start * rate)))
                end_frame = min(reader.getnframes(), max(start_frame, round(end * rate)))
                reader.setpos(start_frame)
                payload = reader.readframes(end_frame - start_frame)
                params = reader.getparams()
            with wave.open(str(temporary), "wb") as writer:
                writer.setparams(params)
                writer.writeframes(payload)
            if temporary.stat().st_size <= 44:
                raise ValueError(f"empty evalkit audio slice: {uri}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return str(target)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _manifest(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"evalkit cache manifest must be an object: {path}")
        return value

    @staticmethod
    @lru_cache(maxsize=32)
    def _payload(path: Path) -> bytes:
        return path.read_bytes()

    @staticmethod
    def _single_float(query: dict[str, list[str]], key: str, uri: str) -> float:
        values = query.get(key)
        if not values or len(values) != 1:
            raise ValueError(f"evalkit cache URI requires one {key}: {uri}")
        return float(values[0])


def materialize_evalkit_rows(
    rows: list[dict[str, Any]],
    *,
    resolver: dict[str, Any],
    output_root: str | Path,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if resolver.get("scheme") != "evalkit-cache":
        raise ValueError(f"unsupported media resolver: {resolver.get('scheme')!r}")
    materializer = EvalkitCacheMaterializer(
        cache_root=str(resolver.get("cache_root") or ""),
        cache_dataset=str(resolver.get("cache_dataset") or ""),
        output_root=output_root,
    )
    if workers < 1:
        raise ValueError("workers must be positive")
    tasks = sorted(
        {
            (kind, str(uri))
            for row in rows
            for kind, field in (("frame", "images"), ("audio", "audios"))
            for uri in row.get(field) or []
        }
    )

    def resolve(task: tuple[str, str]) -> tuple[tuple[str, str], str]:
        kind, uri = task
        return task, materializer.resolve(uri, expected_kind=kind)

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as executor:
        resolved = dict(executor.map(resolve, tasks))
    materialized: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        row["images"] = [resolved[("frame", str(uri))] for uri in row.get("images") or []]
        row["audios"] = [resolved[("audio", str(uri))] for uri in row.get("audios") or []]
        materialized.append(row)
    return materialized
