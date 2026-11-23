"""Shared-media Parquet pooling, indexing, and lazy row resolution."""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

MEDIA_TABLE_DIR = "media"


def _parquet_parts(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("part-*.parquet"))
    if not paths:
        raise ValueError(f"no Parquet shards under {directory}")
    return paths


class ParquetFilePool:
    """Share one LRU of open Parquet files across all media sources in a worker."""

    def __init__(self, max_open_files: int = 32) -> None:
        self.__setstate__({"max_open_files": max_open_files})

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            inherited = self._files
            # A thread may hold the parent process lock at fork. Replace it before
            # accessing handles, and reopen every Parquet file in the child.
            self.__setstate__({"max_open_files": self.max_open_files})
            for parquet in inherited.values():
                parquet.close()

    @property
    def open_file_count(self) -> int:
        self._ensure_process()
        return len(self._files)

    def read_row_group(self, path: str | Path, row_group: int) -> dict[str, Any]:
        self._ensure_process()
        target = Path(path)
        with self._lock:
            parquet = self._files.pop(target, None)
            if parquet is None:
                parquet = pq.ParquetFile(target)
            self._files[target] = parquet
            while len(self._files) > self.max_open_files:
                _, evicted = self._files.popitem(last=False)
                evicted.close()
            # DataLoader already parallelizes across worker processes. Avoid
            # starting another Arrow thread fan-out for a single media row.
            return parquet.read_row_group(row_group, use_threads=False).to_pylist()[0]

    def close(self) -> None:
        self._ensure_process()
        with self._lock:
            for parquet in self._files.values():
                parquet.close()
            self._files.clear()

    def __getstate__(self) -> dict[str, Any]:
        return {"max_open_files": self.max_open_files}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.max_open_files = max(1, int(state["max_open_files"]))
        self._files: OrderedDict[Path, pq.ParquetFile] = OrderedDict()
        self._lock = threading.Lock()
        self._pid = os.getpid()


class IndexedMediaStore(ABC):
    """Common metadata scan and pooled reads for shared-media layouts."""

    media_label = ""
    index_count_error = "media metadata row count does not match row groups"
    index_columns: tuple[str, ...] | list[str] = ("media_id",)
    required_columns: tuple[str, ...] = ()
    media_schema: Any = None

    def __init__(
        self,
        source_root: Path,
        *,
        max_open_files: int = 32,
        file_pool: ParquetFilePool | None = None,
    ) -> None:
        self.source_root = source_root
        self._owns_file_pool = file_pool is None
        self._file_pool = file_pool or ParquetFilePool(max_open_files)
        self._index: dict[str, Any] = {}
        self._build_index()

    def _build_index(self) -> None:
        for path, row_group, media_id, row in self._iter_index_rows(
            _parquet_parts(self.source_root / MEDIA_TABLE_DIR),
            list(self.index_columns),
        ):
            self._index[media_id] = self._location(path, row_group, media_id, row)

    @abstractmethod
    def _location(self, path: Path, row_group: int, media_id: str, row: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _validate_index(self, parquet: pq.ParquetFile, path: Path) -> None:
        if self.media_schema is not None:
            if parquet.schema_arrow != self.media_schema:
                raise ValueError(f"Agent shared-media schema mismatch: {path}")
            if parquet.metadata.num_rows != parquet.metadata.num_row_groups:
                raise ValueError(f"Agent media requires one row per row group: {path}")
        else:
            missing = sorted(set(self.required_columns) - set(parquet.schema_arrow.names))
            if missing:
                raise ValueError(f"media shard {path} is missing columns: {missing}")
            for group in range(parquet.metadata.num_row_groups):
                if parquet.metadata.row_group(group).num_rows != 1:
                    raise ValueError(f"media row group must contain one row: {path}#{group}")

    def _iter_index_rows(
        self,
        paths: list[Path],
        columns: list[str],
    ) -> Iterator[tuple[Path, int, str, dict[str, Any]]]:
        for path in paths:
            with pq.ParquetFile(path) as parquet:
                self._validate_index(parquet, path)
                # One metadata-only scan per shard avoids repeated remote storage reads.
                rows = parquet.read(columns=columns).to_pylist()
                if len(rows) != parquet.metadata.num_row_groups:
                    raise ValueError(f"{self.index_count_error}: {path}")
                for row_group, row in enumerate(rows):
                    media_id = str(row["media_id"])
                    if media_id in self._index:
                        raise ValueError(f"duplicate {self.media_label}media_id: {media_id}")
                    yield path, row_group, media_id, row

    @property
    def media_ids(self) -> set[str]:
        return set(self._index)

    def owns(self, media_id: str) -> bool:
        return media_id in self._index

    def metadata(self, media_id: str) -> Any:
        try:
            return self._index[media_id]
        except KeyError as exc:
            raise KeyError(f"unknown {self.media_label}media_id {media_id!r} in {self.source_root}") from exc

    def read(self, media_id: str) -> dict[str, Any]:
        location = self.metadata(media_id)
        return self._file_pool.read_row_group(location.path, location.row_group)

    def close(self) -> None:
        if self._owns_file_pool:
            self._file_pool.close()


class MediaRegistry:
    """Own source stores while sharing one bounded file pool."""

    store_class: type[IndexedMediaStore]
    reuse_media_ids = False

    def __init__(self, *, file_pool: ParquetFilePool | None = None) -> None:
        self._file_pool = file_pool or ParquetFilePool()
        self._by_root: dict[Path, IndexedMediaStore] = {}
        self._by_media_id: dict[str, IndexedMediaStore] = {}

    def register(self, store: IndexedMediaStore) -> None:
        if store.source_root in self._by_root:
            return
        media_ids = store.media_ids
        if not self.reuse_media_ids:
            overlap = self._by_media_id.keys() & media_ids
            if overlap:
                raise ValueError(f"media_id collision across sources: {sorted(overlap)[:3]}")
        self._by_root[store.source_root] = store
        for media_id in media_ids:
            self._by_media_id.setdefault(media_id, store)

    def store_for_root(self, source_root: str | Path) -> IndexedMediaStore:
        root = Path(source_root).expanduser().resolve()
        store = self._by_root.get(root)
        if store is None:
            store = self.store_class(root, file_pool=self._file_pool)
            self.register(store)
        return store

    def store_for(self, media_id: str) -> IndexedMediaStore:
        try:
            return self._by_media_id[media_id]
        except KeyError as exc:
            raise KeyError(f"no registered shared-media store owns {media_id!r}") from exc

    def owns(self, media_id: str) -> bool:
        return media_id in self._by_media_id

    def resolve_row(self, row: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        for store in self._by_root.values():
            store.close()
        self._file_pool.close()

    def __bool__(self) -> bool:
        return bool(self._by_root)


class LazyMediaDataset:
    """Resolve media for row fetches while leaving column access lightweight."""

    resolved_columns: tuple[str, ...] = ()

    def __init__(self, dataset: Any, registry: MediaRegistry) -> None:
        self.dataset = dataset
        self.registry = registry

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def column_names(self) -> list[str]:
        columns = list(self.dataset.column_names)
        for name in self.resolved_columns:
            if name not in columns:
                columns.append(name)
        return columns

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self.dataset[key]
        if isinstance(key, slice):
            return [self[index] for index in range(*key.indices(len(self)))]
        if isinstance(key, (list, tuple)):
            return [self[int(index)] for index in key]
        return self.registry.resolve_row(self.dataset[key])

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "dataset"), name)
