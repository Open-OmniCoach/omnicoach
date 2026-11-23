from __future__ import annotations

import pickle
from pathlib import Path
from typing import ClassVar

from omnicoach_sft.media.parquet_pool import ParquetFilePool


class _FakeTable:
    def __init__(self, path: Path, row_group: int) -> None:
        self.path = path
        self.row_group = row_group

    def to_pylist(self) -> list[dict[str, str | int]]:
        return [{"path": str(self.path), "row_group": self.row_group}]


class _FakeParquetFile:
    instances: ClassVar[dict[Path, _FakeParquetFile]] = {}

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.closed = False
        self.instances[self.path] = self

    def read_row_group(self, row_group: int, *, use_threads: bool):
        assert use_threads is False
        return _FakeTable(self.path, row_group)

    def close(self) -> None:
        self.closed = True


def test_parquet_file_pool_is_globally_bounded_and_pickle_safe(monkeypatch) -> None:
    _FakeParquetFile.instances = {}
    monkeypatch.setattr("omnicoach_sft.media.parquet_pool.pq.ParquetFile", _FakeParquetFile)
    pool = ParquetFilePool(max_open_files=2)
    paths = [Path(f"/tmp/media-{index}.parquet") for index in range(3)]

    assert pool.read_row_group(paths[0], 4)["row_group"] == 4
    pool.read_row_group(paths[1], 0)
    pool.read_row_group(paths[0], 5)
    pool.read_row_group(paths[2], 0)

    assert pool.open_file_count == 2
    assert _FakeParquetFile.instances[paths[0]].closed is False
    assert _FakeParquetFile.instances[paths[1]].closed is True
    assert _FakeParquetFile.instances[paths[2]].closed is False

    restored = pickle.loads(pickle.dumps(pool))
    assert restored.max_open_files == 2
    assert restored.open_file_count == 0

    pool.close()
    assert _FakeParquetFile.instances[paths[0]].closed is True
    assert _FakeParquetFile.instances[paths[2]].closed is True
