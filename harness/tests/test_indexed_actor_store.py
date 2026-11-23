from __future__ import annotations

import threading

import pytest

from omnicoach_harness.artifacts.indexed_actor_store import IndexedActorStore
from omnicoach_harness.artifacts.store import JsonlResultStore


def test_resume_recovers_only_unindexed_tail(tmp_path, monkeypatch) -> None:
    path = tmp_path / "actor_runs.jsonl"
    store = IndexedActorStore(path, key_field="run_key")
    store.append({"run_key": "one", "episode": {"status": "completed"}})
    # Emulate data fsync completed but journal append interrupted.
    plain = JsonlResultStore(path, key_field="run_key")
    plain.append({"run_key": "two", "episode": {"status": "error"}})
    recovered = IndexedActorStore(path, key_field="run_key")
    assert recovered.statuses == {"one": "completed", "two": "error"}
    assert recovered.get("two")["episode"]["status"] == "error"
    with pytest.raises(ValueError, match="duplicate"):
        recovered.append({"run_key": "one", "episode": {"status": "completed"}})
    # A complete index resumes without opening the large data file.
    original = type(path).open

    def guarded(self, *args, **kwargs):
        assert self != path, "unnecessary full data read on indexed resume"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", guarded)
    again = IndexedActorStore(path, key_field="run_key")
    assert again.contains("one")
    assert again.contains("two")


def test_torn_derived_index_is_repaired_without_changing_data(tmp_path) -> None:
    path = tmp_path / "actor_runs.jsonl"
    store = IndexedActorStore(path, key_field="run_key")
    store.append({"run_key": "one", "episode": {"status": "completed"}})
    before = path.read_bytes()
    with store.index_path.open("ab") as f:
        f.write(b'{"key":')
    resumed = IndexedActorStore(path, key_field="run_key")
    assert resumed.contains("one")
    assert path.read_bytes() == before
    with path.open("ab") as f:
        f.write(b'{"run_key":')
    with pytest.raises(ValueError, match="incomplete actor output tail"):
        IndexedActorStore(path, key_field="run_key")


def test_indexed_actor_store_concurrent_appends_stay_synchronized(tmp_path) -> None:
    path = tmp_path / "actor_runs.jsonl"
    store = IndexedActorStore(path, key_field="run_key")
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5)
            for idx in range(25):
                store.append({"run_key": f"w{worker_id}-k{idx}#r0", "episode": {"status": "completed"}})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []

    reopened = IndexedActorStore(path, key_field="run_key")
    assert len(reopened.statuses) == 100
    assert reopened.get("w3-k24#r0") == {"run_key": "w3-k24#r0", "episode": {"status": "completed"}}
