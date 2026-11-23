"""Operational regression checks: no live provider, services, or process signals."""

from __future__ import annotations

import json
import tracemalloc
from collections import Counter

import pytest

from omnicoach_harness.artifacts.store import JsonlResultStore
from omnicoach_harness.domain.task import QuestionType, TaskInput, TaskLabel
from omnicoach_harness.protocol.route import build
from omnicoach_harness.providers import telemetry
from omnicoach_harness.runtime import batch_runner
from omnicoach_harness.runtime.config import BatchOptions


def test_ten_thousand_large_results_release_completed_futures_and_index(tmp_path, monkeypatch) -> None:
    cases = [
        (
            TaskInput(
                task_id=str(i),
                video_path="unused",
                question="q",
                question_type=QuestionType.MCQ,
                options=("A", "B"),
            ),
            TaskLabel(answer="A"),
        )
        for i in range(10000)
    ]

    def execute(self, item, **kwargs):
        del self, kwargs
        return {"run_key": f"{item[0].task_id}#r0", "payload": item[0].task_id + "x" * 65536}

    monkeypatch.setattr(batch_runner.ActorBatchSession, "execute", execute)
    tracemalloc.start()
    batch_runner.run_actor_batch(
        cases,
        route=build(),
        model_factory=lambda: None,
        model_name="fake",
        max_tokens=16,
        options=BatchOptions(output_dir=tmp_path, workers=8, rollouts=1),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 32 * 1024**2
    store = JsonlResultStore(tmp_path / "actor_runs.jsonl", key_field="run_key")
    assert len(store._offsets) == 10000
    assert store.get("9999#r0")["payload"].startswith("9999")
    store.replace_many([{"run_key": "1#r0", "payload": "fixed"}], archive_path=tmp_path / "history.jsonl")
    reopened = JsonlResultStore(store.path, key_field="run_key")
    assert reopened.get("1#r0")["payload"] == "fixed"
    assert len(reopened._offsets) == 10000
    with pytest.raises(ValueError, match="duplicate"):
        reopened.append({"run_key": "1#r0"})
    assert json.loads((tmp_path / "history.jsonl").read_text())["artifact"]["payload"].startswith("1")
    # A torn tail is rejected explicitly and never silently discarded.
    with store.path.open("ab") as handle:
        handle.write(b'{"run_key":')
    with pytest.raises(json.JSONDecodeError):
        JsonlResultStore(store.path, key_field="run_key")


def test_entry_telemetry_debits_attempt_budget_once(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "_counts", Counter())
    monkeypatch.setattr(telemetry, "_limit", 2)
    monkeypatch.setattr(telemetry, "_deadline", None)
    for _ in range(2):
        telemetry.record("model", "attempt", entry="unit")
    with pytest.raises(telemetry.ProviderBudgetExhausted):
        telemetry.record("model", "attempt", entry="unit")
    telemetry.record("model", "success", entry="unit")
    counts = telemetry.snapshot()
    assert counts["attempt"] == counts["model:attempt"] == counts["model:entry:unit:attempt"] == 2
    assert counts["model:success"] == counts["model:entry:unit:success"] == 1
