from __future__ import annotations

import pytest

from omnicoach_harness.artifacts.store import JsonlResultStore, iter_jsonl, read_json, write_json_atomic
from omnicoach_harness.cli import _model_factory
from omnicoach_harness.runtime import io as runtime_io
from omnicoach_harness.runtime.config import RunConfig
from omnicoach_harness.runtime.io import load_actor_runs, load_run_keys, load_task_ids
from omnicoach_harness.runtime.sampling import rollout_seed


def test_load_task_ids_from_analysis_jsonl(tmp_path) -> None:
    path = tmp_path / "analysis.jsonl"
    path.write_text('{"task_id":"a"}\n{"task_id":"b"}\n', encoding="utf-8")
    assert load_task_ids(path) == {"a", "b"}


def test_load_run_keys_from_jsonl(tmp_path) -> None:
    path = tmp_path / "run_keys.jsonl"
    path.write_text('{"run_key":"a#r0"}\n{"run_key":"b#r2"}\n', encoding="utf-8")
    assert load_run_keys(path) == {"a#r0", "b#r2"}


def test_load_actor_runs_filters_before_decoding_source_groups(tmp_path, monkeypatch) -> None:
    path = tmp_path / "actor_runs.jsonl"
    path.write_text(
        '{"episode":{"task_id":"keep","rollout_index":0}}\n'
        '{"episode":{"task_id":"drop","rollout_index":0}}\n'
        '{"episode":{"task_id":"keep","rollout_index":1}}\n',
        encoding="utf-8",
    )

    decoded = 0
    original_loads = runtime_io.json.loads

    def counting_loads(value):
        nonlocal decoded
        decoded += 1
        return original_loads(value)

    monkeypatch.setattr(runtime_io.json, "loads", counting_loads)
    groups = load_actor_runs(path, task_ids={"keep"})

    assert set(groups) == {"keep"}
    assert [row["episode"]["rollout_index"] for row in groups["keep"]] == [0, 1]
    assert decoded == 2


def test_run_config_defaults_to_unlimited_429_retries(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """stage: actor
route: {entrypoint: package.route:build}
io: {input: input.json, output_dir: output}
provider: {model: model}
execution: {workers: 1}
""",
        encoding="utf-8",
    )

    config = RunConfig.load(path)

    assert config.retry_429_forever is True
    assert config.request_timeout_seconds == 300.0
    assert config.sampling_seed_base is None


def test_run_config_loads_explicit_sampling_contract(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """stage: actor
route: {entrypoint: package.route:build}
io: {input: input.json, output_dir: output}
provider: {model: model}
execution: {workers: 1, rollouts: 3}
sampling: {base_seed: 20260814, temperature: 1.0, top_p: 0.95}
""",
        encoding="utf-8",
    )

    config = RunConfig.load(path)

    assert config.sampling_seed_base == 20260814
    assert config.temperature == 1.0
    assert config.top_p == 0.95


def test_run_config_rejects_typos_in_operational_and_removed_provider_keys(tmp_path) -> None:
    bad_op = tmp_path / "bad_op.yaml"
    bad_op.write_text(
        """stage: actor
route: {entrypoint: package.route:build}
io: {input: input.json, output_dir: output}
provider: {model: model, base_url: http://127.0.0.1:8000/v1}
operational: {fast_resuem: true}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported operational options"):
        RunConfig.load(bad_op)

    ghost_provider = tmp_path / "ghost_provider.yaml"
    ghost_provider.write_text(
        """stage: actor
route: {entrypoint: package.route:build}
io: {input: input.json, output_dir: output}
provider: {model: model, base_url: http://127.0.0.1:8000/v1, retry_interval_seconds: 1.0, max_inflight: 4}
""",
        encoding="utf-8",
    )
    cfg = RunConfig.load(ghost_provider)
    with pytest.raises(ValueError, match="unsupported provider options"):
        _model_factory(cfg)


def test_rollout_seed_is_stable_and_distinct_within_question() -> None:
    values = [
        rollout_seed(base_seed=20260814, task_id="question", rollout_index=index, rollouts=3)
        for index in range(3)
    ]

    assert len(set(values)) == 3
    assert values == [
        rollout_seed(base_seed=20260814, task_id="question", rollout_index=index, rollouts=3)
        for index in range(3)
    ]


def test_jsonl_store_replace_archives_superseded_attempt(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    history = tmp_path / "retry_history.jsonl"
    store = JsonlResultStore(path, key_field="run_key")
    failed = {"run_key": "case#r0", "episode": {"status": "error"}}
    completed = {"run_key": "case#r0", "episode": {"status": "completed"}}
    store.append(failed)

    store.replace(completed, archive_path=history)

    assert list(iter_jsonl(path)) == [completed]
    archived = list(iter_jsonl(history))
    assert len(archived) == 1
    assert archived[0]["superseded_key"] == "case#r0"
    assert archived[0]["artifact"] == failed


def test_jsonl_store_replace_many_archives_and_rewrites_once(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    history = tmp_path / "retry_history.jsonl"
    store = JsonlResultStore(path, key_field="run_key")
    failed = [{"run_key": f"case-{index}#r0", "episode": {"status": "error"}} for index in range(3)]
    completed = [{"run_key": f"case-{index}#r0", "episode": {"status": "completed"}} for index in range(3)]
    for row in failed:
        store.append(row)

    store.replace_many(completed, archive_path=history)

    assert list(iter_jsonl(path)) == completed
    archived = list(iter_jsonl(history))
    assert [row["superseded_key"] for row in archived] == [row["run_key"] for row in failed]
    assert [row["artifact"] for row in archived] == failed


def test_artifact_roundtrip_preserves_nested_tool_and_span_order(tmp_path) -> None:
    from omnicoach_harness.artifacts.schema import canonical_json

    call = {
        "name": "get_multi_span",
        "arguments": {
            "spans": [
                {"modality": "clip", "start": "90s", "end": "100s", "level": "detail"},
                {"modality": "audio", "start": "10s", "end": "20s"},
            ]
        },
    }
    row = {"run_key": "case#r0", "call": call}
    store = JsonlResultStore(tmp_path / "runs.jsonl", key_field="run_key")
    store.append(row)
    history = tmp_path / "history.jsonl"
    store.replace(row, archive_path=history)
    write_json_atomic(tmp_path / "snapshot.json", row)
    values = [
        next(iter(iter_jsonl(store.path))),
        next(iter(iter_jsonl(history)))["artifact"],
        read_json(tmp_path / "snapshot.json"),
    ]
    for value in values:
        spans = value["call"]["arguments"]["spans"]
        assert list(spans[0]) == ["modality", "start", "end", "level"]
        assert list(spans[1]) == ["modality", "start", "end"]
        assert [span["start"] for span in spans] == ["90s", "10s"]
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
