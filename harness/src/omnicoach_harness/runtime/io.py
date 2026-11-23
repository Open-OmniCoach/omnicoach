"""Readers for task files and previously written run artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..artifacts.store import iter_jsonl
from ..domain.task import TaskInput, TaskLabel, split_task_record


def iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        yield from iter_jsonl(source)
        return
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict) and isinstance(value.get("data"), list):
        rows = value["data"]
    else:
        raise ValueError(f"{source} must contain a JSON array, a data array, or JSONL objects")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{source}: item {index} is not an object")
        yield row


def load_cases(path: str | Path) -> list[tuple[TaskInput, TaskLabel]]:
    cases = [split_task_record(item) for item in iter_records(path)]
    ids = [task.task_id for task, _ in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("task input contains duplicate task_id values")
    return cases


def load_task_ids(path: str | Path) -> set[str]:
    task_ids: set[str] = set()
    for row in iter_records(path):
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"{path} contains a record without task_id")
        task_ids.add(task_id)
    return task_ids


def load_run_keys(path: str | Path) -> set[str]:
    run_keys: set[str] = set()
    for row in iter_records(path):
        run_key = str(row.get("run_key") or "").strip()
        if not run_key:
            raise ValueError(f"{path} contains a record without run_key")
        run_keys.add(run_key)
    if not run_keys:
        raise ValueError(f"{path} contains no run keys")
    return run_keys


def load_actor_runs(
    path: str | Path,
    *,
    task_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    source = Path(path)
    rows = (
        _iter_selected_actor_jsonl(source, task_ids)
        if task_ids is not None and source.suffix.casefold() == ".jsonl"
        else iter_records(source)
    )
    for row in rows:
        task_id = _actor_task_id(row)
        if not task_id:
            raise ValueError("actor-run artifact has no task_id")
        if task_ids is not None and task_id not in task_ids:
            continue
        groups.setdefault(task_id, []).append(row)
    return groups


def _iter_selected_actor_jsonl(path: Path, task_ids: set[str]) -> Iterable[dict[str, Any]]:
    if not task_ids:
        return
    encoded_ids = "|".join(re.escape(json.dumps(task_id, ensure_ascii=False)) for task_id in sorted(task_ids))
    task_pattern = re.compile(rf'"task_id"\s*:\s*(?:{encoded_ids})')
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or task_pattern.search(text) is None:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield value


def _actor_task_id(row: Mapping[str, Any]) -> str:
    episode = row.get("episode")
    if isinstance(episode, Mapping):
        return str(episode.get("task_id") or "")
    return str(row.get("task_id") or "")
