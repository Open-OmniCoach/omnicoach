"""`ActorBatchSession` and `run_actor_batch`: concurrent episodes with resumable persistence."""

from __future__ import annotations

import concurrent.futures
import hashlib
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..artifacts.schema import ActorRunArtifact
from ..artifacts.store import JsonlResultStore, iter_jsonl, read_json, write_json_atomic
from ..domain.task import TaskInput, TaskLabel
from ..media.cache import safe_id
from ..providers.protocol import ModelClient
from ..providers.telemetry import record
from .actor_runner import ActorRunner
from .config import BatchOptions, MediaOptions
from .route import ActorRoute
from .sampling import rollout_seed


class ActorBatchSession:
    """One shared actor output boundary with resumable persistence and isolated attempt dirs."""

    def __init__(
        self,
        *,
        route: ActorRoute,
        model_factory: Callable[[], ModelClient],
        model_name: str,
        max_tokens: int,
        options: BatchOptions,
        media: MediaOptions | None = None,
    ) -> None:
        self.options = options
        self.media = media or MediaOptions()
        self.root = Path(options.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.route = route
        self.model_factory = model_factory
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.contract = route.contract_artifact()
        _ensure_actor_contract(self.root, self.contract.as_dict())
        _ensure_sampling_contract(
            self.root,
            rollouts=options.rollouts,
            sampling_seed_base=options.sampling_seed_base,
            temperature=options.temperature,
            top_p=options.top_p,
        )
        _ensure_run_key_filter_contract(self.root, options.run_key_filter)
        store_type: type[JsonlResultStore] = JsonlResultStore
        if options.fast_resume:
            from ..artifacts.indexed_actor_store import IndexedActorStore

            store_type = IndexedActorStore
        self.store = store_type(self.root / "actor_runs.jsonl", key_field="run_key")
        self.work_dir = Path(self.media.work_dir) if self.media.work_dir is not None else self.root / "tmp"
        self.shared_media_cache = (
            Path(self.media.media_cache_dir) if self.media.media_cache_dir is not None else self.root / "media_cache"
        )
        self.start_semaphore = (
            threading.BoundedSemaphore(options.max_concurrent_starts)
            if options.max_concurrent_starts is not None
            else None
        )

    def pending(
        self,
        cases: Iterable[tuple[TaskInput, TaskLabel]],
        *,
        rollouts: int,
        retry_errors: bool,
    ) -> tuple[list[tuple[TaskInput, TaskLabel, int]], set[str]]:
        case_list = list(cases)
        run_key_filter = self.options.run_key_filter
        retry_keys = {
            f"{task.task_id}#r{rollout_index}"
            for rollout_index in range(rollouts)
            for task, _label in case_list
            if run_key_filter is None or f"{task.task_id}#r{rollout_index}" in run_key_filter
            if retry_errors
            and (stored := self.store.get(f"{task.task_id}#r{rollout_index}")) is not None
            and stored.get("episode", {}).get("status") == "error"
        }
        tasks = [
            (task, label, rollout_index)
            for rollout_index in range(rollouts)
            for task, label in case_list
            if run_key_filter is None or f"{task.task_id}#r{rollout_index}" in run_key_filter
            if not self.store.contains(f"{task.task_id}#r{rollout_index}")
            or f"{task.task_id}#r{rollout_index}" in retry_keys
        ]
        return tasks, retry_keys

    def execute(
        self,
        item: tuple[TaskInput, TaskLabel, int],
        *,
        rollouts: int,
        attempts: int,
        retry_errors: bool,
    ) -> dict[str, Any]:
        task, label, rollout_index = item
        run_key = f"{task.task_id}#r{rollout_index}"
        seed = (
            rollout_seed(
                base_seed=self.options.sampling_seed_base,
                task_id=task.task_id,
                rollout_index=rollout_index,
                rollouts=rollouts,
            )
            if self.options.sampling_seed_base is not None
            else None
        )
        episode = None
        run_root = self.work_dir / safe_id(run_key)
        for attempt_index in range(attempts):
            attempt_dir = run_root / f"attempt_{attempt_index:02d}_{uuid.uuid4().hex[:8]}"
            runner = ActorRunner(
                route=self.route,
                model=self.model_factory(),
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                sampling_seed=seed,
                temperature=self.options.temperature,
                top_p=self.options.top_p,
                ffmpeg=self.media.ffmpeg,
                ffprobe=self.media.ffprobe,
                start_semaphore=self.start_semaphore,
                defer_scoring=self.options.defer_scoring,
            )
            episode = runner.run(
                task,
                label,
                rollout_index=rollout_index,
                work_dir=attempt_dir,
                media_cache_dir=self.shared_media_cache,
            )
            if episode.status != "error" or not retry_errors:
                break
        assert episode is not None
        artifact = ActorRunArtifact(episode=episode)
        return {"run_key": run_key, **artifact.as_dict()}

    def persist(self, result: Mapping[str, object], *, replace: bool) -> None:
        if replace:
            self.store.replace(result, archive_path=self.root / "actor_retry_history.jsonl")
        else:
            self.store.append(result)


def _defer_tasks(
    tasks: list[tuple[TaskInput, TaskLabel, int]],
    defer_run_keys: frozenset[str] | None,
) -> Iterable[tuple[TaskInput, TaskLabel, int]]:
    """Keep all pending tasks, moving explicitly deferred keys to the end."""

    if not defer_run_keys:
        yield from tasks
        return
    deferred = []
    for item in tasks:
        key = f"{item[0].task_id}#r{item[2]}"
        if key in defer_run_keys:
            deferred.append(item)
        else:
            yield item
    yield from deferred


def run_actor_batch(
    cases: Iterable[tuple[TaskInput, TaskLabel]],
    *,
    route: ActorRoute,
    model_factory: Callable[[], ModelClient],
    model_name: str,
    max_tokens: int,
    options: BatchOptions,
    media: MediaOptions | None = None,
) -> None:
    session = ActorBatchSession(
        route=route,
        model_factory=model_factory,
        model_name=model_name,
        max_tokens=max_tokens,
        options=options,
        media=media,
    )
    tasks, retry_keys = session.pending(cases, rollouts=options.rollouts, retry_errors=options.retry_errors)
    stop_path = Path(options.intake_stop_path) if options.intake_stop_path else session.root / "STOP_INTAKE"

    def execute(item: tuple[TaskInput, TaskLabel, int]) -> dict[str, Any]:
        return session.execute(
            item,
            rollouts=options.rollouts,
            attempts=options.attempts,
            retry_errors=options.retry_errors,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as executor:
        items = iter(_defer_tasks(tasks, options.defer_run_keys))
        futures: dict[concurrent.futures.Future[dict[str, Any]], bool] = {}
        remaining_replacements = len(retry_keys)
        pending_replacements: list[dict[str, Any]] = []

        def fill_window() -> None:
            while len(futures) < 2 * options.workers and not stop_path.exists():
                item = next(items, None)
                if item is None:
                    break
                futures[executor.submit(execute, item)] = f"{item[0].task_id}#r{item[2]}" in retry_keys

        fill_window()
        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                replacing = futures.pop(future)
                result = future.result()
                if replacing:
                    pending_replacements.append(result)
                    remaining_replacements -= 1
                    if (
                        len(pending_replacements) >= options.retry_replacement_batch_size
                        or remaining_replacements == 0
                    ):
                        session.store.replace_many(
                            pending_replacements,
                            archive_path=session.root / "actor_retry_history.jsonl",
                        )
                        pending_replacements.clear()
                else:
                    session.store.append(result)
                record("actor", str(result.get("episode", {}).get("status", "unknown")))
                del result
            del done, future
            fill_window()
        session.store.replace_many(
            pending_replacements,
            archive_path=session.root / "actor_retry_history.jsonl",
        )


def _bind_contract(
    root: Path,
    name: str,
    expected: Mapping[str, object],
    *,
    check_existing_runs: Callable[[Path], None] | None = None,
) -> None:
    """Bind a run directory to one treatment, or refuse to resume it under another."""

    path = root / name
    if path.is_file():
        existing = read_json(path)
        if existing != dict(expected):
            raise ValueError(f"{name} mismatch: stored={existing!r}, requested={dict(expected)!r}")
        return
    runs_path = root / "actor_runs.jsonl"
    if runs_path.is_file() and runs_path.stat().st_size:
        if check_existing_runs is None:
            raise ValueError(
                f"cannot introduce {name} into an existing actor run directory; use a new output directory"
            )
        check_existing_runs(runs_path)
    write_json_atomic(path, dict(expected))


def _ensure_actor_contract(root: Path, artifact: Mapping[str, object]) -> None:
    _bind_contract(root, "actor_contract.json", artifact)


def _ensure_sampling_contract(
    root: Path,
    *,
    rollouts: int,
    sampling_seed_base: int | None,
    temperature: float | None,
    top_p: float | None,
) -> None:
    if sampling_seed_base is None and temperature is None and top_p is None:
        return

    def check(runs_path: Path) -> None:
        for row in iter_jsonl(runs_path):
            episode = row.get("episode") or {}
            expected_seed = (
                rollout_seed(
                    base_seed=sampling_seed_base,
                    task_id=str(episode.get("task_id") or ""),
                    rollout_index=int(episode.get("rollout_index")),
                    rollouts=rollouts,
                )
                if sampling_seed_base is not None
                else None
            )
            sampling = (episode.get("metadata") or {}).get("sampling") or {}
            if sampling != {"rollout_seed": expected_seed, "temperature": temperature, "top_p": top_p}:
                raise ValueError("existing actor runs have mismatched sampling metadata")

    _bind_contract(
        root,
        "actor_sampling_contract.json",
        {
            "schema_version": "actor_sampling_contract_v1",
            "seed_derivation": "fnv1a64_task_bucket_contiguous_rollout_v1",
            "rollouts": rollouts,
            "base_seed": sampling_seed_base,
            "temperature": temperature,
            "top_p": top_p,
        },
        check_existing_runs=check,
    )


def _ensure_run_key_filter_contract(root: Path, run_keys: frozenset[str] | None) -> None:
    path = root / "actor_run_key_filter_contract.json"
    if run_keys is None:
        if path.is_file():
            raise ValueError("actor run directory requires its bound run-key filter")
        return
    normalized = sorted(set(run_keys))
    if not normalized:
        raise ValueError("run-key filter must not be empty")
    _bind_contract(
        root,
        "actor_run_key_filter_contract.json",
        {
            "schema_version": "actor_run_key_filter_contract_v1",
            "count": len(normalized),
            "sha256": hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest(),
        },
    )
