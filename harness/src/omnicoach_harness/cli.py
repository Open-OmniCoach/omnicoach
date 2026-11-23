"""The `omnicoach-harness` command line: `doctor`, `describe` and `actor`."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable

from .artifacts.store import iter_jsonl
from .media.ffmpeg import require_binary
from .providers.openai_compat import OpenAICompatModelClient
from .providers.protocol import ModelClient
from .runtime.batch_runner import run_actor_batch
from .runtime.config import RunConfig
from .runtime.io import load_cases, load_run_keys, load_task_ids
from .runtime.loader import build_route


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "describe":
        return _describe(args)
    if args.command == "actor":
        return _run_actor_command(RunConfig.load(args.config))
    parser.error(f"unknown command: {args.command}")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnicoach-harness")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check the system media binaries")
    doctor.add_argument("--ffmpeg", default="ffmpeg")
    doctor.add_argument("--ffprobe", default="ffprobe")
    describe = commands.add_parser("describe", help="print one route-owned frozen contract")
    describe.add_argument("--entrypoint", required=True)
    actor = commands.add_parser("actor", help="run or resume actor artifacts")
    actor.add_argument("--config", required=True)
    return parser


def _doctor(args: argparse.Namespace) -> int:
    values = {"ffmpeg": require_binary(args.ffmpeg), "ffprobe": require_binary(args.ffprobe)}
    print(json.dumps(values, ensure_ascii=False, indent=2))
    return 0


def _describe(args: argparse.Namespace) -> int:
    route = build_route(args.entrypoint)
    if hasattr(route, "contract_artifact"):
        value = route.contract_artifact().as_dict()
    elif hasattr(route, "contract_snapshot"):
        value = {"route_id": route.route_id, "contract": route.contract_snapshot()}
    else:
        raise TypeError("entrypoint object exposes no route contract")
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_actor_command(config: RunConfig) -> int:
    media_opts = config.media_options()
    media = None
    if media_opts.workers is not None:
        from .media.executor import configure_media_executor

        media = configure_media_executor(
            workers=media_opts.workers,
            queue_size=media_opts.queue_size,
            decode_bytes=media_opts.decode_bytes,
        )
    from .providers.telemetry import configure_budget

    configure_budget(
        attempts=config.operational.provider_attempt_budget,
        seconds=config.operational.provider_time_budget_seconds,
    )
    telemetry = None
    if config.operational.telemetry:
        from .runtime.telemetry import RunTelemetry

        telemetry = RunTelemetry(config.operational.telemetry_dir or config.output_dir, media)
    try:
        return _actor(config)
    finally:
        if telemetry is not None:
            telemetry.close()
        if media is not None:
            media.close()


def _actor(config: RunConfig) -> int:
    if config.stage != "actor":
        raise ValueError("actor command requires stage: actor")
    route = build_route(config.route_entrypoint)
    cases = load_cases(config.input_path)
    op = config.operational
    if op.task_filter_path:
        task_ids = load_task_ids(op.task_filter_path)
        cases = [(task, label) for task, label in cases if task.task_id in task_ids]
        if len(cases) != len(task_ids):
            present = {task.task_id for task, _ in cases}
            raise ValueError(f"task filter contains {len(task_ids - present)} IDs absent from actor input")
    run_key_filter = load_run_keys(op.run_key_filter_path) if op.run_key_filter_path else None
    defer_run_keys = (
        {str(row["run_key"]) for row in iter_jsonl(op.defer_run_keys_path)}
        if op.defer_run_keys_path
        else None
    )
    if run_key_filter is not None:
        available = {
            f"{task.task_id}#r{rollout_index}"
            for task, _label in cases
            for rollout_index in range(config.rollouts)
        }
        unknown = run_key_filter - available
        if unknown:
            raise ValueError(f"run-key filter contains {len(unknown)} keys absent from actor input/rollouts")
    run_actor_batch(
        cases,
        route=route,
        model_factory=_model_factory(config),
        model_name=config.model,
        max_tokens=config.max_tokens,
        options=config.batch_options(run_key_filter=run_key_filter, defer_run_keys=defer_run_keys),
        media=config.media_options(),
    )
    return 0


def _model_factory(config: RunConfig) -> Callable[[], ModelClient]:
    if config.provider_backend != "openai_compat":
        raise ValueError(f"unsupported provider.backend: {config.provider_backend!r}")
    options = dict(config.provider_options)
    base_urls = options.pop("base_urls", None) or options.pop("base_url", None)
    if not base_urls:
        raise ValueError("provider.base_url is required for openai_compat")
    max_429_retries = options.pop("max_429_retries", 100)
    audio_part_type = str(options.pop("audio_part_type", None) or "audio_url")
    client = OpenAICompatModelClient(
        base_urls=base_urls,
        api_key=str(options.pop("api_key", None) or os.environ.get("OPENAI_API_KEY") or "EMPTY"),
        retries=config.retries,
        retry_429_forever=config.retry_429_forever,
        max_429_retries=int(max_429_retries),
        timeout_seconds=config.request_timeout_seconds,
        audio_part_type=audio_part_type,
        extra_body=options.pop("extra_body", None),
    )
    if options:
        raise ValueError(f"unsupported provider options: {sorted(options)}")
    return lambda: client


if __name__ == "__main__":
    raise SystemExit(main())
