"""Shared configuration and task-JSONL helpers for multi-turn Qwen3-Omni agent recipes.

Maintains strict isolation: never imports recipe-specific packages or credit algorithms.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from slime_ext.agent.protocol import DEFAULT_MEDIA_SETTINGS


def parse_bool_env(raw: str, name: str) -> bool:
    """Parse a strict boolean environment variable (``0``/``1`` or ``false``/``true``)."""
    val = raw.strip().lower()
    if val not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be 0/1 or false/true, got {raw!r}")
    return val in {"1", "true"}


def _deep_merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def resolve_agent_config(
    path: str | Path,
    environment: Mapping[str, str] | None = None,
    *,
    forbidden_config_keys: tuple[str, ...] = (),
    forbidden_agent_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Load, merge default media settings, apply env overrides, and validate ``agent_config``."""
    env = dict(environment or {})
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "agent_config" not in config:
        raise ValueError("config must contain top-level 'agent_config'")

    agent = config["agent_config"]
    if not isinstance(agent, dict):
        raise ValueError("agent_config must be a mapping")
    for key in forbidden_config_keys:
        if key in config:
            raise ValueError(f"config must not define forbidden section {key!r}")
    for key in forbidden_agent_keys:
        if key in agent:
            raise ValueError(f"agent_config must not define forbidden key {key!r}")

    if "MAX_TURNS" in env:
        agent["max_turns"] = int(env["MAX_TURNS"])
    if "FORMAT_REWARD" in env:
        agent["format_reward"] = float(env["FORMAT_REWARD"])
    if "TG_COMPLETE_EVENT_GATE" in env:
        agent["tg_complete_event_gate"] = parse_bool_env(
            env["TG_COMPLETE_EVENT_GATE"], "TG_COMPLETE_EVENT_GATE"
        )

    max_turns = agent.get("max_turns")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("agent_config.max_turns must be an int >= 1")
    fmt = agent.get("format_reward", 0.0)
    if isinstance(fmt, bool) or not isinstance(fmt, (int, float)) or not math.isfinite(fmt) or fmt < 0:
        raise ValueError("agent_config.format_reward must be finite and >= 0")

    raw_media = agent.get("media")
    if raw_media is not None and not isinstance(raw_media, Mapping):
        raise ValueError("agent_config.media must be a mapping when provided")
    media = _deep_merge_dict(DEFAULT_MEDIA_SETTINGS, raw_media or {})
    agent["media"] = media

    init_fps = float(media["initial"]["fps"])
    if not math.isfinite(init_fps) or init_fps <= 0.0:
        raise ValueError("media.initial.fps must be > 0")
    high_max_sec = float(media["resolutions"]["high"]["max_seconds"])
    if not math.isfinite(high_max_sec) or high_max_sec <= 0.0:
        raise ValueError("media.resolutions.high.max_seconds must be > 0")

    mm_cfg = config.setdefault("sglang_mm_process_config", {})
    if isinstance(mm_cfg, dict):
        img_cfg = mm_cfg.setdefault("image", {})
        if isinstance(img_cfg, dict) and "max_pixels" not in img_cfg:
            img_cfg["max_pixels"] = int(media["resolutions"]["high"]["max_pixels"])

    return config


def prepare_agent_jsonl(
    source: Path,
    output: Path,
    *,
    check_files: bool = True,
    metadata_validator: Callable[[Mapping[str, Any], int], None] | None = None,
) -> int:
    """Validate agent task JSONL rows, invoke optional recipe validator, and resolve media paths."""
    if source.resolve() == output.resolve():
        raise ValueError("source and output paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with source.open(encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line_no, line in enumerate(reader, 1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row.get("prompt"), (str, list)) or "label" not in row:
                raise ValueError(f"line {line_no}: task requires prompt and label")
            metadata = dict(row.get("metadata") or {})
            for key in ("task_id", "media_id", "source_name"):
                if not isinstance(metadata.get(key), str) or not metadata[key]:
                    raise ValueError(f"line {line_no}: metadata.{key} must be a non-empty string")
            if metadata["task_id"] in seen:
                raise ValueError(f"duplicate task_id: {metadata['task_id']}")
            seen.add(metadata["task_id"])
            if metadata.get("answer_type") not in {"mcq", "span_set"}:
                raise ValueError(f"line {line_no}: answer_type must be 'mcq' or 'span_set'")

            if metadata_validator is not None:
                metadata_validator(metadata, line_no)

            media = dict(metadata.get("media") or {})
            if type(media.get("row_group")) is not int or media["row_group"] < 0:
                raise ValueError(f"line {line_no}: media.row_group must be a non-negative int")
            for key in ("path", "contract"):
                p = Path(media[key])
                p = p if p.is_absolute() else (source.parent / p)
                if check_files and not p.is_file():
                    raise FileNotFoundError(p)
                media[key] = str(p.resolve())
            metadata["media"] = media
            row["metadata"] = metadata
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not seen:
        raise ValueError("input JSONL contains no tasks")
    return len(seen)


__all__ = [
    "parse_bool_env",
    "prepare_agent_jsonl",
    "resolve_agent_config",
]
