"""Single-turn Qwen3-Omni multimodal dataset normalization and ablation helpers.

STRICT ISOLATION: ZERO ``slime_ext.agent`` or cross-example imports.
Aligned with latest ``THUDM/slime`` ``79aa41a8`` (no ``rollout_global_dataset``).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MEDIA_PLACEHOLDERS = {
    "image": "<image>",
    "audio": "<audio>",
    "video": "<video>",
}

_ABLATION_DROPS = {
    "none": frozenset(),
    "no-audio": frozenset({"audio"}),
    "no-image": frozenset({"image"}),
    "text-only": frozenset({"image", "audio", "video"}),
}

_PLACEHOLDER_RE = {
    "image": re.compile(r"(?:<image>|<\|vision_start\|>\s*<\|image_pad\|>\s*<\|vision_end\|>|<\|image_pad\|>)"),
    "audio": re.compile(r"(?:<audio>|<\|audio_start\|>.*?<\|audio_end\|>|<\|audio_pad\|>|<\|AUDIO\|>)", re.DOTALL),
    "video": re.compile(r"(?:<video>|<\|vision_start\|>\s*<\|video_pad\|>\s*<\|vision_end\|>|<\|video_pad\|>)"),
}


def build_multimodal_messages(
    row: Mapping[str, Any],
    *,
    prompt_key: str = "prompt",
    multimodal_keys: Mapping[str, str] | None = None,
    ablation_mode: str = "none",
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build Qwen3-Omni chat messages from a single-turn dataset row."""
    if ablation_mode not in _ABLATION_DROPS:
        raise ValueError(f"unknown ablation_mode {ablation_mode!r}")
    drop = _ABLATION_DROPS[ablation_mode]
    raw_prompt = row.get(prompt_key)
    if raw_prompt is None:
        raise KeyError(f"row missing prompt key {prompt_key!r}")

    if isinstance(raw_prompt, str):
        messages: list[dict[str, Any]] = [{"role": "user", "content": raw_prompt}]
    elif isinstance(raw_prompt, list):
        messages = copy.deepcopy(raw_prompt)
    else:
        raise ValueError(f"unsupported prompt type {type(raw_prompt).__name__}")

    if row.get("system") and not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": str(row["system"])}] + messages

    mm_keys = dict(multimodal_keys or {"image": "images", "audio": "audios", "video": "videos"})
    queues: dict[str, list[Any]] = {}
    for mtype in MEDIA_PLACEHOLDERS:
        key = mm_keys.get(mtype)
        val = row.get(key) if key else None
        if val is None or mtype in drop:
            queues[mtype] = []
        elif isinstance(val, list):
            queues[mtype] = list(val)
        else:
            queues[mtype] = [val]

    placeholder_to_type = {ph: mt for mt, ph in MEDIA_PLACEHOLDERS.items()}
    split_pat = re.compile("(" + "|".join(re.escape(p) for p in placeholder_to_type) + ")")

    def _resolve(item: Any) -> Any:
        if isinstance(item, str) and base_dir is not None:
            if not item.startswith(("data:", "http://", "https://", "file://")):
                p = Path(item)
                return str(p if p.is_absolute() else (base_dir / p).resolve())
        return item

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            for dropped in drop:
                content = _PLACEHOLDER_RE[dropped].sub("", content)
            parts: list[dict[str, Any]] = []
            for chunk in split_pat.split(content):
                if not chunk:
                    continue
                mtype = placeholder_to_type.get(chunk)
                if mtype is None:
                    parts.append({"type": "text", "text": chunk})
                else:
                    if not queues[mtype]:
                        raise ValueError(
                            f"prompt contains {chunk} placeholder but no {mtype} item was provided"
                        )
                    parts.append({"type": mtype, mtype: _resolve(queues[mtype].pop(0))})
            msg["content"] = parts
        elif isinstance(content, list):
            kept: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    kept.append({"type": "text", "text": str(part)})
                    continue
                ptype = part.get("type")
                if ptype in drop:
                    continue
                if ptype == "text" and isinstance(part.get("text"), str):
                    txt = part["text"]
                    for dropped in drop:
                        txt = _PLACEHOLDER_RE[dropped].sub("", txt)
                    kept.append({**part, "text": txt})
                elif ptype in MEDIA_PLACEHOLDERS:
                    copied = copy.deepcopy(part)
                    if ptype in copied:
                        copied[ptype] = _resolve(copied[ptype])
                    kept.append(copied)
                else:
                    kept.append(copy.deepcopy(part))
            msg["content"] = kept
        else:
            raise ValueError(f"unsupported message content type {type(content).__name__}")

    for mtype, remaining in queues.items():
        if remaining:
            raise ValueError(
                f"multimodal count mismatch: {len(remaining)} leftover {mtype} item(s)"
            )
    return messages


def prepare_jsonl(
    source: Path,
    output: Path,
    *,
    prompt_key: str = "prompt",
    label_key: str = "label",
    ablation_mode: str = "none",
) -> int:
    """Validate single-turn Qwen3-Omni QA JSONL and normalize relative media paths."""
    if source.resolve() == output.resolve():
        raise ValueError("source and output paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_ids: set[str] = set()
    with source.open() as reader, output.open("w") as writer:
        for line_no, line in enumerate(reader, 1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if prompt_key not in row or label_key not in row:
                raise ValueError(f"line {line_no} requires {prompt_key!r} and {label_key!r}")
            meta = dict(row.get("metadata") or {})
            task_id = str(meta.get("task_id") or f"row-{line_no}")
            if task_id in seen_ids:
                raise ValueError(f"duplicate task_id {task_id!r}")
            seen_ids.add(task_id)
            meta["task_id"] = task_id
            messages = build_multimodal_messages(
                row,
                prompt_key=prompt_key,
                ablation_mode=ablation_mode,
                base_dir=source.parent,
            )
            normalized = {
                "prompt": messages,
                "label": row[label_key],
                "metadata": meta,
            }
            writer.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("input JSONL contains no tasks")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablation-mode", default="none")
    args = parser.parse_args()
    count = prepare_jsonl(args.input, args.output, ablation_mode=args.ablation_mode)
    print(f"Prepared {count} single-turn Qwen3-Omni RL prompts: {args.output}")


if __name__ == "__main__":
    main()
