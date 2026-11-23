#!/usr/bin/env python3
"""Encode weighted Agent rows with the pinned Swift/Qwen3-Omni template.

This is a pre-training gate. It loads no model weights, but it does run the
real multimodal processor so media placeholders, labels, and fractional
token-level loss scales are checked together.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from omnicoach_sft.io import iter_jsonl
from omnicoach_sft.train.dataset import AgentSharedMediaRegistry, resolve_agent_shared_source_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--tail-tokens", type=int, default=2048)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--require-recovery",
        action="store_true",
        help="Require and encode one row containing loss-masked invalid history.",
    )
    return parser.parse_args()


def _target_action(messages: list[dict[str, Any]]) -> str | None:
    if not messages:
        return None
    role = messages[-1]["role"]
    if role == "tool_call":
        return "tool_call"
    if role == "assistant" and "<answer>" in str(messages[-1].get("content") or ""):
        return "answer"
    return None


def _weight(value: Any) -> float | None:
    return None if value is None else float(value)


def validate_target_weights(row: dict[str, Any]) -> tuple[str, float]:
    messages = row["messages"]
    action = _target_action(messages)
    if action is None:
        raise ValueError("row does not end in a tool_call or <answer> target")
    target_weight = _weight(messages[-1].get("loss_scale"))
    if target_weight is None or not 0 < target_weight < 1:
        raise ValueError(f"{action} row does not carry a fractional target loss_scale: {target_weight}")
    if action == "tool_call":
        if len(messages) < 2 or messages[-2]["role"] != "assistant":
            raise ValueError("tool_call target is not preceded by its observation/think assistant message")
        assistant_weight = _weight(messages[-2].get("loss_scale"))
        if assistant_weight is None or not math.isclose(assistant_weight, target_weight, abs_tol=1e-12):
            raise ValueError(
                "tool_call target must put the same loss_scale on assistant observation/think and tool_call: "
                f"assistant={assistant_weight}, tool_call={target_weight}"
            )
    channel = row.get("channel")
    if channel not in {None, "agent"}:
        raise ValueError(f"expected channel='agent' when present, got {channel!r}")
    return action, target_weight


def _parquet_paths(source: Path) -> list[Path]:
    if source.is_file() and source.suffix == ".parquet":
        return [source]
    if source.is_dir():
        for candidate in (source / "turns", source):
            paths = sorted(candidate.glob("part-*.parquet"))
            if paths:
                return paths
    return []


def _needed_slots(messages: list[dict[str, Any]], selected: dict[str, dict[str, Any]]) -> tuple[str, bool, bool] | None:
    action = _target_action(messages)
    if action is None:
        return None
    needs_action = action not in selected
    needs_recovery = "recovery" not in selected and any(message.get("loss") is False for message in messages[:-1])
    if not needs_action and not needs_recovery:
        return None
    return action, needs_action, needs_recovery


def _find_parquet_rows(source: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    registry: AgentSharedMediaRegistry | None = None
    shared_root = resolve_agent_shared_source_root(source)
    if shared_root is not None:
        registry = AgentSharedMediaRegistry()
        registry.store_for_root(shared_root)
    try:
        for path in _parquet_paths(source):
            parquet = pq.ParquetFile(path)
            if "channel" not in parquet.schema_arrow.names:
                raise ValueError(f"Agent parquet is missing channel: {path}")
            for row_group in range(parquet.metadata.num_row_groups):
                light_rows = parquet.read_row_group(row_group, columns=["messages", "channel"]).to_pylist()
                for row_index, light_row in enumerate(light_rows):
                    needed = _needed_slots(light_row["messages"], selected)
                    if needed is None:
                        continue
                    try:
                        validate_target_weights(light_row)
                    except ValueError:
                        continue
                    action, needs_action, needs_recovery = needed
                    full_row = parquet.read_row_group(row_group).slice(row_index, 1).to_pylist()[0]
                    if registry is not None:
                        full_row = registry.resolve_row(full_row)
                    if needs_action:
                        selected[action] = full_row
                    if needs_recovery:
                        selected["recovery"] = full_row
                    if set(selected) == {"tool_call", "answer", "recovery"}:
                        return selected
        return selected
    finally:
        if registry is not None:
            registry.close()


def load_representative_rows(source: Path) -> dict[str, dict[str, Any]]:
    parquet_rows = _find_parquet_rows(source)
    if parquet_rows:
        return parquet_rows
    jsonl = source if source.is_file() else source / "train.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"no Agent parquet shards or train.jsonl found under {source}")
    selected: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(jsonl):
        action = _target_action(row["messages"])
        if action is None:
            continue
        needs_action = action not in selected
        needs_recovery = (
            "recovery" not in selected
            and any(message.get("loss") is False for message in row["messages"][:-1])
        )
        if not needs_action and not needs_recovery:
            continue
        try:
            validate_target_weights(row)
        except ValueError:
            continue
        if needs_action:
            selected[action] = row
        if needs_recovery:
            selected["recovery"] = row
        if set(selected) == {"tool_call", "answer", "recovery"}:
            break
    return selected


def _decode_supervised(template: Any, input_ids: list[int], labels: list[int]) -> str:
    supervised_ids = [
        token for token, label in zip(input_ids, labels, strict=True) if label != -100
    ]
    if not supervised_ids:
        raise AssertionError("encoded row has no supervised tokens")
    return template.safe_decode(supervised_ids)


def encode_and_validate(
    template: Any,
    row: dict[str, Any],
    *,
    require_observation: bool = True,
) -> dict[str, Any]:
    action, target_weight = validate_target_weights(row)
    encoded = template.encode(row)
    input_ids = list(encoded["input_ids"])
    labels = list(encoded["labels"])
    token_weights = [float(value) for value in encoded.get("loss_scale") or []]
    if not token_weights or len(token_weights) != len(input_ids):
        raise AssertionError("is_binary_loss_scale=false did not produce token-level loss_scale")
    if len(labels) != len(input_ids):
        raise AssertionError("input_ids/labels length mismatch")

    supervised_positions = [index for index, label in enumerate(labels) if label != -100]
    masked_positions = [index for index, label in enumerate(labels) if label == -100]
    wrong_supervised = [
        (index, token_weights[index])
        for index in supervised_positions
        if not math.isclose(token_weights[index], target_weight, rel_tol=0, abs_tol=1e-6)
    ]
    if wrong_supervised:
        raise AssertionError(
            "supervised tokens do not all carry the exact trajectory weight: "
            f"expected={target_weight}, first_mismatches={wrong_supervised[:8]}"
        )
    wrong_masked = [
        (index, token_weights[index])
        for index in masked_positions
        if not math.isclose(token_weights[index], 0.0, rel_tol=0, abs_tol=1e-12)
    ]
    if wrong_masked:
        raise AssertionError(
            "masked tokens carry non-zero loss weight: "
            f"first_mismatches={wrong_masked[:8]}"
        )

    supervised_text = _decode_supervised(template, input_ids, labels)
    required_markers = ["<think>", "<|im_end|>"]
    if require_observation:
        required_markers.insert(0, "<observation>")
    required_markers.append("<tool_call>" if action == "tool_call" else "<answer>")
    missing_markers = [marker for marker in required_markers if marker not in supervised_text]
    if missing_markers:
        raise AssertionError(
            "supervised-only decode is missing target protocol markers: "
            f"action={action}, missing={missing_markers}, text={supervised_text!r}"
        )
    if "<tool_response>" in supervised_text or "</tool_response>" in supervised_text:
        raise AssertionError(f"historical tool response leaked into supervised text: {supervised_text!r}")

    supervised_tokens = len(supervised_positions)
    return {
        "action": action,
        "target_weight": target_weight,
        "input_tokens": len(input_ids),
        "supervised_tokens": supervised_tokens,
        "fractional_tokens": supervised_tokens,
        "masked_tokens": len(masked_positions),
        "supervised_text": supervised_text,
        "input_ids": input_ids,
        "labels": labels,
    }


def _decode_supervised_tail(template: Any, result: dict[str, Any], tail_tokens: int) -> dict[str, str]:
    input_ids = result["input_ids"]
    start = max(0, len(input_ids) - max(1, tail_tokens))
    return {
        "rendered_tail": template.safe_decode(input_ids[start:]),
        "supervised_tail": result["supervised_text"],
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    rows = load_representative_rows(source)
    contract_root = source if source.is_dir() else source.parent
    contract_path = contract_root / "dataset_contract.json"
    protocol = None
    if contract_path.is_file():
        protocol = json.loads(contract_path.read_text(encoding="utf-8")).get("protocol")
    require_observation = protocol != "native-tool-v3-seconds-think-only"
    missing = {"tool_call", "answer"} - set(rows)
    if args.require_recovery and "recovery" not in rows:
        missing.add("recovery")
    if missing:
        raise ValueError(f"could not find fractional representative rows for actions: {sorted(missing)}")

    from swift.model import get_processor
    from swift.template import get_template

    from omnicoach_sft.train.loss import install_exact_agent_suffix_loss_scale

    processor = get_processor(args.model, use_hf=True)
    template = get_template(
        processor,
        max_length=args.max_length,
        truncation_strategy="raise",
        agent_template="hermes",
        remove_unused_columns=False,
        loss_scale="last_round",
        is_binary_loss_scale=False,
        template_backend="swift",
    )
    install_exact_agent_suffix_loss_scale(template)
    template.set_mode("train")

    report = {
        "ok": True,
        "source": str(args.source.resolve()),
        "model": args.model,
        "template_backend": "swift",
        "agent_template": "hermes",
        "loss_scale": "last_round",
        "is_binary_loss_scale": False,
        "protocol": protocol,
        "require_observation": require_observation,
        "rows": [],
    }
    cases = ["tool_call", "answer"]
    if "recovery" in rows:
        cases.append("recovery")
    for case in cases:
        result = encode_and_validate(
            template,
            rows[case],
            require_observation=require_observation,
        )
        result["case"] = case
        decoded = _decode_supervised_tail(template, result, args.tail_tokens)
        report["rows"].append(
            {
                key: value
                for key, value in result.items()
                if key not in {"input_ids", "labels", "supervised_text"}
            }
        )
        print(f"\n=== {case} rendered tail ===\n{decoded['rendered_tail']}")
        print(f"\n=== {case} supervised-only decode ===\n{decoded['supervised_tail']}")

    rendered_report = json.dumps(report, ensure_ascii=False, indent=2)
    print(f"\n=== validation ===\n{rendered_report}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered_report + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"encode smoke failed: {exc}", file=sys.stderr)
        raise
