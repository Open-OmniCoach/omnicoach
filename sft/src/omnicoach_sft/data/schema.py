"""Validation rules and pack-length contracts for Agent SFT rows."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import wave
from collections import Counter
from itertools import pairwise
from typing import Any

from omnicoach_sft.media.image_tokens import (
    QWEN_IMAGE_MAX_TOKENS,
    QWEN_IMAGE_MIN_TOKENS,
    QWEN_IMAGE_PATCH_SIZE,
    QWEN_IMAGE_SPATIAL_MERGE_SIZE,
    QWEN_IMAGE_TOKEN_COUNT_BASIS,
    QWEN_IMAGE_WRAPPER_TOKENS,
    webp_dimensions as _webp_dimensions,
)

ALLOWED_ROLES = {"system", "user", "assistant", "tool_call", "tool_response"}
MEDIA_KEYS = {"image": "images", "audio": "audios", "video": "videos"}
FRAME_TIMESTAMP_RE = re.compile(r"<([0-9]+(?:\.[0-9]+)?) seconds>\s*<image>")
CONTROL_PREFIXES = ("[ERROR]", "[PARSER_STATE]", "[NOTICE]", "[NO_PROGRESS]")
PACK_IMAGE_MAX_PIXELS = 147456
PACK_IMAGE_MIN_PIXELS = 1024
PACK_IMAGE_RESIZE_FACTOR = 16
PACK_IMAGE_QUALITY = 90
PACK_IMAGE_WEBP_METHOD = 4
PACK_AUDIO_SAMPLE_RATE = 16000

_ALLOWED_NEXT_ROLES = {
    "system": {"user"},
    "user": {"assistant"},
    "assistant": {"tool_call"},
    "tool_call": {"tool_response"},
    "tool_response": {"assistant"},
}


def image_token_stats() -> dict[str, Any]:
    return {
        "image_token_count_basis": QWEN_IMAGE_TOKEN_COUNT_BASIS,
        "image_token_count_column": "omni_image_num_tokens",
        "image_patch_size": QWEN_IMAGE_PATCH_SIZE,
        "image_spatial_merge_size": QWEN_IMAGE_SPATIAL_MERGE_SIZE,
        "image_min_tokens": QWEN_IMAGE_MIN_TOKENS,
        "image_max_tokens": QWEN_IMAGE_MAX_TOKENS,
        "image_wrapper_tokens": QWEN_IMAGE_WRAPPER_TOKENS,
    }


def agent_pack_length_stats() -> dict[str, Any]:
    return {
        "image_max_pixels": PACK_IMAGE_MAX_PIXELS,
        **image_token_stats(),
        "audio_sample_rate": PACK_AUDIO_SAMPLE_RATE,
        "audio_sample_count_basis": "resampled_16khz",
    }


def agent_media_encoding() -> dict[str, Any]:
    return {
        "images": {
            "format": "webp",
            "quality": PACK_IMAGE_QUALITY,
            "method": PACK_IMAGE_WEBP_METHOD,
            "resize_factor": PACK_IMAGE_RESIZE_FACTOR,
            "min_pixels": PACK_IMAGE_MIN_PIXELS,
            "max_pixels": PACK_IMAGE_MAX_PIXELS,
        },
        "audios": {
            "format": "wav_pcm_s16le",
            "sample_rate": PACK_AUDIO_SAMPLE_RATE,
            "channels": 1,
        },
        "layout": "direct_per_turn_row",
        "media_retention": "rollout_snapshot_keep_recent_1",
    }


def _fail(context: str, message: str) -> None:
    raise ValueError(f"{context}: {message}")


def _is_control_message(message: dict[str, Any]) -> bool:
    return (
        message.get("role") in {"user", "tool_response"}
        and isinstance(message.get("content"), str)
        and message["content"].lstrip().startswith(CONTROL_PREFIXES)
    )


def _parse_tools(raw_tools: Any, *, context: str) -> set[str]:
    if not isinstance(raw_tools, str):
        _fail(context, "tools must be a JSON string")
    try:
        tools = json.loads(raw_tools)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: invalid tools JSON: {exc}") from exc
    if not isinstance(tools, list) or not tools:
        _fail(context, "tools JSON must be a non-empty list")
    names: set[str] = set()
    for tool_index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            _fail(context, f"tools[{tool_index}] must be an OpenAI function tool")
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            _fail(context, f"tools[{tool_index}].function.name must be a string")
        name = function["name"]
        if not name or name in names:
            _fail(context, f"duplicate or empty tool name: {name!r}")
        if not isinstance(function.get("parameters"), dict):
            _fail(context, f"tool {name!r} is missing a parameters object")
        names.add(name)
    return names


def _validate_embedded_image(value: Any, *, context: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes) or not value["bytes"]:
        _fail(context, "embedded image must contain non-empty bytes")
    payload = value["bytes"]
    digest = hashlib.sha256(payload).hexdigest()
    if value.get("path") != f"{digest}.webp":
        _fail(context, "embedded image path must be its SHA-256 digest with .webp suffix")
    width, height = _webp_dimensions(payload, context=context)
    if width % PACK_IMAGE_RESIZE_FACTOR or height % PACK_IMAGE_RESIZE_FACTOR:
        _fail(context, f"embedded image dimensions must be divisible by {PACK_IMAGE_RESIZE_FACTOR}")
    pixels = width * height
    if pixels < PACK_IMAGE_MIN_PIXELS or pixels > PACK_IMAGE_MAX_PIXELS:
        _fail(context, f"embedded image pixels out of range: {width}x{height}")


def _validate_embedded_audio(value: Any, *, context: str) -> None:
    if not isinstance(value, bytes) or not value:
        _fail(context, "embedded audio must be non-empty bytes")
    try:
        with wave.open(io.BytesIO(value), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frames = audio.getnframes()
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"{context}: embedded audio must be a PCM WAV: {exc}") from exc
    if (channels, sample_width, sample_rate) != (1, 2, PACK_AUDIO_SAMPLE_RATE):
        _fail(
            context,
            "embedded audio must be mono 16-bit PCM WAV at "
            f"{PACK_AUDIO_SAMPLE_RATE} Hz, got channels={channels} width={sample_width} rate={sample_rate}",
        )
    if frames <= 0:
        _fail(context, "embedded audio has no samples")


def _validate_media_value(value: Any, *, kind: str, embedded_media: bool, context: str) -> None:
    if kind == "image" and embedded_media:
        _validate_embedded_image(value, context=context)
        return
    if kind == "audio" and embedded_media:
        _validate_embedded_audio(value, context=context)
        return
    if not isinstance(value, str) or not value:
        _fail(context, f"{kind} media must be a non-empty path string")


def _validate_tool_call_message(content: str, *, tool_names: set[str], context: str) -> None:
    try:
        call = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: invalid tool_call JSON: {exc}") from exc
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        _fail(context, "tool_call must contain exactly name and arguments")
    if call["name"] not in tool_names:
        _fail(context, f"unknown tool name: {call['name']!r}")
    if not isinstance(call["arguments"], dict):
        _fail(context, "tool_call arguments must be an object")


def _validate_messages(
    messages: list[Any],
    *,
    tool_names: set[str],
    context: str,
) -> tuple[list[str], int, Counter[str], list[int]]:
    roles: list[str] = []
    tool_calls = 0
    placeholder_counts: Counter[str] = Counter(dict.fromkeys(MEDIA_KEYS, 0))
    media_message_indices: list[int] = []
    for message_index, message in enumerate(messages):
        message_context = f"{context}.messages[{message_index}]"
        if not isinstance(message, dict):
            _fail(message_context, "message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            _fail(message_context, f"unsupported role: {role!r}")
        if not isinstance(content, str):
            _fail(message_context, "content must be a string")
        if message.get("loss") is not None and not isinstance(message.get("loss"), bool):
            _fail(message_context, "loss must be boolean when present")
        roles.append(role)

        message_has_media = False
        for kind in MEDIA_KEYS:
            count = content.count(f"<{kind}>")
            placeholder_counts[kind] += count
            message_has_media = message_has_media or count > 0
        if message_has_media:
            media_message_indices.append(message_index)

        if role == "system" and message_index != 0:
            _fail(message_context, "system is only allowed as the first message")
        if role == "tool_call":
            tool_calls += 1
            if message_index == 0 or roles[-2] != "assistant":
                _fail(message_context, "tool_call must immediately follow assistant")
            _validate_tool_call_message(content, tool_names=tool_names, context=message_context)
        elif role == "tool_response" and (message_index == 0 or roles[-2] != "tool_call"):
            _fail(message_context, "tool_response must immediately follow tool_call")
    return roles, tool_calls, placeholder_counts, media_message_indices


def _validate_role_transitions(messages: list[dict[str, Any]], roles: list[str], *, context: str) -> None:
    start_roles = roles[:2] if roles[0] == "system" else roles[:1]
    if start_roles not in (["system", "user"], ["user"]):
        _fail(context, f"conversation must start with system,user or user; got {start_roles}")
    for message_index, (left, right) in enumerate(pairwise(roles)):
        masked_invalid_feedback = (
            left == "assistant"
            and right == "user"
            and messages[message_index].get("loss") is False
            and messages[message_index + 1]["content"].lstrip().startswith(("[ERROR]", "[NO_PROGRESS]"))
        )
        if right not in _ALLOWED_NEXT_ROLES[left] and not masked_invalid_feedback:
            _fail(context, f"invalid role transition at {message_index}: {left} -> {right}")
    if roles[-1] not in {"assistant", "tool_call"}:
        _fail(context, f"terminal role must be assistant or tool_call, got {roles[-1]!r}")


def _validate_media_window(
    messages: list[dict[str, Any]],
    roles: list[str],
    media_message_indices: list[int],
    *,
    context: str,
) -> None:
    if len(media_message_indices) > 1:
        _fail(context, f"keep_recent=1 permits at most one media-bearing message, got {media_message_indices}")
    if not media_message_indices:
        return
    media_message_index = media_message_indices[0]
    media_role = roles[media_message_index]
    if media_role not in {"user", "tool_response"}:
        _fail(context, f"media-bearing message must be init user or latest tool_response, got {media_role!r}")
    later_inputs = [
        index
        for index in range(media_message_index + 1, len(messages))
        if roles[index] in {"user", "tool_response"} and not _is_control_message(messages[index])
    ]
    if later_inputs:
        _fail(context, f"media-bearing message is not the latest environment input: {later_inputs}")


def _validate_target_loss_scale(
    messages: list[dict[str, Any]],
    roles: list[str],
    *,
    require_target_loss_scale: bool,
    context: str,
) -> tuple[set[int], float | None]:
    target_indices = {len(messages) - 1}
    if roles[-1] == "tool_call":
        target_indices.add(len(messages) - 2)
    ordered_targets = sorted(target_indices)
    target_loss_scales = [messages[index].get("loss_scale") for index in ordered_targets]
    if require_target_loss_scale:
        for target_index, target_loss_scale in zip(ordered_targets, target_loss_scales, strict=True):
            if isinstance(target_loss_scale, bool) or not isinstance(target_loss_scale, (int, float)):
                _fail(f"{context}.messages[{target_index}]", "target loss_scale must be numeric")
            if not math.isfinite(float(target_loss_scale)) or float(target_loss_scale) <= 0:
                _fail(f"{context}.messages[{target_index}]", "target loss_scale must be finite and > 0")
        if len({float(value) for value in target_loss_scales}) != 1:
            _fail(context, "assistant and terminal tool_call must have identical loss_scale")
    for message_index, message in enumerate(messages):
        if message_index not in target_indices and message.get("loss_scale") is not None:
            _fail(f"{context}.messages[{message_index}]", "historical messages must not set loss_scale")
        if message_index in target_indices and message.get("loss") is False:
            _fail(f"{context}.messages[{message_index}]", "current target must not be loss-masked")
    raw_scale = target_loss_scales[-1]
    return target_indices, float(raw_scale) if raw_scale is not None else None


def _validate_media_payloads(
    row: dict[str, Any],
    placeholder_counts: Counter[str],
    *,
    embedded_media: bool,
    context: str,
) -> None:
    for kind, column in MEDIA_KEYS.items():
        values = row.get(column) or []
        if not isinstance(values, list):
            _fail(context, f"{column} must be a list")
        if placeholder_counts[kind] != len(values):
            _fail(
                context,
                f"{kind} placeholder/media mismatch: placeholders={placeholder_counts[kind]} media={len(values)}",
            )
        for value in values:
            _validate_media_value(value, kind=kind, embedded_media=embedded_media, context=context)
    if placeholder_counts["video"]:
        _fail(context, "Agent rows must contain frame/audio evidence, not video payloads")


def _validate_tool_media_alignment(
    messages: list[dict[str, Any]],
    roles: list[str],
    media_message_indices: list[int],
    placeholder_counts: Counter[str],
    *,
    context: str,
) -> None:
    if not media_message_indices or roles[media_message_indices[0]] != "tool_response":
        return
    response_index = media_message_indices[0]
    tool_name = json.loads(messages[response_index - 1]["content"])["name"]
    image_count = placeholder_counts["image"]
    audio_count = placeholder_counts["audio"]
    if tool_name == "get_frames" and (image_count < 1 or audio_count != 0):
        _fail(context, "get_frames response must contain images only")
    if tool_name == "get_audio" and (image_count != 0 or audio_count != 1):
        _fail(context, "get_audio response must contain exactly one audio payload")
    if tool_name == "get_clip" and (image_count < 1 or audio_count not in {0, 1}):
        _fail(context, "get_clip response must contain images and at most one same-span audio payload")
    if tool_name == "get_multi_span" and image_count + audio_count < 1:
        _fail(context, "get_multi_span response must contain at least one image or audio payload")


def _validate_timestamps(
    messages: list[dict[str, Any]],
    placeholder_counts: Counter[str],
    *,
    context: str,
) -> None:
    timestamp_count = sum(len(FRAME_TIMESTAMP_RE.findall(message["content"])) for message in messages)
    if timestamp_count != placeholder_counts["image"]:
        _fail(
            context,
            f"timestamped image mismatch: timestamped={timestamp_count} images={placeholder_counts['image']}",
        )


def validate_agent_row(
    row: dict[str, Any],
    *,
    context: str,
    embedded_media: bool,
    require_target_loss_scale: bool = True,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        _fail(context, "row must be an object")
    tool_names = _parse_tools(row.get("tools"), context=context)
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        _fail(context, "messages must be a non-empty list")

    roles, tool_calls, placeholder_counts, media_message_indices = _validate_messages(
        messages,
        tool_names=tool_names,
        context=context,
    )
    _validate_role_transitions(messages, roles, context=context)
    _validate_media_window(messages, roles, media_message_indices, context=context)
    target_indices, target_loss_scale = _validate_target_loss_scale(
        messages,
        roles,
        require_target_loss_scale=require_target_loss_scale,
        context=context,
    )
    _validate_media_payloads(row, placeholder_counts, embedded_media=embedded_media, context=context)
    _validate_tool_media_alignment(messages, roles, media_message_indices, placeholder_counts, context=context)
    _validate_timestamps(messages, placeholder_counts, context=context)

    if row.get("channel") is not None and row["channel"] != "agent":
        _fail(context, f"channel must be 'agent', got {row['channel']!r}")
    return {
        "terminal_role": roles[-1],
        "tool_calls": tool_calls,
        "target_loss_scale": target_loss_scale,
        "masked_invalid_history_messages": sum(
            int(message.get("loss") is False)
            for index, message in enumerate(messages)
            if index not in target_indices and message.get("role") in {"assistant", "tool_call"}
        ),
        "media_counts": {kind: placeholder_counts[kind] for kind in MEDIA_KEYS},
    }
