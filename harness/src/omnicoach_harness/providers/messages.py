"""Provider message rendering (`to_provider_messages`) and data-URL byte estimation."""

from __future__ import annotations

import base64
import json
import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..domain.message import MediaRef, Message

VALID_AUDIO_PART_TYPES = ("audio_url", "input_audio", "image_url")


def estimated_data_url_bytes(path: str | Path, *, mime_type: str | None = None) -> int:
    source = Path(path)
    size = source.stat().st_size
    mime = mime_type or mimetypes.guess_type(str(source))[0] or "application/octet-stream"
    return 4 * ((size + 2) // 3) + len(f"data:{mime};base64,") + 256


def data_url(media: MediaRef) -> str:
    source = media.source()
    if source.startswith(("data:", "http://", "https://")):
        return source
    path = Path(source)
    mime = media.mime_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _input_audio_payload(
    media: MediaRef,
    media_url_resolver: Callable[[MediaRef], str],
) -> dict[str, str]:
    url = media_url_resolver(media)
    mime = media.mime_type or ""
    if url.startswith("data:"):
        header, _, encoded = url.partition(",")
        if not header or not encoded:
            raise ValueError("malformed data URL for input_audio")
        if header.startswith("data:"):
            mime = header[5:].split(";", 1)[0] or mime
    elif url.startswith(("http://", "https://")):
        raise ValueError("audio_part_type='input_audio' requires base64/local audio, not an HTTP URL")
    else:
        raise ValueError(f"unsupported audio URL scheme for input_audio: {url[:32]!r}")

    suffix = Path(media.local_path).suffix.lstrip(".").lower() if media.local_path else ""
    audio_format = suffix or (mime.split("/", 1)[-1].split(";", 1)[0].lower() if "/" in mime else "m4a")
    return {"data": encoded, "format": audio_format}


def _render_media_part(
    part_type: str,
    media: MediaRef,
    *,
    media_url_resolver: Callable[[MediaRef], str],
    audio_part_type: str,
) -> dict[str, Any]:
    if part_type == "audio":
        if audio_part_type == "audio_url":
            return {"type": "audio_url", "audio_url": {"url": media_url_resolver(media)}}
        if audio_part_type == "input_audio":
            return {"type": "input_audio", "input_audio": _input_audio_payload(media, media_url_resolver)}
        if audio_part_type == "image_url":
            return {"type": "image_url", "image_url": {"url": media_url_resolver(media)}}
    payload: dict[str, Any] = {"url": media_url_resolver(media)}
    if part_type == "image":
        payload["detail"] = "auto"
    return {"type": "image_url", "image_url": payload}


def to_provider_messages(
    messages: tuple[Message, ...],
    *,
    media_url_resolver: Callable[[MediaRef], str] = data_url,
    audio_part_type: str = "audio_url",
) -> list[dict[str, Any]]:
    if audio_part_type not in VALID_AUDIO_PART_TYPES:
        raise ValueError(
            f"unsupported audio_part_type {audio_part_type!r}; expected one of {VALID_AUDIO_PART_TYPES}"
        )
    rendered: list[dict[str, Any]] = []
    for item in messages:
        legacy_tool_result = item.role == "tool" and item.tool_call_id is None
        if item.role == "system":
            rendered.append(
                {
                    "role": "system",
                    "content": "".join(part.text or "" for part in item.parts if part.type == "text"),
                }
            )
            continue
        parts: list[dict[str, Any]] = []
        if legacy_tool_result:
            parts.append({"type": "text", "text": "TOOL_RESULT"})
        for part in item.parts:
            if part.type == "text":
                parts.append({"type": "text", "text": part.text or ""})
            elif part.media is not None:
                parts.append(
                    _render_media_part(
                        part.type,
                        part.media,
                        media_url_resolver=media_url_resolver,
                        audio_part_type=audio_part_type,
                    )
                )
        value: dict[str, Any] = {
            "role": "user" if legacy_tool_result else item.role,
            "content": parts or None,
        }
        if item.tool_calls:
            value["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments
                        if call.raw_arguments is not None
                        else json.dumps(dict(call.arguments), ensure_ascii=False, separators=(",", ":")),
                    },
                }
                for call in item.tool_calls
            ]
        if item.tool_call_id is not None:
            value["tool_call_id"] = item.tool_call_id
        rendered.append(value)
    return rendered
