"""`retain_latest_media`, which keeps media on only the most recent message."""

from __future__ import annotations

from dataclasses import replace

from ..domain.message import Message, text_part


def _omit_media(value: Message, *, marker: str = "[MEDIA OMITTED]") -> Message:
    if not any(part.type != "text" for part in value.parts):
        return value
    kept = tuple(part for part in value.parts if part.type == "text")
    return replace(value, parts=(*kept, text_part(marker)))


def retain_latest_media(messages: list[Message], *, keep_recent: int) -> None:
    media_indices = [index for index, item in enumerate(messages) if any(part.type != "text" for part in item.parts)]
    for index in media_indices[: -max(0, keep_recent)] if keep_recent else media_indices:
        messages[index] = _omit_media(messages[index])

