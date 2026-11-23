"""`OmniOTAContract` and shared schema primitives for the `omni_ota_v1` route."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

ROUTE_ID = "omni_ota_v1"

TIMECODE_PATTERN = r"^\d+:[0-5]\d\.\d$"
VALID_LEVELS = ("coarse", "detail")
VALID_RESOLUTIONS = ("default", "high")
VISUAL_MODALITIES = ("frames", "clip")
SPAN_MODALITIES = ("audio", "frames", "clip")


def timecode_schema() -> dict[str, Any]:
    return {"type": "string", "pattern": TIMECODE_PATTERN}


def level_schema() -> dict[str, Any]:
    return {"type": "string", "enum": list(VALID_LEVELS)}


def resolution_schema(*, include_default: bool = False) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "enum": list(VALID_RESOLUTIONS)}
    if include_default:
        schema["default"] = "default"
    return schema


def visual_properties(*, include_default: bool = False) -> dict[str, Any]:
    return {
        "start": timecode_schema(),
        "end": timecode_schema(),
        "level": level_schema(),
        "resolution": resolution_schema(include_default=include_default),
    }


def audio_span_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "modality": {"type": "string", "enum": ["audio"]},
            "start": timecode_schema(),
            "end": timecode_schema(),
        },
        "required": ["modality", "start", "end"],
        "additionalProperties": False,
    }


def visual_span_schema(
    modalities: Sequence[str],
    *,
    include_default: bool = False,
    modality_last: bool = False,
) -> dict[str, Any]:
    modality_prop = {"type": "string", "enum": list(modalities)}
    props = (
        {**visual_properties(include_default=include_default), "modality": modality_prop}
        if modality_last
        else {"modality": modality_prop, **visual_properties(include_default=include_default)}
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["modality", "start", "end", "level"],
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class OmniOTAContract:
    max_steps: int = 32
    max_frames: int = 256
    max_audio_seconds: float = 256.0
    max_clip_seconds: float = 60.0
    min_audio_seconds: float = 0.1
    framepack_fps: float = 0.5
    framepack_max_frames: int = 256
    framepack_max_pixels: int = 224 * 224
    framepack_request_bytes: int = 90_000_000
    default_max_side: int = 384
    high_max_side: int = 1280
    high_max_span_seconds: float = 5.0
    multi_max_spans: int = 8
    multi_max_total_frames: int = 256
    multi_max_total_audio_seconds: float = 128.0
    multi_max_high_resolution_spans: int = 2
    keep_recent_media: int = 1

    @property
    def response_format(self) -> dict[str, Any]:
        from .protocol import response_schema

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "observe_think_action",
                "strict": True,
                "schema": response_schema(),
            },
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "semantic_name": "omni_ota_v1",
            "treatment": {
                "kind": "temporal_interval_output_contract",
                "scope": "TR_only",
                "point_answer_policy": "recoverable_invalid_action",
                "semantic_scoring": "deferred_to_temporal_grounding_judge",
            },
            "temporal_answer": {
                "single": "[MM:SS.m,MM:SS.m]",
                "multiple": "[MM:SS.m,MM:SS.m]; [MM:SS.m,MM:SS.m]",
                "requires_positive_width": True,
                "requires_within_video": True,
                "fixed_width_point_expansion": False,
            },
            "timecode": "mmss_exactly_one_decimal",
            "observation_citation": "[MM:SS.m,MM:SS.m]",
            "action_shape": "canonical_flat_type_only",
            "direct_first_answer": True,
            "ordered_multi_span": True,
            "response_format": self.response_format,
            "tool_observation": {
                "provider_visible": "compact_header_then_timestamped_media",
                "persisted_info": "rich_diagnostics",
                "repeated_media_labels": False,
            },
            "visual_sampling": {
                "coarse_fps": 1.0,
                "detail_fps": 2.0,
                "timestamp_mode": "fixed_interval_from_start_endpoint_exclusive",
            },
            "high_resolution": {
                "enabled": True,
                "default_max_side": self.default_max_side,
                "high_max_side": self.high_max_side,
                "high_max_span_seconds": self.high_max_span_seconds,
            },
            "multi_span": {
                "max_spans": self.multi_max_spans,
                "max_total_frames": self.multi_max_total_frames,
                "max_total_audio_seconds": self.multi_max_total_audio_seconds,
                "max_high_resolution_spans": self.multi_max_high_resolution_spans,
            },
        }

    def implementation_snapshot(self) -> dict[str, Any]:
        return {
            "resources": {
                "max_steps": self.max_steps,
                "max_frames": self.max_frames,
                "max_audio_seconds": self.max_audio_seconds,
                "min_audio_seconds": self.min_audio_seconds,
                "max_clip_seconds": self.max_clip_seconds,
            },
            "framepack": {
                "fps": self.framepack_fps,
                "max_frames": self.framepack_max_frames,
                "max_pixels": self.framepack_max_pixels,
                "include_full_audio": True,
                "audio_required_when_source_has_audio": True,
            },
            "media_keep_recent": self.keep_recent_media,
            "tool_result_info": "persist_rich_provider_hidden_metadata",
        }


CONTRACT = OmniOTAContract()
