"""The artifact envelopes (`ActorContractArtifact`, `ActorRunArtifact`) and their JSON encoding."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.episode import EpisodeResult


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorContractArtifact:
    route_id: str
    snapshot: Mapping[str, Any]
    public_protocol: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "actor_contract_v1"

    def __post_init__(self) -> None:
        if not self.route_id.strip():
            raise ValueError("actor contract route_id must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "actor_contract",
            "schema_version": self.schema_version,
            "route_id": self.route_id,
            "snapshot": json_safe(self.snapshot),
            "public_protocol": json_safe(self.public_protocol),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorRunArtifact:
    episode: EpisodeResult
    schema_version: str = "actor_run_v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "actor_run",
            "schema_version": self.schema_version,
            "route_id": self.episode.route_id,
            "episode": self.episode.as_dict(),
        }
