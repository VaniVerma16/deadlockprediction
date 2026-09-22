from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["thread", "lock"]
    features: dict[str, int | float] = Field(default_factory=dict)


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    type: Literal["owned_by", "waits_for"]


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    ts_ns: int
    nodes: list[Node]
    edges: list[Edge]
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_snapshot(value: dict[str, Any]) -> Snapshot:
    """Validate a snapshot with Pydantic across Pydantic v1/v2."""

    validator = getattr(Snapshot, "model_validate", None)
    if validator is not None:
        return validator(value)
    return Snapshot.parse_obj(value)
