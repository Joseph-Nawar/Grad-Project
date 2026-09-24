"""Shared public API schemas."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


def validate_bounded_json(
    value: Any,
    *,
    max_bytes: int = 256 * 1024,
    max_depth: int = 12,
    max_nodes: int = 10_000,
    max_string: int = 8_192,
) -> Any:
    """Bound security-sensitive arbitrary JSON before application processing."""

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("bounded JSON payload exceeds the configured limit")
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise ValueError("bounded JSON payload exceeds the configured limit")
        if isinstance(item, str) and len(item) > max_string:
            raise ValueError("bounded JSON payload exceeds the configured limit")
        if isinstance(item, dict):
            for key, child in item.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1)

    visit(value, 0)
    return value


class ApiSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class PageInfo(ApiSchema):
    next_cursor: str | None = None


class ErrorResponse(ApiSchema):
    code: str
    message: str
    details: dict[str, Any]
    correlation_id: str


class ResourceRef(ApiSchema):
    id: UUID


class Timestamped(ApiSchema):
    id: UUID
    created_at: datetime
    updated_at: datetime


class HealthResponse(ApiSchema):
    status: str


class VersionResponse(ApiSchema):
    version: str


class IdentityResponse(ApiSchema):
    subject: str
    roles: list[str]
    facilities: list[str]


class SnapshotResponse(ApiSchema):
    snapshot: dict[str, Any]
    snapshot_sha256: str
    submitted_at: datetime | None
