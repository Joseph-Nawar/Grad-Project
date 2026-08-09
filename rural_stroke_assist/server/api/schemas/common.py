"""Shared public API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
