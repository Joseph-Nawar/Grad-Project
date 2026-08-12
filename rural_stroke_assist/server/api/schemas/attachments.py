"""Attachment schemas."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from rural_stroke_assist.server.api.schemas.common import ApiSchema


class AttachmentResponse(ApiSchema):
    id: UUID
    case_id: UUID
    kind: str
    media_type: str
    size_bytes: int
    checksum_sha256: str
