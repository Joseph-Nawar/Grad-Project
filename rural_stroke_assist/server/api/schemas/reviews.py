"""Review schemas."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from rural_stroke_assist.server.api.schemas.common import ApiSchema


class ReviewClaimRequest(ApiSchema):
    id: UUID


class ReviewCreateRequest(ApiSchema):
    agree: bool
    notes: str = Field(min_length=1, max_length=4000)
    alternative_disposition: str | None = Field(default=None, max_length=512)


class ReviewResponse(ApiSchema):
    id: UUID
    case_id: UUID
    reviewer_subject: str
    agree: bool
    notes: str
    alternative_disposition: str | None
    created_at: str


class ReviewClaimResponse(ApiSchema):
    case_id: UUID
    status: str
    claimed_by: str
