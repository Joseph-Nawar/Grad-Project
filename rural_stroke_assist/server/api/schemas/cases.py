"""Case request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from rural_stroke_assist.server.api.schemas.common import ApiSchema, PageInfo
from rural_stroke_assist.server.api.schemas.assessments import AcuteSymptomsSchema, MetadataInputSchema


class AssessmentInputDraft(ApiSchema):
    session_id: str = Field(default="api-session", min_length=1, max_length=128)
    metadata: MetadataInputSchema | None = None
    acute_symptoms: AcuteSymptomsSchema | None = None


class CaseCreateRequest(ApiSchema):
    id: UUID
    patient_code: str | None = Field(default=None, max_length=128)
    facility: str = Field(min_length=1, max_length=128)
    assessment_input: AssessmentInputDraft = Field(default_factory=AssessmentInputDraft)


class CaseUpdateRequest(ApiSchema):
    patient_code: str | None = Field(default=None, max_length=128)
    assessment_input: AssessmentInputDraft | None = None


class SubmissionRequest(ApiSchema):
    confirmed: bool


class AttachmentSummary(ApiSchema):
    id: UUID
    kind: str
    media_type: str
    size_bytes: int
    checksum_sha256: str


class CaseResponse(ApiSchema):
    id: UUID
    facility: str
    patient_code: str | None
    status: str
    version: int
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime | None = None
    assessment_id: UUID | None = None
    assessment_result: dict[str, Any] | None = None
    assessment_input: AssessmentInputDraft = Field(default_factory=AssessmentInputDraft)
    attachments: list[AttachmentSummary] = Field(default_factory=list)


class CasePage(ApiSchema):
    items: list[CaseResponse]
    page: PageInfo
