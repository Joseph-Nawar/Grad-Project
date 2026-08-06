"""Immutable domain contracts for submitted screening cases."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CaseStatus(StrEnum):
    DRAFT = "DRAFT"
    ASSESSED = "ASSESSED"
    SUBMITTED = "SUBMITTED"
    IN_REVIEW = "IN_REVIEW"
    REVIEWED_AGREED = "REVIEWED_AGREED"
    REVIEWED_OVERRIDDEN = "REVIEWED_OVERRIDDEN"


class AttachmentKind(StrEnum):
    FACE = "face"
    AUDIO = "audio"


@dataclass(frozen=True)
class AttachmentReference:
    kind: AttachmentKind
    relative_path: str
    media_type: str
    size_bytes: int


@dataclass(frozen=True)
class ClinicianReview:
    clinician_name: str
    medical_centre: str
    decision: str
    notes: str
    alternative_disposition: str | None
    reviewed_at: str


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    at: str
    actor: str
    detail: str = ""


@dataclass(frozen=True)
class Case:
    case_id: str
    patient_code: str | None
    collector_identity: str
    facility: str
    assessment_input: Mapping[str, object]
    attachments: tuple[AttachmentReference, ...]
    assessment_result: Mapping[str, object] | None
    created_at: str
    updated_at: str
    submitted_at: str | None
    status: CaseStatus
    clinician_review: ClinicianReview | None = None
    audit_events: tuple[AuditEvent, ...] = field(default_factory=tuple)

    def with_update(self, **changes: object) -> "Case":
        return replace(self, updated_at=utc_now(), **changes)
