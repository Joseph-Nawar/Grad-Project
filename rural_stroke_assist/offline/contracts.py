"""Typed contracts for the collector edge."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkflowState(StrEnum):
    DRAFT = "DRAFT"
    ASSESSED = "ASSESSED"
    QUEUED = "QUEUED"
    SYNCED = "SYNCED"
    CONFLICT = "CONFLICT"
    DEAD_LETTER = "DEAD_LETTER"


class SyncState(StrEnum):
    IDLE = "IDLE"
    PENDING = "PENDING"
    SYNCING = "SYNCING"
    RETRY_WAIT = "RETRY_WAIT"
    BLOCKED_AUTH = "BLOCKED_AUTH"
    CONFLICT = "CONFLICT"
    DEAD_LETTER = "DEAD_LETTER"
    SYNCED = "SYNCED"


class OutboxState(StrEnum):
    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    RETRY_WAIT = "RETRY_WAIT"
    BLOCKED_AUTH = "BLOCKED_AUTH"
    CONFLICT = "CONFLICT"
    DEAD_LETTER = "DEAD_LETTER"
    ACKED = "ACKED"


class OutboxEventType(StrEnum):
    CREATE_CASE = "CREATE_CASE"
    UPLOAD_ATTACHMENT = "UPLOAD_ATTACHMENT"
    IMPORT_ASSESSMENT = "IMPORT_ASSESSMENT"
    SUBMIT_CASE = "SUBMIT_CASE"


class AttachmentRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: str
    kind: str
    media_type: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("attachment_id", "kind", "media_type")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("attachment reference values must not be empty")
        return value


class AssessmentEnvelope(BaseModel):
    """Portable immutable assessment representation with logical media refs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    case_id: str
    assessment_id: str
    request: dict[str, Any]
    result: dict[str, Any]
    provenance: dict[str, Any]

    @field_validator("case_id", "assessment_id")
    @classmethod
    def resource_id_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("resource identifiers must not be empty")
        return value

    @classmethod
    def _validate_path_free(cls, value: Any) -> None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise ValueError("assessment envelope must be path-free and must not contain raw media")
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).lower()
                if normalized.endswith("path") or normalized.endswith("_path") or normalized in {"file", "filename"}:
                    raise ValueError("assessment envelope must be path-free and must not contain raw media")
                cls._validate_path_free(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                cls._validate_path_free(item)
        elif isinstance(value, str) and (value.startswith(("/", "\\\\")) or (len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"})):
            raise ValueError("assessment envelope must be path-free and must not contain raw media")

    @classmethod
    def validate_path_free(cls, value: Any) -> Any:
        cls._validate_path_free(value)
        return value

    def model_post_init(self, __context: Any) -> None:
        self._validate_path_free(self.model_dump(mode="json"))

