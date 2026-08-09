"""Assessment request and response schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field

from rural_stroke_assist.server.api.schemas.common import ApiSchema


class MetadataInputSchema(ApiSchema):
    """Public mirror of the canonical ten-field metadata adapter contract."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    age: float | None = Field(default=None, ge=0, le=120)
    hypertension: int | None = Field(default=None, ge=0, le=1)
    heart_disease: int | None = Field(default=None, ge=0, le=1)
    avg_glucose_level: float | None = Field(default=None, ge=0)
    bmi: float | None = Field(default=None, ge=0)
    gender: Literal["Female", "Male", "Other"] | None = None
    ever_married: Literal["No", "Yes"] | None = None
    work_type: Literal["Govt_job", "Never_worked", "Private", "Self-employed", "children"] | None = None
    residence_type: Literal["Rural", "Urban"] | None = Field(default=None, alias="Residence_type")
    smoking_status: Literal["Unknown", "formerly smoked", "never smoked", "smokes"] | None = None


class AcuteSymptomsSchema(ApiSchema):
    """Public mirror of the deterministic acute symptom/onset contract."""

    face_drooping: bool = False
    arm_weakness: bool = False
    speech_difficulty: bool = False
    balance_or_coordination_loss: bool = False
    vision_disturbance: bool = False
    sudden_severe_headache: bool = False
    confusion_or_understanding_difficulty: bool = False
    symptom_onset_minutes: int | None = Field(default=None, ge=0)
    symptoms_resolved: bool = False


class AssessmentCreateRequest(ApiSchema):
    id: UUID
    case_id: UUID
    face_attachment_id: UUID | None = None
    audio_attachment_id: UUID | None = None
    metadata: MetadataInputSchema | None = None
    acute_symptoms: AcuteSymptomsSchema | None = None
    session_id: str = Field(default="api-session", min_length=1, max_length=128)


class AssessmentResponse(ApiSchema):
    id: UUID
    case_id: UUID
    status: str
    request_snapshot: dict[str, Any]
    result_snapshot: dict[str, Any]
    created_at: str
