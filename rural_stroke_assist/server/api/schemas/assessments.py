"""Assessment request and response schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator

from rural_stroke_assist.server.api.schemas.common import ApiSchema, validate_bounded_json

APPROVED_ASSESSMENT_SCHEMA_VERSION = 1
APPROVED_MODEL_IDS = {
    "face": "canonical-face-v1",
    "speech": "canonical-speech-v1",
    "metadata_context": "canonical-metadata-v1",
    "acute_symptoms": "canonical-acute-v1",
}
APPROVED_FUSION_ID = "canonical-fusion-v1"
APPROVED_EDGE_ARTIFACTS = {
    "face": {
        "artifact": "models/edge/face-trial-003-litert-fp32.tflite",
        "sha256": "12AAFECA62C02EC97741F89CEDDB1393684F175B32D15BFA8DD5B00F49A62F2E",
    },
    "speech": {
        "artifact": "models/edge/speech-trial-001-onnx.onnx",
        "sha256": "D08170F592DD1D45B7B86109077E1FC7EF14A58EC6EF51893CF04EE1B5CE7287",
    },
    "metadata_context": {
        "artifact": "models/experiments/metadata/mvp_metadata_risk_model.pkl",
        "sha256": "350545A6AB71A58373FD11BC2EF9F37BAADCC8420D66186EB6F4BD2B8AA1E8F5",
    },
}
APPROVED_ORIGINAL_ARTIFACTS = {
    "face": {
        "artifact": "models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras",
        "sha256": "C969C473CD369AEFE11815208407320D559AA7683998CA0EE28C53280C7011AC",
    },
    "speech": {
        "artifact": "models/experiments/speech/trial_001_mfcc_random_forest/model.pkl",
        "sha256": "1BDED0B3850EE71E2B775E7B8E3BF80F708EE2FAEE1F74CA64D8102BAE01C925",
    },
    "metadata_context": APPROVED_EDGE_ARTIFACTS["metadata_context"],
}


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
    work_type: (
        Literal["Govt_job", "Never_worked", "Private", "Self-employed", "children"] | None
    ) = None
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


class AssessmentImportRequest(ApiSchema):
    """Collector-generated, path-free assessment envelope."""

    envelope: dict[str, Any]
    assessment_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("envelope", mode="before")
    @classmethod
    def validate_bounded_envelope(cls, value: Any) -> Any:
        return validate_bounded_json(value)

    @model_validator(mode="after")
    def validate_trusted_envelope(self) -> "AssessmentImportRequest":
        from rural_stroke_assist.offline.contracts import AssessmentEnvelope

        parsed = AssessmentEnvelope.model_validate(self.envelope)
        if parsed.schema_version != APPROVED_ASSESSMENT_SCHEMA_VERSION:
            raise ValueError("assessment envelope schema version is not approved")
        provenance = parsed.provenance
        if provenance.get("schema_version") != APPROVED_ASSESSMENT_SCHEMA_VERSION:
            raise ValueError("assessment provenance schema version is not approved")
        if provenance.get("fusion_id") != APPROVED_FUSION_ID:
            raise ValueError("assessment provenance fusion identifier is not approved")
        model_ids = provenance.get("model_ids")
        if not isinstance(model_ids, dict) or any(
            APPROVED_MODEL_IDS.get(key) != value for key, value in model_ids.items()
        ):
            raise ValueError("assessment provenance model identifiers are not approved")
        runtime_artifacts = provenance.get("runtime_artifacts")
        if runtime_artifacts is not None:
            if provenance.get("profile") not in {"original", "optimized"} or not isinstance(
                runtime_artifacts, dict
            ):
                raise ValueError("assessment runtime provenance is not approved")
            for modality, artifact in runtime_artifacts.items():
                approved = (
                    APPROVED_EDGE_ARTIFACTS
                    if provenance.get("profile") == "optimized"
                    else APPROVED_ORIGINAL_ARTIFACTS
                ).get(modality)
                if approved is None or not isinstance(artifact, dict):
                    raise ValueError("assessment runtime artifact is not approved")
                if (
                    artifact.get("artifact") != approved["artifact"]
                    or artifact.get("artifact_sha256") != approved["sha256"]
                ):
                    raise ValueError("assessment runtime artifact hash is not approved")
        return self


class AssessmentResponse(ApiSchema):
    id: UUID
    case_id: UUID
    status: str
    request_snapshot: dict[str, Any]
    result_snapshot: dict[str, Any]
    created_at: str
    assessment_hash: str | None = None
