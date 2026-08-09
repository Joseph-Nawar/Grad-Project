"""Assessment application boundary around the existing AssessmentService."""

from __future__ import annotations

from pathlib import Path

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.metadata_adapter import MetadataInput


def build_service_input(*, session_id: str, face_path: Path | None, audio_path: Path | None, metadata: dict | None, acute_symptoms: dict | None) -> AssessmentInput:
    return AssessmentInput(session_id=session_id, face_image_path=face_path, speech_audio_path=audio_path, metadata=MetadataInput.model_validate(metadata) if metadata is not None else None, acute_symptoms=AcuteStrokeSymptoms.model_validate(acute_symptoms) if acute_symptoms is not None else None)
