from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms


class PatientMetadata(BaseModel):
    """Structured metadata collected during a stroke triage session."""

    age: int = Field(..., ge=0, le=120)
    sex: str
    symptom_onset_minutes: Optional[int] = Field(default=None, ge=0)
    hypertension: bool = False
    diabetes: bool = False
    previous_stroke: bool = False
    notes: Optional[str] = None


class AssessmentInput(BaseModel):
    """All raw inputs collected for one assessment session."""

    session_id: str
    face_video_path: Optional[Path] = None
    speech_audio_path: Optional[Path] = None
    face_image_path: Optional[Path] = None
    metadata: PatientMetadata
    acute_symptoms: Optional[AcuteStrokeSymptoms] = None
