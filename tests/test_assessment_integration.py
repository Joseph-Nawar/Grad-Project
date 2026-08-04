from pathlib import Path

import pandas as pd
import pytest

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.inference.registry import BaselineRegistry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = BaselineRegistry.from_file(ROOT / "config" / "baseline_registry.json")


@pytest.mark.integration
def test_real_end_to_end_assessment_uses_canonical_artifacts() -> None:
    face = pd.read_csv(REGISTRY.manifest_path("face_split"))
    speech = pd.read_csv(REGISTRY.manifest_path("speech_split"))
    metadata = pd.read_csv(REGISTRY.manifest_path("metadata_split"))
    row = metadata.loc[metadata["split"] == "test"].iloc[0]
    result = create_default_assessment_service().assess(AssessmentInput(
        session_id="integration",
        face_image_path=Path(face.loc[face["split"] == "test", "path"].iloc[0]),
        speech_audio_path=Path(speech.loc[speech["split"] == "test", "path"].iloc[0]),
        metadata=MetadataInput(
            age=row["age"], hypertension=row["hypertension"], heart_disease=row["heart_disease"],
            avg_glucose_level=row["avg_glucose_level"], bmi=row["bmi"], gender=row["gender"],
            ever_married=row["ever_married"], work_type=row["work_type"],
            Residence_type=row["Residence_type"], smoking_status=row["smoking_status"],
        ),
        acute_symptoms=AcuteStrokeSymptoms(face_drooping=True, arm_weakness=True),
    ))
    assert result.status.value == "complete"
    assert result.fusion is not None
    assert result.fusion.risk_band == "URGENT"
    assert all(execution.evidence.available for execution in result.modality_executions.values())
