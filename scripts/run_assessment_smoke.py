"""Run one real artifact-backed assessment and print a safe JSON summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.inference.registry import load_baseline_registry


def main() -> None:
    registry = load_baseline_registry()
    face_manifest = pd.read_csv(registry.manifest_path("face_split"))
    speech_manifest = pd.read_csv(registry.manifest_path("speech_split"))
    metadata_manifest = pd.read_csv(registry.manifest_path("metadata_split"))
    metadata_row = metadata_manifest.loc[metadata_manifest["split"] == "test"].iloc[0]
    assessment_input = AssessmentInput(
        session_id="developer-smoke",
        face_image_path=Path(face_manifest.loc[face_manifest["split"] == "test", "path"].iloc[0]),
        speech_audio_path=Path(speech_manifest.loc[speech_manifest["split"] == "test", "path"].iloc[0]),
        metadata=MetadataInput(
            age=metadata_row["age"], hypertension=metadata_row["hypertension"],
            heart_disease=metadata_row["heart_disease"], avg_glucose_level=metadata_row["avg_glucose_level"],
            bmi=metadata_row["bmi"], gender=metadata_row["gender"],
            ever_married=metadata_row["ever_married"], work_type=metadata_row["work_type"],
            Residence_type=metadata_row["Residence_type"], smoking_status=metadata_row["smoking_status"],
        ),
        acute_symptoms=AcuteStrokeSymptoms(face_drooping=True, symptom_onset_minutes=60),
    )
    result = create_default_assessment_service().assess(assessment_input)
    print(json.dumps({
        "status": result.status.value,
        "fusion": None if result.fusion is None else {
            "evidence_score": result.fusion.evidence_score,
            "risk_band": result.fusion.risk_band,
            "contributions": dict(result.fusion.modality_contributions),
        },
        "modalities": {
            name: {
                "available": execution.evidence.available,
                "score": execution.evidence.score,
                "quality_status": execution.evidence.quality_status.value,
                "failure": None if execution.failure is None else execution.failure.error_type,
            }
            for name, execution in result.modality_executions.items()
        },
        "warning_count": len(result.warnings),
        "explanation_codes": [item.code for item in result.explanations],
        "timings_ns": {
            "total": result.timings.total_duration_ns,
            "fusion": result.timings.fusion_duration_ns,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
