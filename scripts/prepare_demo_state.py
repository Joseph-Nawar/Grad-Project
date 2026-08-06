"""Seed three non-identifiable public-data demonstration cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.cases.factory import create_default_workflow_service


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", default="runtime_data")
    args = parser.parse_args()
    root = Path(args.runtime_dir)
    service = create_default_workflow_service(root)
    existing = {case.patient_code: case for case in service.list_cases()}
    face_row = (
        pd.read_csv("data/processed/face_split_manifest.csv").query("split == 'test'").iloc[0]
    )
    speech_row = (
        pd.read_csv("data/processed/speech_split_manifest.csv").query("split == 'test'").iloc[0]
    )
    metadata = (
        pd.read_csv("data/processed/metadata_split_manifest.csv").query("split == 'test'").iloc[0]
    )
    metadata_input = {
        "age": None if pd.isna(metadata.age) else metadata.age,
        "hypertension": metadata.hypertension,
        "heart_disease": metadata.heart_disease,
        "avg_glucose_level": metadata.avg_glucose_level,
        "bmi": None if pd.isna(metadata.bmi) else metadata.bmi,
        "gender": metadata.gender,
        "ever_married": metadata.ever_married,
        "work_type": metadata.work_type,
        "residence_type": metadata.Residence_type,
        "smoking_status": metadata.smoking_status,
    }
    for code, agree, alternative in (
        ("DEMO-SUBMITTED", None, None),
        ("DEMO-AGREED", True, None),
        ("DEMO-OVERRIDDEN", False, "Urgent specialist review"),
    ):
        if code in existing:
            print(f"Already present: {code}")
            continue
        assessment_input = AssessmentInput(
            session_id=code,
            face_image_path=Path(face_row.path),
            speech_audio_path=Path(speech_row.path),
            metadata=metadata_input,
            acute_symptoms=AcuteStrokeSymptoms(
                face_drooping=True, speech_difficulty=agree is False, symptom_onset_minutes=30
            ),
        )
        case = service.create_draft(
            collector_identity="Demo collector",
            facility="Demonstration facility",
            patient_code=code,
            assessment_input=assessment_input,
            face_bytes=Path(face_row.path).read_bytes(),
            audio_bytes=Path(speech_row.path).read_bytes(),
            face_filename="public-demo-face.jpg",
            audio_filename="public-demo-speech.wav",
            face_media_type="image/jpeg",
            audio_media_type="audio/wav",
        )
        case = service.assess(case.case_id, actor="Demo collector")
        case = service.submit(case.case_id, actor="Demo collector", confirmed=True)
        if agree is not None:
            case = service.begin_review(case.case_id, actor="Demo clinician")
            case = service.review(
                case.case_id,
                clinician_name="Demo clinician",
                medical_centre="Demonstration centre",
                agree=agree,
                notes="Demonstration review only.",
                alternative_disposition=alternative,
            )
        print(f"Seeded {code}: {case.case_id} ({case.status.value})")
    print("Only public manifest examples and pseudonymous demo codes were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
