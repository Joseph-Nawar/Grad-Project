from pydantic import ValidationError
import pytest
import pandas as pd

from rural_stroke_assist.capture.input_handler import create_assessment_input
from rural_stroke_assist.capture.schemas import PatientMetadata
from rural_stroke_assist.fusion.fusion_engine import (
    determine_triage_level,
    fuse_results,
)
from rural_stroke_assist.modules.face_module import FaceModuleResult
from rural_stroke_assist.modules.metadata_module import analyze_metadata
from rural_stroke_assist.modules.speech_module import SpeechModuleResult
from rural_stroke_assist.reporting.report_generator import generate_text_report


def test_patient_metadata_validates_age_range() -> None:
    with pytest.raises(ValidationError):
        PatientMetadata(age=-1, sex="female")


def test_create_assessment_input_returns_session_and_metadata() -> None:
    metadata = PatientMetadata(age=68, sex="male", hypertension=True)

    assessment = create_assessment_input(metadata=metadata)

    assert assessment.session_id
    assert assessment.metadata == metadata
    assert assessment.face_image_path is None
    assert assessment.speech_audio_path is None


def test_metadata_module_returns_score_in_valid_range() -> None:
    metadata = PatientMetadata(
        age=72,
        sex="female",
        hypertension=True,
        diabetes=True,
        previous_stroke=True,
        symptom_onset_minutes=90,
    )

    result = analyze_metadata(metadata)

    assert 0.0 <= result.contextual_risk_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert "Age is above 60." in result.evidence


@pytest.mark.parametrize(
    ("score", "expected_level"),
    [
        (0.2, "Low concern"),
        (0.5, "Moderate concern"),
        (0.8, "High concern"),
    ],
)
def test_determine_triage_level_thresholds(score: float, expected_level: str) -> None:
    assert determine_triage_level(score) == expected_level


def test_fusion_and_report_generation_produce_valid_output() -> None:
    face_result = FaceModuleResult(
        facial_asymmetry_score=0.8,
        confidence=0.7,
        evidence=["Facial asymmetry detected."],
        warnings=[],
    )
    speech_result = SpeechModuleResult(
        speech_abnormality_score=0.5,
        confidence=0.6,
        evidence=["Speech irregularity detected."],
        warnings=["Audio sample was short."],
    )
    metadata_result = analyze_metadata(
        PatientMetadata(age=67, sex="male", hypertension=True, symptom_onset_minutes=120)
    )

    fusion_result = fuse_results(face_result, speech_result, metadata_result)
    report = generate_text_report(fusion_result)

    assert 0.0 <= fusion_result.final_risk_score <= 1.0
    assert 0.0 <= fusion_result.confidence <= 1.0
    assert fusion_result.triage_level in {"Low concern", "Moderate concern", "High concern"}
    assert "does not diagnose stroke" in report
    assert "Important Safety Note" in report

from rural_stroke_assist.preprocessing.face_audit import (
    create_duplicate_report,
    create_clean_face_manifest,
)


def test_face_duplicate_report_detects_same_class_duplicates():
    manifest = pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg", "c.jpg"],
            "class_label": ["Stroke", "Stroke", "NonStroke"],
            "file_hash": ["hash1", "hash1", "hash2"],
        }
    )

    report = create_duplicate_report(manifest)

    duplicate_row = report[report["file_hash"] == "hash1"].iloc[0]

    assert duplicate_row["file_count"] == 2
    assert duplicate_row["class_count"] == 1
    assert duplicate_row["is_duplicate"] is True or duplicate_row["is_duplicate"] == True
    assert duplicate_row["is_cross_class_duplicate"] is False or duplicate_row["is_cross_class_duplicate"] == False


def test_clean_face_manifest_removes_cross_class_duplicates():
    manifest = pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg", "c.jpg"],
            "class_label": ["Stroke", "NonStroke", "Stroke"],
            "file_hash": ["hash1", "hash1", "hash2"],
        }
    )

    clean_manifest = create_clean_face_manifest(manifest)

    assert len(clean_manifest) == 1
    assert clean_manifest.iloc[0]["file_hash"] == "hash2"