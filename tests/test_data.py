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


from rural_stroke_assist.preprocessing.splitting import (
    add_stratified_split,
    add_group_split,
)


def test_stratified_split_assigns_all_rows():
    df = pd.DataFrame(
        {
            "feature": range(20),
            "label": [0] * 10 + [1] * 10,
        }
    )

    split_df = add_stratified_split(df, label_column="label")

    assert "split" in split_df.columns
    assert split_df["split"].isna().sum() == 0
    assert set(split_df["split"].unique()) == {"train", "val", "test"}


def test_group_split_prevents_group_leakage():
    rows = []

    # 6 control speakers, 5 files each
    for speaker_id in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        for i in range(5):
            rows.append(
                {
                    "speaker_id": speaker_id,
                    "label": "control",
                    "file_id": f"{speaker_id}_{i}",
                }
            )

    # 6 dysarthric speakers, 5 files each
    for speaker_id in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        for i in range(5):
            rows.append(
                {
                    "speaker_id": speaker_id,
                    "label": "dysarthric",
                    "file_id": f"{speaker_id}_{i}",
                }
            )

    df = pd.DataFrame(rows)

    split_df = add_group_split(
        df,
        group_column="speaker_id",
        label_column="label",
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
    )

    group_split_counts = (
        split_df[["speaker_id", "split"]]
        .drop_duplicates()
        .groupby("speaker_id")["split"]
        .nunique()
    )

    assert group_split_counts.max() == 1
    assert set(split_df["split"].unique()) == {"train", "val", "test"}


from rural_stroke_assist.preprocessing.face_dataset import (
    compute_class_weights_from_manifest,
)


def test_compute_class_weights_from_manifest_returns_both_classes():
    df = pd.DataFrame(
        {
            "class_label": ["NonStroke", "NonStroke", "Stroke"],
        }
    )

    weights = compute_class_weights_from_manifest(df)

    assert set(weights.keys()) == {0, 1}
    assert weights[1] > weights[0]

from pathlib import Path

from rural_stroke_assist.preprocessing.fer2013 import (
    infer_fer2013_label_from_path,
    infer_fer2013_split_from_path,
    FER2013_LABEL_MAPPING,
)


def test_fer2013_label_inference():
    path = Path("data/raw/fer2013/train/happy/example.png")

    assert infer_fer2013_label_from_path(path) == "happy"
    assert FER2013_LABEL_MAPPING["happy"] == 3


def test_fer2013_split_inference():
    train_path = Path("data/raw/fer2013/train/angry/example.png")
    test_path = Path("data/raw/fer2013/test/angry/example.png")

    assert infer_fer2013_split_from_path(train_path) == "train"
    assert infer_fer2013_split_from_path(test_path) == "test"


import numpy as np
import pandas as pd

from rural_stroke_assist.features.speech_features import (
    extract_mfcc_summary_features,
    extract_basic_audio_features,
)
from rural_stroke_assist.modeling.speech_baselines import (
    get_speech_feature_columns,
)


def test_extract_mfcc_summary_features_returns_expected_keys():
    sample_rate = 16000
    signal = np.random.randn(sample_rate).astype(np.float32)

    features = extract_mfcc_summary_features(
        signal=signal,
        sample_rate=sample_rate,
        n_mfcc=13,
    )

    assert "mfcc_1_mean" in features
    assert "mfcc_13_std" in features
    assert len(features) == 26


def test_extract_basic_audio_features_returns_duration():
    sample_rate = 16000
    signal = np.random.randn(sample_rate).astype(np.float32)

    features = extract_basic_audio_features(
        signal=signal,
        sample_rate=sample_rate,
    )

    assert "duration_seconds" in features
    assert features["duration_seconds"] > 0


def test_get_speech_feature_columns_excludes_metadata():
    df = pd.DataFrame(
        {
            "path": ["a.wav"],
            "label": ["control"],
            "label_encoded": [0],
            "speaker_id": ["FC01"],
            "split": ["train"],
            "mfcc_1_mean": [0.1],
            "rms_mean": [0.2],
        }
    )

    feature_columns = get_speech_feature_columns(df)

    assert "mfcc_1_mean" in feature_columns
    assert "rms_mean" in feature_columns
    assert "label" not in feature_columns