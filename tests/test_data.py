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


import pandas as pd
import pytest

from rural_stroke_assist.preprocessing.metadata import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_metadata_modeling_frame,
    validate_metadata_columns,
    build_metadata_preprocessor,
)

def test_prepare_metadata_modeling_frame_drops_id():
    df = pd.DataFrame(
        {
            "id": [1],
            "gender": ["Male"],
            "age": [70],
            "hypertension": [1],
            "heart_disease": [0],
            "ever_married": ["Yes"],
            "work_type": ["Private"],
            "Residence_type": ["Urban"],
            "avg_glucose_level": [120.0],
            "bmi": [28.0],
            "smoking_status": ["formerly smoked"],
            "stroke": [1],
            "split": ["train"],
        }
    )

    result = prepare_metadata_modeling_frame(df)

    assert "id" not in result.columns
    assert TARGET_COLUMN in result.columns
    assert "split" in result.columns


def test_validate_metadata_columns_raises_for_missing_column():
    df = pd.DataFrame({"age": [70], "stroke": [1]})

    with pytest.raises(ValueError):
        validate_metadata_columns(df)


def test_build_metadata_preprocessor_transforms_dataframe():
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "age": [70, 55],
            "hypertension": [1, 0],
            "heart_disease": [0, 1],
            "ever_married": ["Yes", "No"],
            "work_type": ["Private", "Self-employed"],
            "Residence_type": ["Urban", "Rural"],
            "avg_glucose_level": [120.0, 90.0],
            "bmi": [28.0, None],
            "smoking_status": ["formerly smoked", "never smoked"],
        }
    )

    preprocessor = build_metadata_preprocessor()
    transformed = preprocessor.fit_transform(df[FEATURE_COLUMNS])

    assert transformed.shape[0] == 2

from rural_stroke_assist.modeling.metadata_baselines import (
    build_logistic_regression_metadata_pipeline,
)

def test_build_logistic_regression_metadata_pipeline_has_expected_steps():
    pipeline = build_logistic_regression_metadata_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.modules.acute_symptom_module import (
    assess_acute_stroke_symptoms,
)


ACUTE_RISK_ORDER = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "URGENT": 3,
}


def test_acute_symptom_assessment_low_case():
    symptoms = AcuteStrokeSymptoms()

    result = assess_acute_stroke_symptoms(symptoms)

    assert 0.0 <= result.acute_symptom_score <= 1.0
    assert result.risk_band == "LOW"
    assert result.hard_escalation is False


def test_acute_symptom_assessment_urgent_fast_case():
    symptoms = AcuteStrokeSymptoms(
        face_drooping=True,
        arm_weakness=True,
        speech_difficulty=True,
        symptom_onset_minutes=60,
    )

    result = assess_acute_stroke_symptoms(symptoms)

    assert result.acute_symptom_score >= 0.85
    assert result.risk_band == "URGENT"
    assert result.hard_escalation is True


@pytest.mark.parametrize(
    "symptoms",
    [
        AcuteStrokeSymptoms(face_drooping=True),
        AcuteStrokeSymptoms(arm_weakness=True),
        AcuteStrokeSymptoms(speech_difficulty=True),
    ],
)
def test_single_fast_sign_is_at_least_moderate(symptoms: AcuteStrokeSymptoms):
    result = assess_acute_stroke_symptoms(symptoms)

    assert ACUTE_RISK_ORDER[result.risk_band] >= ACUTE_RISK_ORDER["MODERATE"]
    assert result.hard_escalation is False


def test_two_fast_signs_trigger_urgent_hard_escalation():
    symptoms = AcuteStrokeSymptoms(
        face_drooping=True,
        arm_weakness=True,
    )

    result = assess_acute_stroke_symptoms(symptoms)

    assert result.risk_band == "URGENT"
    assert result.hard_escalation is True


def test_balance_and_vision_are_at_least_moderate():
    symptoms = AcuteStrokeSymptoms(
        balance_or_coordination_loss=True,
        vision_disturbance=True,
    )

    result = assess_acute_stroke_symptoms(symptoms)

    assert ACUTE_RISK_ORDER[result.risk_band] >= ACUTE_RISK_ORDER["MODERATE"]
    assert result.hard_escalation is False


def test_core_fast_plus_extension_is_high_or_urgent():
    symptoms = AcuteStrokeSymptoms(
        speech_difficulty=True,
        confusion_or_understanding_difficulty=True,
    )

    result = assess_acute_stroke_symptoms(symptoms)

    assert ACUTE_RISK_ORDER[result.risk_band] >= ACUTE_RISK_ORDER["HIGH"]
    assert result.hard_escalation is False


def test_acute_symptom_assessment_onset_unknown_warning():
    symptoms = AcuteStrokeSymptoms(face_drooping=True)

    result = assess_acute_stroke_symptoms(symptoms)

    assert any("unknown" in warning.lower() for warning in result.warnings)


@pytest.mark.parametrize(
    "symptoms",
    [
        AcuteStrokeSymptoms(),
        AcuteStrokeSymptoms(face_drooping=True),
        AcuteStrokeSymptoms(balance_or_coordination_loss=True, vision_disturbance=True),
        AcuteStrokeSymptoms(
            face_drooping=True,
            arm_weakness=True,
            speech_difficulty=True,
            symptom_onset_minutes=30,
        ),
    ],
)
def test_acute_symptom_assessment_score_is_bounded(symptoms: AcuteStrokeSymptoms):
    result = assess_acute_stroke_symptoms(symptoms)

    assert 0.0 <= result.acute_symptom_score <= 1.0



from rural_stroke_assist.modules.fusion_module import (
    FusionInput,
    fuse_multimodal_scores,
)


FUSION_RISK_ORDER = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "URGENT": 3,
}


def test_fusion_low_concern_case():
    result = fuse_multimodal_scores(
        FusionInput(
            face_acute_score=0.1,
            speech_acute_score=0.1,
            acute_symptom_score=0.0,
            metadata_contextual_risk_score=0.2,
        )
    )

    assert 0.0 <= result.fused_score <= 1.0
    assert result.risk_band == "LOW"


def test_fusion_metadata_high_only_remains_low():
    result = fuse_multimodal_scores(
        FusionInput(
            face_acute_score=0.05,
            speech_acute_score=0.05,
            acute_symptom_score=0.05,
            metadata_contextual_risk_score=0.95,
        )
    )

    assert result.risk_band == "LOW"


def test_fusion_hard_escalation_forces_urgent():
    result = fuse_multimodal_scores(
        FusionInput(
            face_acute_score=0.2,
            speech_acute_score=0.2,
            acute_symptom_score=0.85,
            metadata_contextual_risk_score=0.1,
            acute_symptom_hard_escalation=True,
        )
    )

    assert result.fused_score >= 0.85
    assert result.risk_band == "URGENT"


def test_fusion_missing_modality_renormalizes_weights():
    result = fuse_multimodal_scores(
        FusionInput(
            face_acute_score=None,
            speech_acute_score=0.8,
            acute_symptom_score=0.4,
            metadata_contextual_risk_score=0.2,
        )
    )

    assert "face" not in result.normalized_weights_used
    assert abs(sum(result.normalized_weights_used.values()) - 1.0) < 1e-8
    assert 0.0 <= result.fused_score <= 1.0


def test_fusion_speech_very_high_plus_symptoms_moderate_is_at_least_moderate():
    result = fuse_multimodal_scores(
        FusionInput(
            face_acute_score=0.25,
            speech_acute_score=0.92,
            acute_symptom_score=0.35,
            metadata_contextual_risk_score=0.40,
        )
    )

    assert FUSION_RISK_ORDER[result.risk_band] >= FUSION_RISK_ORDER["MODERATE"]


def test_fusion_face_and_speech_high_reaches_high_band():
    result = fuse_multimodal_scores(
        FusionInput(
            face_acute_score=0.88,
            speech_acute_score=0.81,
            acute_symptom_score=0.35,
            metadata_contextual_risk_score=0.55,
        )
    )

    assert result.risk_band == "HIGH"


def test_fusion_rejects_invalid_score():
    import pytest

    with pytest.raises(ValueError):
        fuse_multimodal_scores(
            FusionInput(
                face_acute_score=1.5,
                speech_acute_score=0.2,
            )
        )
