import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

import scripts.metadata_pretrained_foundation_experiment as experiment


def test_requested_split_loader_never_returns_unrequested_rows(tmp_path):
    manifest = tmp_path / "metadata.csv"
    manifest.write_text(
        "age,stroke,split\n40,0,train\n70,1,val\n80,1,test\n",
        encoding="utf-8",
    )

    train_val = experiment.load_requested_splits(manifest, ("train", "val"))

    assert set(train_val["split"]) == {"train", "val"}
    assert "test" not in set(train_val["split"])


def test_candidate_frame_preserves_contract_order_and_mixed_types():
    frame = pd.DataFrame({
        "age": [70.0], "hypertension": [1], "heart_disease": [0],
        "avg_glucose_level": [120.0], "bmi": [28.0], "gender": ["Male"],
        "ever_married": ["Yes"], "work_type": ["Private"],
        "Residence_type": ["Urban"], "smoking_status": ["never smoked"],
        "stroke": [1], "split": ["train"],
    })

    candidate = experiment.prepare_candidate_frame(frame)

    assert list(candidate.columns) == experiment.FEATURE_COLUMNS
    assert all(pd.api.types.is_numeric_dtype(candidate[name]) for name in experiment.NUMERIC_FEATURES)
    assert all(pd.api.types.is_object_dtype(candidate[name]) for name in experiment.CATEGORICAL_FEATURES)
    assert experiment.CATEGORICAL_FEATURE_INDICES == [5, 6, 7, 8, 9]


def test_metrics_match_project_implementation_and_add_balanced_accuracy():
    from rural_stroke_assist.evaluation.metrics import classification_metrics, metric_dict

    result = experiment.calculate_binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
    reference = metric_dict(classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9]))

    assert result["confusion_matrix"] == reference["confusion_matrix"]
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.75
    for name in ("roc_auc", "pr_auc", "brier", "calibration_error", "sensitivity", "specificity", "precision", "f1"):
        assert result[name] == pytest.approx(reference[name])


def test_stratified_bootstrap_is_deterministic_and_matches_project_bootstrap():
    first = experiment.stratified_bootstrap_metrics(
        [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], iterations=20, seed=42,
    )
    second = experiment.stratified_bootstrap_metrics(
        [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], iterations=20, seed=42,
    )

    from rural_stroke_assist.evaluation.bootstrap import stratified_bootstrap_ci
    from rural_stroke_assist.evaluation.metrics import classification_metrics
    reference = stratified_bootstrap_ci(
        [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9],
        metric=lambda y, p: classification_metrics(y, p).pr_auc,
        iterations=20, seed=42, metric_name="pr_auc",
    )

    assert first == second
    assert first["pr_auc"]["seed"] == 42
    assert first["pr_auc"]["iterations"] == 20
    assert first["pr_auc"]["lower"] == reference.lower
    assert first["pr_auc"]["upper"] == reference.upper


def test_resource_protocol_is_explicit_and_deterministic():
    protocol = experiment.build_resource_protocol_payload("cpu")

    assert protocol == {
        "device": "cpu",
        "validation_sample_rule": "first_128_validation_rows_in_manifest_order",
        "validation_sample_size": 128,
        "validation_sample_positions": list(range(128)),
        "warmup_runs": 2,
        "timed_repetitions": 5,
        "preprocessing_included": "candidate_native_inside_predict_proba_only",
        "model_load_and_context_fit_excluded": True,
        "p50_method": "numpy.median_of_five_wall_clock_milliseconds",
    }
