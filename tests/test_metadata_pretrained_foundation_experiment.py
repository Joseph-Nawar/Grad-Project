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


def test_tabpfn_builder_requests_v2_and_canonical_categorical_positions(monkeypatch):
    captured = {}

    class FakeModelVersion:
        V2 = "v2"

    class FakeClassifier:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            captured["version"] = version
            captured["kwargs"] = kwargs
            return object()

    monkeypatch.setitem(sys.modules, "tabpfn", types.SimpleNamespace(TabPFNClassifier=FakeClassifier))
    monkeypatch.setitem(sys.modules, "tabpfn.constants", types.SimpleNamespace(ModelVersion=FakeModelVersion))

    experiment.build_tabpfn_v2("cpu")

    assert captured["version"] == "v2"
    assert captured["kwargs"]["categorical_features_indices"] == [5, 6, 7, 8, 9]
    assert captured["kwargs"]["ignore_pretraining_limits"] is True


def test_tabicl_builder_pins_v2_checkpoint(monkeypatch):
    captured = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "tabicl", types.SimpleNamespace(TabICLClassifier=FakeClassifier))

    experiment.build_tabicl_v2("cpu")

    assert captured["checkpoint_version"] == "tabicl-classifier-v2-20260212.ckpt"
    assert captured["random_state"] == experiment.SEED


def test_requirements_pin_exact_tabular_foundation_versions():
    requirements = Path("requirements-pretrained-metadata-experiment.txt").read_text(encoding="utf-8")

    assert "tabpfn==9.0.0" in requirements
    assert "tabicl==2.2.0" in requirements
    assert "scikit-learn==1.4.2" in requirements


def test_training_subset_fingerprint_binds_order_schema_dtypes_and_contract():
    frame = pd.DataFrame({
        "age": [40.0], "hypertension": [0], "heart_disease": [0],
        "avg_glucose_level": [90.0], "bmi": [25.0], "gender": ["Female"],
        "ever_married": ["No"], "work_type": ["Private"],
        "Residence_type": ["Rural"], "smoking_status": ["never smoked"],
        "stroke": [0], "split": ["train"],
    })

    first = experiment.training_subset_fingerprint(frame)
    second = experiment.training_subset_fingerprint(frame.copy())

    assert first == second
    assert first["sha256"]
    assert first["feature_columns"] == experiment.FEATURE_COLUMNS
    assert first["dtypes"]["age"] == "float64"
