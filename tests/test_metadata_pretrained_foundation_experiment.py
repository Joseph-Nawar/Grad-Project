import json
from pathlib import Path
import subprocess
import sys
import types

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


def test_validation_requests_only_train_and_val(monkeypatch, tmp_path):
    requested = []

    def fake_loader(path, splits):
        requested.append(tuple(splits))
        return pd.DataFrame({
            "age": [40.0, 70.0], "hypertension": [0, 1], "heart_disease": [0, 0],
            "avg_glucose_level": [90.0, 180.0], "bmi": [25.0, 31.0],
            "gender": ["Female", "Male"], "ever_married": ["No", "Yes"],
            "work_type": ["Private", "Private"], "Residence_type": ["Rural", "Urban"],
            "smoking_status": ["never smoked", "formerly smoked"],
            "stroke": [0, 1], "split": ["train", "val"],
        })

    def fake_baseline_runner(frame, output_dir):
        return {"candidate_id": "logistic_regression", "status": "completed", "metrics": {"pr_auc": 0.2}}

    def fake_candidate_runner(frame, output_dir, candidate_id):
        return {"candidate_id": candidate_id, "status": "completed", "metrics": {"pr_auc": 0.1}}

    monkeypatch.setattr(experiment, "load_requested_splits", fake_loader)

    experiment.run_validation(
        tmp_path,
        Path("data/processed/metadata_split_manifest.csv"),
        candidate_ids=("tabpfn_v2", "tabicl_v2"),
        loader=fake_loader,
        baseline_runner=fake_baseline_runner,
        candidate_runner=fake_candidate_runner,
    )

    assert requested == [("train", "val")]


def test_selection_snapshot_binds_candidate_and_resource_hashes(tmp_path):
    write_completed_validation_fixture(tmp_path)

    selection = experiment.run_selection(tmp_path)

    assert selection["test_used_for_selection"] is False
    assert selection["candidate_id"] == "tabicl_v2"
    assert selection["protocol_sha256"]
    assert set(selection["validation_evidence"]) == {"tabpfn_v2", "tabicl_v2"}
    assert all(item["status"] == "completed" for item in selection["validation_evidence"].values())
    assert selection["resource_comparison_sha256"]


def test_material_pr_auc_gap_selects_best_candidate_before_resources(tmp_path):
    write_completed_validation_fixture(tmp_path)
    set_validation_pr_auc(tmp_path, "tabicl_v2", 0.10)

    selection = experiment.run_selection(tmp_path)

    assert selection["candidate_id"] == "tabpfn_v2"


def test_tampered_validation_evidence_blocks_test_before_test_rows(monkeypatch, tmp_path):
    write_completed_validation_fixture(tmp_path)
    experiment.run_selection(tmp_path)
    (tmp_path / "validation_tabpfn_v2.json").write_text("{}\n", encoding="utf-8")
    loaded = False

    def fail_if_test_loaded(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("test rows were loaded before hash verification")

    monkeypatch.setattr(experiment, "load_requested_splits", fail_if_test_loaded)

    with pytest.raises(ValueError, match="validation evidence hash mismatch"):
        experiment.run_test(tmp_path, Path("manifest.csv"))

    assert loaded is False


def test_one_failed_candidate_selects_completed_candidate_and_binds_failure(tmp_path):
    write_one_failed_validation_fixture(tmp_path)

    selection = experiment.run_selection(tmp_path)

    assert selection["candidate_id"] == "tabpfn_v2"
    assert selection["validation_evidence"]["tabicl_v2"]["status"] == "failed"
    assert selection["validation_evidence"]["tabicl_v2"]["failure_artifact_sha256"]


def test_zero_completed_candidates_prohibits_selection_and_test(tmp_path):
    write_zero_completed_validation_fixture(tmp_path)

    with pytest.raises(ValueError, match="zero completed pretrained candidates"):
        experiment.run_selection(tmp_path)
    assert not (tmp_path / "selection_frozen.json").exists()
    with pytest.raises(ValueError, match="selection is not frozen"):
        experiment.run_test(tmp_path, Path("manifest.csv"))


def test_selection_and_test_are_one_time_operations(tmp_path):
    write_completed_validation_fixture(tmp_path)
    experiment.run_selection(tmp_path)
    with pytest.raises(ValueError, match="selection is already frozen"):
        experiment.run_selection(tmp_path)
    mark_test_completed(tmp_path)
    with pytest.raises(ValueError, match="test stage has already run"):
        experiment.run_test(tmp_path, Path("manifest.csv"))


def write_completed_validation_fixture(root):
    (root / "protocol.json").write_text(json.dumps({
        "protocol_version": "metadata_pretrained_foundation_trial_001",
        "manifest_sha256": "manifest-hash",
        "selection_margin": 0.01,
    }, sort_keys=True) + "\n", encoding="utf-8")

    resource_rows = []
    for candidate_id, pr_auc, size_bytes in (
        ("tabpfn_v2", 0.22, 200),
        ("tabicl_v2", 0.21, 100),
    ):
        prediction_path = root / f"predictions_{candidate_id}.csv"
        prediction_path.write_text("y_true,score\n0,0.1\n1,0.9\n", encoding="utf-8")
        metric_path = root / f"validation_{candidate_id}.json"
        metric_path.write_text(json.dumps({
            "candidate_id": candidate_id,
            "status": "completed",
            "metrics": {"pr_auc": pr_auc},
            "prediction_artifact": prediction_path.name,
            "prediction_artifact_sha256": experiment.sha256_file(prediction_path),
            "model_identity": {
                "model_id": candidate_id,
                "checkpoint_id": f"{candidate_id}-checkpoint",
                "package_version": "pinned-test-version",
            },
        }, sort_keys=True) + "\n", encoding="utf-8")
        resource_rows.append(f"{candidate_id},{size_bytes},1.0,1000\n")

    (root / "validation_comparison.csv").write_text(
        "candidate_id,pr_auc\ntabpfn_v2,0.22\ntabicl_v2,0.21\n",
        encoding="utf-8",
    )
    (root / "resource_comparison.csv").write_text(
        "candidate_id,checkpoint_size_bytes,validation_inference_p50_ms,peak_rss_bytes\n"
        + "".join(resource_rows),
        encoding="utf-8",
    )


def mark_test_completed(root):
    (root / "test_lock.json").write_text(
        json.dumps({"status": "completed"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def replace_candidate_with_failure(root, candidate_id):
    failure_path = root / f"failure_{candidate_id}.json"
    failure_path.write_text(json.dumps({
        "candidate_id": candidate_id,
        "stage": "validation_inference",
        "error_type": "RuntimeError",
        "sanitized_message": "synthetic test failure",
        "environment": {"python": "test", "package_version": "pinned-test-version"},
    }, sort_keys=True) + "\n", encoding="utf-8")
    (root / f"validation_{candidate_id}.json").write_text(json.dumps({
        "candidate_id": candidate_id,
        "status": "failed",
        "model_identity": {
            "model_id": candidate_id,
            "checkpoint_id": f"{candidate_id}-checkpoint",
            "package_version": "pinned-test-version",
        },
        "failure_artifact": failure_path.name,
        "failure_artifact_sha256": experiment.sha256_file(failure_path),
    }, sort_keys=True) + "\n", encoding="utf-8")


def set_validation_pr_auc(root, candidate_id, pr_auc):
    path = root / f"validation_{candidate_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"]["pr_auc"] = pr_auc
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_one_failed_validation_fixture(root):
    write_completed_validation_fixture(root)
    replace_candidate_with_failure(root, "tabicl_v2")


def write_zero_completed_validation_fixture(root):
    write_completed_validation_fixture(root)
    replace_candidate_with_failure(root, "tabpfn_v2")
    replace_candidate_with_failure(root, "tabicl_v2")


def write_completed_validation_and_test_fixture(root):
    write_completed_validation_fixture(root)
    experiment.run_selection(root)
    (root / "selected_test_metrics.json").write_text(
        json.dumps({"candidate_id": "tabicl_v2", "status": "completed"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_report_contains_required_sections_and_contextual_semantics(tmp_path):
    write_completed_validation_and_test_fixture(tmp_path)

    experiment.write_report(tmp_path)
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")

    for heading in (
        "## Baseline", "## Pretrained candidates", "## Validation comparison",
        "## Validation decision", "## Final frozen test result",
        "## Calibration and class-imbalance interpretation",
        "## Engineering trade-off", "## Final model-role recommendation",
        "## Limitations",
    ):
        assert heading in report
    assert "contextual" in report.lower()
    assert "acute stroke probability" in report.lower()
    assert "clinical" in report.lower()


def test_artifact_hash_manifest_excludes_itself(tmp_path):
    (tmp_path / "validation_comparison.csv").write_text("candidate_id\n", encoding="utf-8")
    (tmp_path / "validation_metrics.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "artifact_hashes.json").write_text("{}\n", encoding="utf-8")

    manifest = experiment.write_artifact_hash_manifest(tmp_path)

    assert manifest["validation_comparison.csv"]
    assert manifest["validation_metrics.json"]
    assert "artifact_hashes.json" not in manifest


def test_selection_checksum_is_one_way(tmp_path):
    write_completed_validation_fixture(tmp_path)
    experiment.run_selection(tmp_path)

    snapshot = json.loads((tmp_path / "selection_frozen.json").read_text(encoding="utf-8"))
    checksum = (tmp_path / "selection_frozen.sha256").read_text(encoding="utf-8").strip()

    assert "selection_frozen.sha256" not in snapshot
    assert checksum == experiment.sha256_file(tmp_path / "selection_frozen.json")


def test_runner_imports_torch_before_sklearn_backed_project_modules():
    result = subprocess.run(
        [sys.executable, "-c", "import scripts.metadata_pretrained_foundation_experiment; import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_json_reader_accepts_windows_utf8_bom(tmp_path):
    path = tmp_path / "bom.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8-sig")

    assert experiment._read_json(path) == {"ok": True}
