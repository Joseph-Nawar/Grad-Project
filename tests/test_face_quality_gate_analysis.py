from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts import face_quality_gate_analysis as analysis


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(
    r"C:\Users\Asus\Downloads\Uni_work\Grad-Project\codex_workplace\rural-stroke-assist"
)
SNAPSHOT = EXTERNAL / "experiment_reports" / "face_quality_gate_analysis" / "starting_fingerprints.json"


def _example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["n1", "n2", "p1", "p2", "p3"],
            "class_label": ["NonStroke", "NonStroke", "Stroke", "Stroke", "Stroke"],
            "y_true": [0, 0, 1, 1, 1],
            "score": [0.1, 0.8, 0.7, 0.4, 0.9],
            "prediction": [0, 1, 1, 0, 1],
            "gate_accepted": [True, False, True, True, False],
        }
    )


def test_canonical_test_ids_are_unique_and_match_manifest_class_counts() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/face_split_manifest.csv")
    test_samples = analysis.canonical_test_samples(manifest)
    expected = manifest.loc[manifest["split"] == "test"]

    assert test_samples["sample_id"].is_unique
    assert len(test_samples) == len(expected)
    assert test_samples["class_label"].value_counts().to_dict() == expected[
        "class_label"
    ].value_counts().to_dict()


def test_prediction_alignment_requires_exactly_one_score_per_sample() -> None:
    samples = pd.DataFrame({"sample_id": ["a", "b"], "class_label": ["NonStroke", "Stroke"]})
    predictions = pd.DataFrame(
        {"file_hash": ["b", "a"], "true_label": ["Stroke", "NonStroke"], "stroke_probability": [0.8, 0.2]}
    )

    aligned = analysis.align_prediction_scores(samples, predictions)

    assert aligned["sample_id"].tolist() == ["a", "b"]
    assert aligned["score"].tolist() == [0.2, 0.8]
    with pytest.raises(ValueError):
        analysis.align_prediction_scores(samples, pd.concat([predictions, predictions.iloc[[0]]]))


def test_gate_alignment_requires_exactly_one_outcome_per_sample() -> None:
    samples = pd.DataFrame({"sample_id": ["a", "b"]})
    gate = pd.DataFrame({"sample_id": ["b", "a"], "gate_accepted": [False, True]})

    aligned = analysis.align_gate_outcomes(samples, gate)

    assert aligned["gate_accepted"].tolist() == [True, False]
    with pytest.raises(ValueError):
        analysis.align_gate_outcomes(samples, gate.iloc[:1])


def test_scores_are_finite_and_bounded_and_classifier_threshold_is_frozen() -> None:
    analysis.validate_score_vector([0.0, 0.5, 1.0])
    for scores in ([float("nan")], [-0.01], [1.01]):
        with pytest.raises(ValueError):
            analysis.validate_score_vector(scores)
    assert analysis.CLASSIFIER_THRESHOLD == 0.5


def test_wilson_interval_has_valid_ordered_bounds() -> None:
    lower, upper = analysis.wilson_interval(50, 100)
    assert 0 <= lower < 0.5 < upper <= 1
    assert analysis.wilson_interval(0, 0) == (None, None)
    with pytest.raises(ValueError):
        analysis.wilson_interval(2, 1)


def test_coverage_annotation_is_anchored_above_confidence_interval() -> None:
    for estimate, upper in ((0.336, 0.390), (0.375, 0.442), (0.264, 0.353)):
        assert analysis.coverage_label_position(estimate, upper) > upper


def test_coverage_and_contingency_tables_reconcile() -> None:
    frame = _example_frame()
    summary, by_class, contingency = analysis.coverage_tables(frame)

    assert int(summary.loc[summary["group"] == "overall", "accepted"].iloc[0]) == 3
    for label, values in contingency["class_counts"].items():
        row = by_class.loc[by_class["class_label"] == label].iloc[0]
        assert int(row["accepted"]) == values["accepted"]
        assert int(row["rejected"]) == values["rejected"]
        assert values["accepted"] + values["rejected"] == int(row["n"])
    assert sum(
        contingency["accepted_rejected_by_class"][state][label]
        for state in ("accepted", "rejected")
        for label in ("negative", "positive")
    ) == len(frame)


def test_subset_confusion_matrices_sum_to_subset_n() -> None:
    frame = _example_frame()
    metrics = analysis.subset_performance(frame, "accepted", frame["gate_accepted"])
    assert sum(sum(row) for row in json.loads(metrics["confusion_matrix"])) == metrics["n"]


def test_undefined_auc_metrics_remain_unavailable() -> None:
    frame = _example_frame().iloc[:2].copy()
    metrics = analysis.subset_performance(frame, "negative_only", [True, True])
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None


def test_stratified_bootstrap_is_deterministic_for_fixed_seed() -> None:
    frame = _example_frame()
    first = analysis.stratified_bootstrap_intervals(frame["y_true"], frame["score"], iterations=100, seed=9)
    second = analysis.stratified_bootstrap_intervals(frame["y_true"], frame["score"], iterations=100, seed=9)
    assert first == second


def test_test_analysis_excludes_train_and_validation_ids() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/face_split_manifest.csv")
    test_samples = analysis.canonical_test_samples(manifest)
    analysis.assert_test_only_samples(test_samples, manifest)
    assert set(test_samples["sample_id"]).isdisjoint(
        set(manifest.loc[manifest["split"].isin(["train", "val"]), "file_hash"])
    )


def test_analysis_script_does_not_call_model_training_apis() -> None:
    tree = ast.parse(Path(analysis.__file__).read_text(encoding="utf-8"))
    forbidden = {"fit", "fit_generator", "train", "train_on_batch"}
    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (isinstance(node.func, ast.Attribute) or isinstance(node.func, ast.Name))
    ]
    assert not forbidden.intersection(calls)


def test_production_and_model_fingerprints_match_external_start_snapshot() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert analysis.verify_starting_fingerprints(ROOT, snapshot) == []


def test_output_path_guard_rejects_paths_outside_experiment_directory() -> None:
    output_root = ROOT / "reports/experiments/face_quality_gate_analysis"
    assert analysis.output_path(output_root, "sample_results.csv").parent == output_root.resolve()
    with pytest.raises(ValueError):
        analysis.output_path(output_root, "../outside.csv")


def test_analysis_script_can_start_from_scripts_directory() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(Path(analysis.__file__).resolve()), "--help"],
        cwd=ROOT / "scripts",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--fingerprint-snapshot" in result.stdout


def test_nonreproducing_persisted_prediction_export_is_marked_nonreusable() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/face_split_manifest.csv")
    predictions = pd.read_csv(
        ROOT / "data/processed/experiments/face_trial_003_mobilenetv2_balanced_160_predictions.csv"
    )
    historical = json.loads(
        (ROOT / "reports/evaluation/phase4/final_complete/face.json").read_text(encoding="utf-8")
    )

    result = analysis.saved_prediction_reuse_check(
        analysis.canonical_test_samples(manifest), predictions, historical["direct"]
    )

    assert result["reusable"] is False
    assert result["failed_metric_count"] > 0


def test_historical_rejection_reproduction_exposes_passed_invariant() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/face_split_manifest.csv")
    samples = analysis.canonical_test_samples(manifest)
    rejected = pd.read_csv(ROOT / "reports/evaluation/phase4/final_complete/face_rejections.csv")
    path_to_id = {
        str(row.path).replace("/", "\\").casefold(): str(row.sample_id)
        for row in samples.itertuples(index=False)
    }
    gate_rows = []
    for row in rejected.itertuples(index=False):
        codes = [code for code in str(row.quality_findings).split(";") if code]
        gate_rows.append(
            {
                "sample_id": path_to_id[str(row.path).replace("/", "\\").casefold()],
                "gate_accepted": False,
                "quality_reasons": json.dumps(codes),
                "quality_findings": json.dumps([{"code": code} for code in codes]),
                "quality_measurements": json.dumps({}),
            }
        )

    result = analysis.verify_historical_rejections(ROOT, samples, pd.DataFrame(gate_rows))

    assert result["passed"] is True
    assert result["matches"] is True
    measurements = analysis.attach_historical_rejection_measurements(
        ROOT, samples, pd.DataFrame(gate_rows)
    )
    first = json.loads(measurements.iloc[0]["quality_measurements"])
    assert "phase4_persisted_face_count" in first
    assert first["phase4_measurement_source"].endswith("face_rejections.csv")
