"""Final held-out operational coverage analysis for the frozen face branch.

This script reuses the persisted Trial 003 scores and invokes the current
FaceAdapter with its unchanged OpenCV quality assessor. The injected runner is
deliberately inert: adapter scores are discarded, because the aligned frozen
scores already exist. No model is loaded or trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import average_precision_score, roc_auc_score

from rural_stroke_assist.evaluation.metrics import classification_metrics, metric_dict
from rural_stroke_assist.inference.contracts import QualityStatus
from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.registry import BaselineRegistry
from rural_stroke_assist.quality.face_quality import OpenCVFaceQualityAssessor


CLASSIFIER_THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42
METRIC_TOLERANCE = 1e-8
EXPERIMENT_RELATIVE_PATH = Path("reports/experiments/face_quality_gate_analysis")
DEFAULT_EXTERNAL_ROOT = Path(
    r"C:\Users\Asus\Downloads\Uni_work\Grad-Project\codex_workplace\rural-stroke-assist"
)
DEFAULT_FINGERPRINT_SNAPSHOT = (
    DEFAULT_EXTERNAL_ROOT
    / "experiment_reports"
    / "face_quality_gate_analysis"
    / "starting_fingerprints.json"
)
SOURCE_FIELD_NAMES = {
    "source",
    "source_dataset",
    "source_name",
    "origin",
    "origin_dataset",
    "acquisition_source",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_starting_fingerprints(repo_root: Path, snapshot: dict[str, Any]) -> list[str]:
    """Return baseline source/artifact paths that are missing or have drifted."""
    mismatches: list[str] = []
    for relative, expected in snapshot.get("files", {}).items():
        path = repo_root / Path(relative)
        if not path.is_file():
            mismatches.append(f"missing: {relative}")
        elif sha256_file(path).lower() != str(expected["sha256"]).lower():
            mismatches.append(f"hash_mismatch: {relative}")
    return mismatches


def output_path(output_root: Path, relative: str | Path) -> Path:
    """Resolve an output path and reject traversal outside the experiment dir."""
    root = output_root.resolve()
    result = (root / Path(relative)).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Analysis output escapes approved directory: {relative}") from exc
    return result


def canonical_test_samples(split_manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"file_hash", "class_label", "split", "path"}
    missing = required - set(split_manifest.columns)
    if missing:
        raise ValueError(f"Face split manifest is missing columns: {sorted(missing)}")
    test = split_manifest.loc[split_manifest["split"].astype(str).str.lower() == "test"].copy()
    if test.empty:
        raise ValueError("Face split manifest has no test rows.")
    test["sample_id"] = test["file_hash"].astype(str)
    if test["sample_id"].duplicated().any():
        raise ValueError("Canonical test sample IDs are duplicated.")
    if not test["class_label"].isin(["NonStroke", "Stroke"]).all():
        raise ValueError("Canonical test manifest contains an unsupported class label.")
    test["y_true"] = test["class_label"].map({"NonStroke": 0, "Stroke": 1}).astype(int)
    return test.reset_index(drop=True)


def assert_test_only_samples(test_samples: pd.DataFrame, full_manifest: pd.DataFrame) -> None:
    if not (test_samples["split"].astype(str).str.lower() == "test").all():
        raise ValueError("Non-test split rows entered the face test analysis.")
    test_ids = set(test_samples["sample_id"].astype(str))
    non_test_ids = set(
        full_manifest.loc[
            full_manifest["split"].astype(str).str.lower().isin(["train", "val", "validation"]),
            "file_hash",
        ].astype(str)
    )
    if test_ids & non_test_ids:
        raise ValueError("Train/validation sample IDs overlap the canonical test analysis.")


def validate_score_vector(scores: Iterable[float]) -> None:
    values = np.asarray(list(scores), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Every raw classifier score must be finite.")
    if not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("Every raw classifier score must be in [0, 1].")


def align_prediction_scores(
    test_samples: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    required = {"file_hash", "stroke_probability", "true_label"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {sorted(missing)}")
    pred = predictions.copy()
    pred["sample_id"] = pred["file_hash"].astype(str)
    if pred["sample_id"].duplicated().any():
        raise ValueError("Prediction artifact contains duplicate sample IDs.")
    reference_ids = test_samples["sample_id"].astype(str).tolist()
    if len(pred) != len(reference_ids) or set(pred["sample_id"]) != set(reference_ids):
        raise ValueError("Prediction/test alignment is not exactly one-to-one.")
    pred["score"] = pd.to_numeric(pred["stroke_probability"], errors="raise").astype(float)
    validate_score_vector(pred["score"])
    labels = test_samples.set_index("sample_id")["class_label"]
    pred_labels = pred.set_index("sample_id")["true_label"].astype(str)
    if not labels.sort_index().equals(pred_labels.reindex(labels.index).sort_index()):
        raise ValueError("Prediction labels do not match the canonical test manifest.")
    pred["prediction"] = (pred["score"] >= CLASSIFIER_THRESHOLD).astype(int)
    if "predicted_label_encoded" in pred:
        stored = pd.to_numeric(pred["predicted_label_encoded"], errors="raise").astype(int)
        if not np.array_equal(stored.to_numpy(), pred["prediction"].to_numpy()):
            raise ValueError("Persisted predicted labels do not match the frozen 0.5 threshold.")
    aligned = test_samples.merge(
        pred[["sample_id", "score", "prediction"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned["score"].isna().any() or len(aligned) != len(test_samples):
        raise ValueError("At least one canonical test sample has no raw classifier score.")
    return aligned


def align_gate_outcomes(test_samples: pd.DataFrame, gate_outcomes: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "gate_accepted"}
    missing = required - set(gate_outcomes.columns)
    if missing:
        raise ValueError(f"Gate evidence is missing columns: {sorted(missing)}")
    gate = gate_outcomes.copy()
    gate["sample_id"] = gate["sample_id"].astype(str)
    if gate["sample_id"].duplicated().any():
        raise ValueError("Quality-gate evidence contains duplicate sample IDs.")
    sample_ids = test_samples["sample_id"].astype(str)
    if len(gate) != len(sample_ids) or set(gate["sample_id"]) != set(sample_ids):
        raise ValueError("Quality-gate/test alignment is not exactly one-to-one.")
    if not gate["gate_accepted"].map(lambda item: isinstance(item, (bool, np.bool_))).all():
        raise ValueError("Quality-gate outcomes must be explicit boolean values.")
    aligned = test_samples.merge(
        gate,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned["gate_accepted"].isna().any():
        raise ValueError("At least one canonical test sample has no quality-gate outcome.")
    accepted = int(aligned["gate_accepted"].sum())
    rejected = int((~aligned["gate_accepted"].astype(bool)).sum())
    if accepted + rejected != len(aligned):
        raise AssertionError("Accepted and rejected counts do not sum to the test total.")
    return aligned


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly between zero and one.")
    if total == 0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = z * np.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total))) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _coverage_row(scope: str, n: int, accepted: int) -> dict[str, Any]:
    rejected = n - accepted
    low, high = wilson_interval(accepted, n)
    reject_low, reject_high = wilson_interval(rejected, n)
    return {
        "group": scope,
        "n": n,
        "accepted": accepted,
        "rejected": rejected,
        "coverage": accepted / n if n else None,
        "coverage_ci_95_lower": low,
        "coverage_ci_95_upper": high,
        "rejection_rate": rejected / n if n else None,
        "rejection_ci_95_lower": reject_low,
        "rejection_ci_95_upper": reject_high,
    }


def coverage_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"class_label", "gate_accepted"}
    if required - set(frame.columns):
        raise ValueError("Coverage table input requires class_label and gate_accepted.")
    if frame["gate_accepted"].isna().any():
        raise ValueError("Coverage cannot be calculated with missing gate states.")
    overall_accepted = int(frame["gate_accepted"].astype(bool).sum())
    summary = pd.DataFrame([_coverage_row("overall", len(frame), overall_accepted)])
    by_class_rows: list[dict[str, Any]] = []
    class_counts: dict[str, dict[str, Any]] = {}
    for label in ("NonStroke", "Stroke"):
        group = frame.loc[frame["class_label"] == label]
        accepted = int(group["gate_accepted"].astype(bool).sum())
        row = _coverage_row(label, len(group), accepted)
        row["class_label"] = label
        row["negative_n"] = int((group["class_label"] == "NonStroke").sum())
        row["positive_n"] = int((group["class_label"] == "Stroke").sum())
        by_class_rows.append(row)
        class_counts[label] = {
            "n": len(group),
            "accepted": accepted,
            "rejected": len(group) - accepted,
            "coverage": accepted / len(group) if len(group) else None,
        }
    by_class = pd.DataFrame(by_class_rows)
    negative_coverage = class_counts["NonStroke"]["coverage"]
    positive_coverage = class_counts["Stroke"]["coverage"]
    positive = class_counts["Stroke"]
    negative = class_counts["NonStroke"]
    table = [
        [positive["accepted"], positive["rejected"]],
        [negative["accepted"], negative["rejected"]],
    ]
    fisher = fisher_exact(table, alternative="two-sided")
    contingency = {
        "orientation": {
            "rows": ["Stroke (positive class)", "NonStroke (negative class)"],
            "columns": ["accepted", "rejected"],
        },
        "table": table,
        "class_counts": class_counts,
        "accepted_rejected_by_class": {
            "accepted": {
                "negative": negative["accepted"],
                "positive": positive["accepted"],
            },
            "rejected": {
                "negative": negative["rejected"],
                "positive": positive["rejected"],
            },
        },
        "class_conditional_coverage_difference": {
            "positive_minus_negative": positive_coverage - negative_coverage
            if positive_coverage is not None and negative_coverage is not None
            else None,
            "absolute": abs(positive_coverage - negative_coverage)
            if positive_coverage is not None and negative_coverage is not None
            else None,
            "positive_to_negative_coverage_ratio": positive_coverage / negative_coverage
            if positive_coverage is not None and negative_coverage not in (None, 0)
            else None,
        },
        "fisher_exact_two_sided": {
            "odds_ratio_positive_vs_negative_acceptance": float(fisher.statistic),
            "p_value": float(fisher.pvalue),
            "interpretation": "descriptive class-dependent acceptance disparity test",
        },
    }
    assert sum(sum(row) for row in table) == len(frame)
    assert overall_accepted + int((~frame["gate_accepted"].astype(bool)).sum()) == len(frame)
    return summary, by_class, contingency


def subset_performance(
    frame: pd.DataFrame,
    subset_name: str,
    mask: Iterable[bool],
    threshold: float = CLASSIFIER_THRESHOLD,
) -> dict[str, Any]:
    if threshold != CLASSIFIER_THRESHOLD:
        raise ValueError("The frozen classifier decision threshold must remain exactly 0.5.")
    selected = frame.loc[np.asarray(list(mask), dtype=bool)].copy()
    y = selected["y_true"].to_numpy(dtype=int)
    scores = selected["score"].to_numpy(dtype=float)
    validate_score_vector(scores)
    if len(y) != len(scores):
        raise ValueError("Subset labels and scores are not aligned.")
    if len(y):
        summary = metric_dict(classification_metrics(y, scores, threshold=CLASSIFIER_THRESHOLD))
        matrix = summary["confusion_matrix"]
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        accuracy = (tn + tp) / len(y)
        result = {key: value for key, value in summary.items() if key != "confusion_matrix"}
    else:
        matrix = ((0, 0), (0, 0))
        accuracy = None
        result = {
            "n": 0,
            "positive": 0,
            "negative": 0,
            "sensitivity": None,
            "specificity": None,
            "precision": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
            "brier": None,
            "calibration_error": None,
        }
    result.update(
        {
            "subset": subset_name,
            "threshold": CLASSIFIER_THRESHOLD,
            "accuracy": accuracy,
            "confusion_matrix": json.dumps(matrix, separators=(",", ":")),
        }
    )
    return result


def stratified_bootstrap_intervals(
    y_true: Iterable[int],
    scores: Iterable[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(scores), dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Bootstrap labels and scores must have equal non-zero length.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Bootstrap requires binary labels encoded as 0 and 1.")
    validate_score_vector(p)
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    if len(np.unique(y)) < 2:
        return [
            {
                "metric": name,
                "estimate": None,
                "ci_95_lower": None,
                "ci_95_upper": None,
                "iterations": iterations,
                "seed": seed,
                "status": "unavailable: both classes are required",
            }
            for name in ("roc_auc", "pr_auc")
        ]
    point = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == label) for label in (0, 1)]
    draws: dict[str, list[float]] = {"roc_auc": [], "pr_auc": []}
    for _ in range(iterations):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in strata]
        )
        sample_y, sample_p = y[indices], p[indices]
        draws["roc_auc"].append(float(roc_auc_score(sample_y, sample_p)))
        draws["pr_auc"].append(float(average_precision_score(sample_y, sample_p)))
    return [
        {
            "metric": name,
            "estimate": point[name],
            "ci_95_lower": float(np.quantile(values, 0.025)),
            "ci_95_upper": float(np.quantile(values, 0.975)),
            "iterations": iterations,
            "seed": seed,
            "status": "available",
        }
        for name, values in draws.items()
    ]


def explicit_source_field(columns: Iterable[str]) -> str | None:
    return next((str(column) for column in columns if str(column).lower() in SOURCE_FIELD_NAMES), None)


def _finding_dict(finding: Any) -> dict[str, str]:
    status = finding.status.value if hasattr(finding.status, "value") else str(finding.status)
    return {"code": str(finding.code), "message": str(finding.message), "status": status}


def _numeric_measurements(findings: list[dict[str, str]]) -> dict[str, float | None]:
    blur: float | None = None
    luminance: float | None = None
    for finding in findings:
        blur_match = re.search(r"variance=([-+]?\d+(?:\.\d+)?)", finding["message"])
        luminance_match = re.search(r"luminance[^()]*\(([-+]?\d+(?:\.\d+)?)\)", finding["message"], re.I)
        if blur_match:
            blur = float(blur_match.group(1))
        if luminance_match:
            luminance = float(luminance_match.group(1))
    return {"blur_variance_reported_in_warning": blur, "mean_luminance_reported_in_warning": luminance}


class _DiscardingRunner:
    """Adapter-compatible runner whose score is intentionally not retained."""

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return np.asarray([[0.5]], dtype=np.float32)


def run_current_quality_gate(
    repo_root: Path, test_samples: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = BaselineRegistry.from_file(repo_root / "config/baseline_registry.json")
    assessor = OpenCVFaceQualityAssessor()
    adapter = FaceAdapter(registry=registry, runner=_DiscardingRunner(), quality_assessor=assessor)
    rows: list[dict[str, Any]] = []
    image_hash_matches = True
    for sample in test_samples.itertuples(index=False):
        image_path = Path(str(sample.path)).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Canonical face test image is missing: {image_path}")
        image_hash_matches &= sha256_file(image_path).lower() == str(sample.file_hash).lower()
        if not image_hash_matches:
            raise ValueError(f"Canonical image hash does not match manifest: {sample.sample_id}")
        evidence = adapter.infer(image_path)
        findings = [_finding_dict(item) for item in evidence.quality_findings]
        reasons = [item["code"] for item in findings if item["status"] == QualityStatus.REJECT.value]
        relative_identity = image_path.relative_to(repo_root.resolve()).as_posix()
        measurements = _numeric_measurements(findings)
        if hasattr(sample, "width"):
            measurements["manifest_image_width_px"] = int(sample.width)
        if hasattr(sample, "height"):
            measurements["manifest_image_height_px"] = int(sample.height)
        rows.append(
            {
                "sample_id": str(sample.sample_id),
                "relative_file_identity": relative_identity,
                "gate_accepted": bool(evidence.available),
                "quality_status": evidence.quality_status.value,
                "quality_reasons": json.dumps(reasons, separators=(",", ":")),
                "quality_findings": json.dumps(findings, separators=(",", ":"), ensure_ascii=False),
                "quality_measurements": json.dumps(measurements, separators=(",", ":")),
            }
        )
    quality_config = {
        "assessor": "OpenCVFaceQualityAssessor",
        "min_face_size_px": int(assessor.min_face_size_px),
        "min_image_dimension_px": int(assessor.min_image_dimension_px),
        "blur_variance_threshold": float(assessor.blur_variance_threshold),
        "min_luminance": float(assessor.min_luminance),
        "max_luminance": float(assessor.max_luminance),
        "checks": ["image dimensions", "Haar face count", "minimum detected face size", "blur warning", "lighting warning", "pose not assessed"],
        "quality_measurement_limitation": "The current gate reports blur/luminance values only when its warning message includes them; it does not expose detector face count or box size on accepted inputs. Existing Phase 4 per-rejection count/size values are included only for rejected sample identities.",
    }
    return pd.DataFrame(rows), {"quality_config": quality_config, "image_hashes_match_manifest": image_hash_matches}


def _path_identity(value: str) -> str:
    return str(value).replace("/", "\\").casefold()


def verify_historical_rejections(
    repo_root: Path, test_samples: pd.DataFrame, gate_outcomes: pd.DataFrame
) -> dict[str, Any]:
    path = repo_root / "reports/evaluation/phase4/final_complete/face_rejections.csv"
    historical = pd.read_csv(path)
    by_id = test_samples.set_index("sample_id")
    current_rejected = gate_outcomes.loc[~gate_outcomes["gate_accepted"].astype(bool)]
    old_by_path = {_path_identity(row.path): row for row in historical.itertuples(index=False)}
    current_by_path: dict[str, Any] = {}
    for row in current_rejected.itertuples(index=False):
        source_path = str(by_id.loc[str(row.sample_id), "path"])
        current_by_path[_path_identity(source_path)] = row
    if set(old_by_path) != set(current_by_path):
        return {
            "matches": False,
            "passed": False,
            "historical_rejected_n": len(old_by_path),
            "replayed_rejected_n": len(current_by_path),
            "identity_mismatch_n": len(set(old_by_path) ^ set(current_by_path)),
            "reason_code_mismatch_n": None,
        }
    reason_mismatches = 0
    for path_key, old_row in old_by_path.items():
        current = current_by_path[path_key]
        old_codes = [item for item in str(old_row.quality_findings).split(";") if item]
        new_codes = [item["code"] for item in json.loads(current.quality_findings)]
        if Counter(old_codes) != Counter(new_codes):
            reason_mismatches += 1
    return {
        "matches": reason_mismatches == 0,
        "passed": reason_mismatches == 0,
        "historical_rejected_n": len(old_by_path),
        "replayed_rejected_n": len(current_by_path),
        "identity_mismatch_n": 0,
        "reason_code_mismatch_n": reason_mismatches,
    }


def attach_historical_rejection_measurements(
    repo_root: Path, test_samples: pd.DataFrame, gate_outcomes: pd.DataFrame
) -> pd.DataFrame:
    """Attach existing Phase 4 face counts/sizes to identity-matched rejects."""
    path = repo_root / "reports/evaluation/phase4/final_complete/face_rejections.csv"
    historical = pd.read_csv(path)
    historical_by_path = {
        _path_identity(str(row.path)): row for row in historical.itertuples(index=False)
    }
    source_path_by_id = test_samples.set_index("sample_id")["path"].astype(str).to_dict()
    result = gate_outcomes.copy()
    for index, row in result.iterrows():
        if bool(row["gate_accepted"]):
            continue
        identity = _path_identity(source_path_by_id[str(row["sample_id"])])
        persisted = historical_by_path.get(identity)
        if persisted is None:
            raise ValueError(f"No persisted Phase 4 measurement row for rejected sample {row['sample_id']}.")
        measurements = json.loads(str(row["quality_measurements"]))
        face_count = getattr(persisted, "face_count", None)
        face_size = getattr(persisted, "face_size_px", None)
        measurements["phase4_persisted_face_count"] = int(face_count) if pd.notna(face_count) else None
        measurements["phase4_persisted_face_size_px"] = float(face_size) if pd.notna(face_size) else None
        measurements["phase4_measurement_source"] = "reports/evaluation/phase4/final_complete/face_rejections.csv"
        result.at[index, "quality_measurements"] = json.dumps(measurements, separators=(",", ":"))
    return result


def rejection_reason_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rejected = frame.loc[~frame["gate_accepted"].astype(bool)]
    codes_by_sample = {
        str(row.sample_id): json.loads(row.quality_reasons)
        for row in rejected.itertuples(index=False)
    }
    counts: Counter[str] = Counter(code for codes in codes_by_sample.values() for code in set(codes))
    rows: list[dict[str, Any]] = []
    for reason in sorted(counts):
        selected_ids = {sample_id for sample_id, codes in codes_by_sample.items() if reason in codes}
        by_label = {}
        for label in ("NonStroke", "Stroke"):
            class_rejected = rejected.loc[rejected["class_label"] == label]
            count = sum(str(item) in selected_ids for item in class_rejected["sample_id"])
            class_total = int((frame["class_label"] == label).sum())
            by_label[label] = {"count": int(count), "rate_within_class": count / class_total if class_total else None}
        rows.append(
            {
                "rejection_reason": reason,
                "count": counts[reason],
                "percent_of_rejected_samples": 100.0 * counts[reason] / len(rejected) if len(rejected) else None,
                "nonstroke_count": by_label["NonStroke"]["count"],
                "nonstroke_rate_of_class": by_label["NonStroke"]["rate_within_class"],
                "stroke_count": by_label["Stroke"]["count"],
                "stroke_rate_of_class": by_label["Stroke"]["rate_within_class"],
                "multiple_reasons_can_apply": True,
            }
        )
    combo_counter: Counter[tuple[str, ...]] = Counter(
        tuple(sorted(set(codes))) for codes in codes_by_sample.values()
    )
    combo_rows: list[dict[str, Any]] = []
    for combo, count in sorted(combo_counter.items(), key=lambda item: (-item[1], item[0])):
        matching = [sample_id for sample_id, codes in codes_by_sample.items() if tuple(sorted(set(codes))) == combo]
        by_label = {
            label: sum(str(frame.loc[frame["sample_id"] == sid, "class_label"].iloc[0]) == label for sid in matching)
            for label in ("NonStroke", "Stroke")
        }
        combo_rows.append(
            {
                "reason_combination": json.dumps(combo, separators=(",", ":")),
                "count": count,
                "percent_of_rejected_samples": 100.0 * count / len(rejected) if len(rejected) else None,
                "nonstroke_count": by_label["NonStroke"],
                "stroke_count": by_label["Stroke"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(combo_rows)


def coverage_aware_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, y in (("NonStroke", 0), ("Stroke", 1)):
        group = frame.loc[frame["class_label"] == label]
        accepted = group.loc[group["gate_accepted"].astype(bool)]
        rejected_n = int((~group["gate_accepted"].astype(bool)).sum())
        rows.append(
            {
                "dataset_class": label,
                "accepted_and_predicted_positive": int((accepted["prediction"] == 1).sum()),
                "accepted_and_predicted_negative": int((accepted["prediction"] == 0).sum()),
                "rejected_unavailable_face_evidence": rejected_n,
                "n": len(group),
                "meaning": "Rejected rows represent unavailable face evidence, not a negative clinical prediction.",
            }
        )
    return pd.DataFrame(rows)


def score_distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        ("accepted positive", "Stroke", True),
        ("rejected positive", "Stroke", False),
        ("accepted negative", "NonStroke", True),
        ("rejected negative", "NonStroke", False),
    ]
    for name, label, accepted in groups:
        selected = frame.loc[
            (frame["class_label"] == label) & (frame["gate_accepted"].astype(bool) == accepted), "score"
        ].astype(float)
        rows.append(
            {
                "group": name,
                "class_label": label,
                "gate_accepted": accepted,
                "n": int(selected.size),
                "median_score": float(selected.median()) if selected.size else None,
                "q1_score": float(selected.quantile(0.25)) if selected.size else None,
                "q3_score": float(selected.quantile(0.75)) if selected.size else None,
                "iqr_score": float(selected.quantile(0.75) - selected.quantile(0.25)) if selected.size else None,
                "mean_score": float(selected.mean()) if selected.size else None,
                "standard_deviation_score": float(selected.std(ddof=1)) if selected.size > 1 else None,
            }
        )
    return pd.DataFrame(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def _plot_coverage(summary: pd.DataFrame, by_class: pd.DataFrame, path: Path) -> None:
    lookup = {str(row.group): row for row in summary.itertuples(index=False)}
    lookup.update({str(row.class_label): row for row in by_class.itertuples(index=False)})
    labels = ["Overall", "Negative", "Positive"]
    keys = ["overall", "NonStroke", "Stroke"]
    values = [float(lookup[key].coverage) for key in keys]
    lower = [float(lookup[key].coverage_ci_95_lower) for key in keys]
    upper = [float(lookup[key].coverage_ci_95_upper) for key in keys]
    yerr = np.asarray([[value - low for value, low in zip(values, lower)], [high - value for value, high in zip(values, upper)]])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(labels, np.asarray(values) * 100.0, color=["#537895", "#6c9a8b", "#bf6b63"], width=0.62)
    ax.errorbar(np.arange(len(values)), np.asarray(values) * 100.0, yerr=yerr * 100.0, fmt="none", ecolor="#252525", capsize=5, linewidth=1.2)
    for bar, value, upper_value, key in zip(bars, values, upper, keys):
        n = int(lookup[key].n)
        accepted = int(lookup[key].accepted)
        label_y = coverage_label_position(value, upper_value) * 100.0
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{accepted}/{n}  {value:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Operational face-branch coverage (%)")
    ax.set_ylim(0, max(100, float(max(np.asarray(values) * 100.0 + yerr[1] * 100.0)) + 12))
    ax.set_title("Held-out face-branch coverage with 95% Wilson intervals")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def coverage_label_position(estimate: float, upper_ci: float) -> float:
    """Return a fractional y position just above a coverage interval."""
    return max(float(estimate), float(upper_ci)) + 0.025


def _plot_score_distributions(frame: pd.DataFrame, path: Path) -> None:
    groups = [
        ("Accepted\npositive", "Stroke", True, "#bf6b63"),
        ("Rejected\npositive", "Stroke", False, "#e0aaa5"),
        ("Accepted\nnegative", "NonStroke", True, "#537895"),
        ("Rejected\nnegative", "NonStroke", False, "#9bb7c8"),
    ]
    values = [
        frame.loc[(frame["class_label"] == label) & (frame["gate_accepted"].astype(bool) == accepted), "score"].astype(float).to_numpy()
        for _, label, accepted, _ in groups
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    box = ax.boxplot(values, patch_artist=True, showmeans=True, showfliers=True, widths=0.58)
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(group[3])
        patch.set_alpha(0.78)
        patch.set_edgecolor("#333333")
    ax.set_xticks(np.arange(1, len(groups) + 1))
    ax.set_xticklabels([group[0] for group in groups])
    ax.set_ylabel("Frozen MobileNetV2 Stroke-class score")
    ax.set_ylim(0, 1)
    ax.set_title("Classifier score by gate outcome and dataset class")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for index, sample_scores in enumerate(values, start=1):
        ax.text(index, 0.985, f"n={len(sample_scores)}", ha="center", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_rejection_reasons(reason_table: pd.DataFrame, path: Path) -> None:
    if reason_table.empty:
        raise ValueError("Cannot draw rejection reasons without real granular gate reasons.")
    labels = reason_table["rejection_reason"].astype(str).tolist()
    nonstroke = reason_table["nonstroke_rate_of_class"].astype(float).to_numpy() * 100.0
    stroke = reason_table["stroke_rate_of_class"].astype(float).to_numpy() * 100.0
    positions = np.arange(len(labels))
    height = min(7.4, max(4.2, 0.55 * len(labels) + 1.7))
    fig, ax = plt.subplots(figsize=(8.6, height))
    ax.barh(positions - 0.18, nonstroke, height=0.34, label="Negative class", color="#537895")
    ax.barh(positions + 0.18, stroke, height=0.34, label="Positive class", color="#bf6b63")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Rejected samples with reason (% of class)")
    ax.set_title("Operational gate rejection reasons by dataset class")
    ax.legend(frameon=False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(0.99, -0.12, "Reasons may overlap within a sample.", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compare_metrics(subset: str, observed: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    fields = [
        "n", "positive", "negative", "roc_auc", "pr_auc", "sensitivity", "specificity",
        "precision", "f1", "brier", "calibration_error",
    ]
    for name in fields:
        if name not in expected:
            continue
        actual, target = observed[name], expected[name]
        if name in {"n", "positive", "negative"}:
            passed = int(actual) == int(target)
        else:
            passed = actual is not None and abs(float(actual) - float(target)) <= METRIC_TOLERANCE
        comparisons.append(
            {"subset": subset, "metric": name, "observed": actual, "expected": target, "passed": passed}
        )
    expected_matrix = json.dumps(expected["confusion_matrix"], separators=(",", ":"))
    comparisons.append(
        {
            "subset": subset,
            "metric": "confusion_matrix",
            "observed": observed["confusion_matrix"],
            "expected": expected_matrix,
            "passed": observed["confusion_matrix"] == expected_matrix,
        }
    )
    return comparisons


def saved_prediction_reuse_check(
    test_samples: pd.DataFrame, predictions: pd.DataFrame, historical_direct: dict[str, Any]
) -> dict[str, Any]:
    """Determine whether persisted scores reproduce the frozen direct evidence."""
    aligned = align_prediction_scores(test_samples, predictions)
    observed = subset_performance(aligned, "full held-out test set", np.ones(len(aligned), dtype=bool))
    comparisons = _compare_metrics("direct", observed, historical_direct)
    failures = [item for item in comparisons if not item["passed"]]
    return {
        "reusable": not failures,
        "failed_metric_count": len(failures),
        "comparisons": comparisons,
        "observed_direct_metrics": observed,
        "tolerance": METRIC_TOLERANCE,
    }


def infer_frozen_model_scores(
    repo_root: Path, test_samples: pd.DataFrame, expected_model_hash: str
) -> pd.DataFrame:
    """Run one inference pass over test images with the selected frozen model."""
    from rural_stroke_assist.modeling.face_inference import preprocess_face_image

    registry = BaselineRegistry.from_file(repo_root / "config/baseline_registry.json")
    model_path = registry.path_for("face")
    before = sha256_file(model_path)
    if before.lower() != expected_model_hash.lower():
        raise RuntimeError("Selected MobileNetV2 hash changed before held-out inference.")
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    scores: list[float] = []
    for sample in test_samples.itertuples(index=False):
        image_path = Path(str(sample.path)).resolve()
        batch = preprocess_face_image(image_path, image_size=(160, 160))
        raw = np.asarray(model.predict(batch, verbose=0))
        if raw.size != 1:
            raise ValueError(f"Selected model returned unexpected output shape for {sample.sample_id}: {raw.shape}")
        scores.append(float(raw.reshape(-1)[0]))
    after = sha256_file(model_path)
    if after != before:
        raise RuntimeError("Selected MobileNetV2 artifact hash changed during held-out inference.")
    validate_score_vector(scores)
    return pd.DataFrame(
        {
            "file_hash": test_samples["sample_id"].astype(str).to_numpy(),
            "true_label": test_samples["class_label"].astype(str).to_numpy(),
            "stroke_probability": scores,
            "score_source": "frozen selected MobileNetV2 inference",
        }
    )


def _assert_historical_metrics(
    direct: dict[str, Any], accepted: dict[str, Any], historical_face: dict[str, Any]
) -> dict[str, Any]:
    direct_expected = historical_face["direct"]
    accepted_expected = historical_face["adapter_aware"]
    comparisons = _compare_metrics("direct", direct, direct_expected)
    comparisons.extend(_compare_metrics("accepted", accepted, accepted_expected))
    failures = [item for item in comparisons if not item["passed"]]
    return {
        "passed": not failures,
        "absolute_tolerance_for_float_metrics": METRIC_TOLERANCE,
        "count_and_confusion_matrix_tolerance": "exact",
        "comparisons": comparisons,
        "failures": failures,
    }


def run_analysis(repo_root: Path, fingerprint_snapshot: Path = DEFAULT_FINGERPRINT_SNAPSHOT) -> Path:
    repo_root = repo_root.resolve()
    output_root = repo_root / EXPERIMENT_RELATIVE_PATH
    snapshot_data = json.loads(fingerprint_snapshot.read_text(encoding="utf-8"))
    start_mismatches = verify_starting_fingerprints(repo_root, snapshot_data)
    if start_mismatches:
        raise RuntimeError("Repository sources/artifacts drifted before analysis: " + "; ".join(start_mismatches))

    split_path = repo_root / "data/processed/face_split_manifest.csv"
    clean_path = repo_root / "data/processed/face_clean_manifest.csv"
    prediction_path = repo_root / "data/processed/experiments/face_trial_003_mobilenetv2_balanced_160_predictions.csv"
    face_evidence_path = repo_root / "reports/evaluation/phase4/final_complete/face.json"
    split_hash = sha256_file(split_path)
    registry_data = json.loads((repo_root / "config/baseline_registry.json").read_text(encoding="utf-8"))
    component = registry_data["components"]["face"]
    expected_model_hash = str(component["sha256"]).lower()
    model_path = repo_root / component["path"]
    model_hash_before = sha256_file(model_path)
    if model_hash_before.lower() != expected_model_hash:
        raise RuntimeError("Selected Trial 003 model hash does not match baseline registry.")
    if float(component["thresholds"]["binary_decision"]) != CLASSIFIER_THRESHOLD:
        raise RuntimeError("Selected face classifier threshold is not exactly 0.5.")
    manifest_expected_hash = str(registry_data["manifests"]["face_split"]["sha256"]).lower()
    if split_hash.lower() != manifest_expected_hash:
        raise RuntimeError("Canonical face split manifest hash does not match the registry.")

    split_manifest = pd.read_csv(split_path)
    clean_manifest = pd.read_csv(clean_path)
    if "file_hash" not in clean_manifest or clean_manifest["file_hash"].duplicated().any():
        raise ValueError("Clean face manifest has missing or duplicate stable sample identities.")
    if set(clean_manifest["file_hash"].astype(str)) != set(split_manifest["file_hash"].astype(str)):
        raise ValueError("Clean and split face manifests do not contain the same sample identities.")
    if clean_manifest.set_index("file_hash")["class_label"].sort_index().to_dict() != split_manifest.set_index("file_hash")["class_label"].sort_index().to_dict():
        raise ValueError("Clean and split face manifests disagree on dataset labels.")

    test_samples = canonical_test_samples(split_manifest)
    assert_test_only_samples(test_samples, split_manifest)
    historical_face = json.loads(face_evidence_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path)
    saved_reuse_check = saved_prediction_reuse_check(
        test_samples, predictions, historical_face["direct"]
    )
    gate_rows, gate_runtime = run_current_quality_gate(repo_root, test_samples)
    gate_rows = attach_historical_rejection_measurements(repo_root, test_samples, gate_rows)
    aligned_gate = align_gate_outcomes(test_samples, gate_rows)
    candidate_scored = align_prediction_scores(test_samples, predictions)
    candidate_scored_gate = align_gate_outcomes(candidate_scored, aligned_gate[gate_rows.columns])
    candidate_direct = subset_performance(
        candidate_scored_gate,
        "full held-out test set",
        np.ones(len(candidate_scored_gate), dtype=bool),
    )
    candidate_accepted = subset_performance(
        candidate_scored_gate,
        "gate-accepted subset",
        candidate_scored_gate["gate_accepted"].to_numpy(dtype=bool),
    )
    saved_full_metric_check = _assert_historical_metrics(
        candidate_direct, candidate_accepted, historical_face
    )
    prediction_artifact_reused = bool(saved_reuse_check["reusable"] and saved_full_metric_check["passed"])
    if prediction_artifact_reused:
        scored = candidate_scored
        score_source = "data/processed/experiments/face_trial_003_mobilenetv2_balanced_160_predictions.csv"
        fallback_reason = None
    else:
        predictions = infer_frozen_model_scores(repo_root, test_samples, expected_model_hash)
        scored = align_prediction_scores(test_samples, predictions)
        score_source = "one frozen-model inference pass using the Phase 4 preprocessing contract"
        fallback_reason = "Persisted score CSV failed direct and/or accepted-subset Phase 4 metric reproduction."
    scored_gate = align_gate_outcomes(scored, aligned_gate[gate_rows.columns])
    if len(scored_gate) != len(test_samples):
        raise AssertionError("Aligned score/gate analysis row count is not canonical test N.")

    # Keep canonical split fields and exact evidence paths in the joined table.
    relative_ids = {
        str(row.sample_id): str(row.relative_file_identity)
        for row in aligned_gate.itertuples(index=False)
    }
    scored_gate["relative_file_identity"] = scored_gate["sample_id"].map(relative_ids)
    scored_gate["gate_accepted"] = scored_gate["gate_accepted"].astype(bool)
    scored_gate["raw_mobilenetv2_score"] = scored_gate["score"].astype(float)
    scored_gate["prediction_at_0_5"] = scored_gate["prediction"].astype(int)
    scored_gate["dataset_label"] = scored_gate["class_label"]
    if scored_gate["sample_id"].duplicated().any() or scored_gate["score"].isna().any():
        raise AssertionError("Aligned per-sample analysis contains duplicate IDs or missing scores.")
    validate_score_vector(scored_gate["raw_mobilenetv2_score"])
    if int(scored_gate["gate_accepted"].sum()) + int((~scored_gate["gate_accepted"]).sum()) != len(scored_gate):
        raise AssertionError("Accepted plus rejected does not equal the canonical test sample count.")

    source_field = explicit_source_field(split_manifest.columns)
    direct = subset_performance(scored_gate, "full held-out test set", np.ones(len(scored_gate), dtype=bool))
    accepted_mask = scored_gate["gate_accepted"].to_numpy(dtype=bool)
    rejected_mask = ~accepted_mask
    accepted = subset_performance(scored_gate, "gate-accepted subset", accepted_mask)
    rejected = subset_performance(
        scored_gate,
        "raw classifier behavior on gate-rejected samples; not operational face-branch performance",
        rejected_mask,
    )
    historical_checks = _assert_historical_metrics(direct, accepted, historical_face)
    historical_accepted = int(historical_face["adapter_coverage"]["accepted"])
    historical_rejected = verify_historical_rejections(repo_root, test_samples, aligned_gate)
    if int(accepted["n"]) != historical_accepted or not historical_checks["passed"] or not historical_rejected["matches"]:
        raise RuntimeError(
            "Canonical evidence drift detected; analytical interpretation stopped. "
            f"metric_checks={historical_checks['failures']}; rejected_identity_reasons={historical_rejected}"
        )

    coverage_summary, coverage_by_class, contingency = coverage_tables(scored_gate)
    reason_summary, reason_combinations = rejection_reason_tables(scored_gate)
    outcome_table = coverage_aware_outcomes(scored_gate)
    score_summary = score_distribution_summary(scored_gate)
    performance_rows = [direct, accepted, rejected]
    subset_table = pd.DataFrame(performance_rows)
    bootstrap_rows: list[dict[str, Any]] = []
    for name, mask in (
        ("full held-out test set", np.ones(len(scored_gate), dtype=bool)),
        ("gate-accepted subset", accepted_mask),
        ("gate-rejected raw classifier behavior; descriptive only", rejected_mask),
    ):
        selected = scored_gate.loc[mask]
        for interval in stratified_bootstrap_intervals(
            selected["y_true"], selected["score"], iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED
        ):
            bootstrap_rows.append({"subset": name, **interval})
    bootstrap_table = pd.DataFrame(bootstrap_rows)

    historical_coverage = json.loads((repo_root / "reports/evaluation/phase4/final_complete/face_coverage.json").read_text(encoding="utf-8"))
    current_coverage = coverage_summary.iloc[0]
    if int(current_coverage["accepted"]) != int(historical_coverage["accepted"]) or int(current_coverage["rejected"]) != int(historical_coverage["rejected"]):
        raise RuntimeError("Replayed quality gate counts disagree with persisted Phase 4 coverage evidence.")
    quality_config = gate_runtime["quality_config"]
    model_hash_after = sha256_file(model_path)
    end_mismatches = verify_starting_fingerprints(repo_root, snapshot_data)
    if end_mismatches or model_hash_after != model_hash_before:
        raise RuntimeError("A frozen source/artifact hash changed during analysis.")
    if split_hash != sha256_file(split_path):
        raise RuntimeError("Canonical train/validation/test split manifest changed during analysis.")

    sample_columns = [
        "sample_id",
        "relative_file_identity",
        "file_hash",
        "class_label",
        "dataset_label",
        "y_true",
        "raw_mobilenetv2_score",
        "prediction_at_0_5",
        "gate_accepted",
        "quality_status",
        "quality_reasons",
        "quality_findings",
        "quality_measurements",
    ]
    sample_results = scored_gate[sample_columns].copy()
    source_files = [
        "data/processed/face_clean_manifest.csv",
        "data/processed/face_split_manifest.csv",
        "data/processed/experiments/face_trial_003_mobilenetv2_balanced_160_predictions.csv",
        "models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras",
        "config/baseline_registry.json",
        "reports/evaluation/phase4/final_complete/face.json",
        "reports/evaluation/phase4/final_complete/face_coverage.json",
        "reports/evaluation/phase4/final_complete/face_rejections.csv",
    ]
    source_manifest = {
        "canonical_clean_manifest": "data/processed/face_clean_manifest.csv",
        "canonical_split_manifest": "data/processed/face_split_manifest.csv",
        "selected_face_artifact": component["path"],
        "selected_face_artifact_sha256": model_hash_before,
        "selected_face_model_id": "Trial 003 MobileNetV2 balanced 160",
        "raw_score_source": score_source,
        "raw_score_source_sha256": model_hash_before if not prediction_artifact_reused else sha256_file(prediction_path),
        "persisted_prediction_csv_sha256": sha256_file(prediction_path),
        "persisted_prediction_csv_reused": prediction_artifact_reused,
        "persisted_prediction_metric_reuse_check": {
            "direct_only": saved_reuse_check,
            "direct_and_gate_accepted": saved_full_metric_check,
            "fallback_reason": fallback_reason,
        },
        "phase4_direct_and_gate_evidence": "reports/evaluation/phase4/final_complete/face.json",
        "phase4_per_sample_rejected_gate_evidence": "reports/evaluation/phase4/final_complete/face_rejections.csv",
        "source_field": source_field,
        "source_stratification_available": source_field is not None,
        "source_stratification_limitation": None
        if source_field
        else "The canonical manifest has no explicit reliable source/origin field; source groups were not inferred from file paths.",
        "quality_measurement_availability": "Image dimensions are in the canonical manifest; blur/luminance values are captured from current gate warning messages when present; historical Phase 4 face_count and face_size_px are identity-joined for rejected samples only. The current adapter does not expose those detector measurements for accepted samples.",
        "manifest_rows": int(len(split_manifest)),
        "test_n": int(len(test_samples)),
        "test_class_counts": {str(k): int(v) for k, v in test_samples["class_label"].value_counts().items()},
        "split_class_counts": {
            f"{split}/{label}": int(count)
            for (split, label), count in split_manifest.groupby(["split", "class_label"]).size().items()
        },
        "image_hashes_match_manifest": bool(gate_runtime["image_hashes_match_manifest"]),
        "source_artifact_hashes": {
            name: {"sha256": sha256_file(repo_root / name), "size_bytes": (repo_root / name).stat().st_size}
            for name in source_files
        },
    }
    analysis_config = {
        "analysis": "face quality-gate operational coverage; final held-out analysis",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": snapshot_data["head"],
        "repository_branch": snapshot_data["branch"],
        "canonical_test_split": "test",
        "positive_dataset_label": "Stroke",
        "classifier_threshold": CLASSIFIER_THRESHOLD,
        "metric_helper": "rural_stroke_assist.evaluation.metrics.classification_metrics",
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED, "method": "stratified sample bootstrap, resampling each class with replacement while preserving class counts", "confidence": 0.95},
        "coverage_interval": "95% Wilson score interval",
        "class_acceptance_test": "two-sided Fisher exact test; descriptive only",
        "prediction_artifact_reused": prediction_artifact_reused,
        "model_inference_executed": not prediction_artifact_reused,
        "prediction_fallback_reason": fallback_reason,
        "training_or_fine_tuning_executed": False,
        "gate_execution": "Current FaceAdapter and OpenCVFaceQualityAssessor; injected analysis-only inert runner; quality checks and image preprocessing are production implementations unchanged.",
        "quality_thresholds": quality_config,
        "preprocessing_contract": component["preprocessing_contract"],
        "direct_metric_tolerance_vs_phase4": METRIC_TOLERANCE,
        "rejected_subset_interpretation": "raw classifier behavior on gate-rejected samples; not operational face-branch performance",
        "dataset_label_limitation": "Research/proxy dataset labels, not prospective patient-level clinical ground truth.",
    }

    # Refuse any caller-selected output location outside the single approved tree.
    approved = (repo_root / EXPERIMENT_RELATIVE_PATH).resolve()
    if output_root.resolve() != approved:
        raise ValueError("Experiment output directory is outside the approved repository path.")
    figures_root = output_path(output_root, "figures")
    figures_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_path(output_root, "analysis_config.json"), analysis_config)
    _write_json(output_path(output_root, "source_manifest.json"), source_manifest)
    _write_csv(output_path(output_root, "sample_results.csv"), sample_results)
    _write_csv(output_path(output_root, "coverage_summary.csv"), coverage_summary)
    _write_csv(output_path(output_root, "coverage_by_class.csv"), coverage_by_class)
    if source_field:
        source_rows = []
        for (source, label), group in scored_gate.groupby([source_field, "class_label"], dropna=False):
            accepted_n = int(group["gate_accepted"].sum())
            row = _coverage_row(f"{source}/{label}", len(group), accepted_n)
            row.update({"source": source, "class_label": label})
            source_rows.append(row)
        _write_csv(output_path(output_root, "coverage_by_source.csv"), pd.DataFrame(source_rows))
    if not reason_summary.empty:
        _write_csv(output_path(output_root, "rejection_reason_summary.csv"), reason_summary)
        _write_csv(output_path(output_root, "rejection_reason_combinations.csv"), reason_combinations)
    _write_csv(output_path(output_root, "subset_performance.csv"), subset_table)
    _write_csv(output_path(output_root, "bootstrap_intervals.csv"), bootstrap_table)
    _write_json(output_path(output_root, "acceptance_contingency.json"), contingency)
    _write_csv(output_path(output_root, "coverage_aware_outcomes.csv"), outcome_table)
    _write_csv(output_path(output_root, "score_distribution_summary.csv"), score_summary)

    figure_paths = {
        "coverage_by_class": "figures/coverage_by_class.png",
        "score_distribution_by_gate": "figures/score_distribution_by_gate.png",
        "rejection_reasons_by_class": "figures/rejection_reasons_by_class.png"
        if not reason_summary.empty
        else "figures/accepted_rejected_by_class.png",
    }
    _plot_coverage(coverage_summary, coverage_by_class, output_path(output_root, figure_paths["coverage_by_class"]))
    _plot_score_distributions(scored_gate, output_path(output_root, figure_paths["score_distribution_by_gate"]))
    if not reason_summary.empty:
        _plot_rejection_reasons(reason_summary, output_path(output_root, figure_paths["rejection_reasons_by_class"]))
    else:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        labels = ["NonStroke", "Stroke"]
        accepted_counts = [int(contingency["class_counts"][label]["accepted"]) for label in labels]
        rejected_counts = [int(contingency["class_counts"][label]["rejected"]) for label in labels]
        ax.bar(labels, accepted_counts, label="Accepted", color="#6c9a8b")
        ax.bar(labels, rejected_counts, bottom=accepted_counts, label="Rejected / unavailable", color="#bf6b63")
        ax.set_ylabel("Held-out test samples")
        ax.set_title("Accepted and rejected face inputs by dataset class")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_path(output_root, figure_paths["rejection_reasons_by_class"]), dpi=300, bbox_inches="tight")
        plt.close(fig)

    runtime_files = snapshot_data["files"]
    runtime_paths = [
        "rural_stroke_assist/inference/face_adapter.py",
        "rural_stroke_assist/inference/runners.py",
        "rural_stroke_assist/quality/face_quality.py",
        "rural_stroke_assist/modeling/face_inference.py",
        "rural_stroke_assist/assessment/factory.py",
        "rural_stroke_assist/assessment/service.py",
        "rural_stroke_assist/modules/fusion_module.py",
        "rural_stroke_assist/ui/common.py",
        "apps/collector_app.py",
        "apps/clinician_app.py",
    ]
    direct_coverage_count = int(coverage_summary.iloc[0]["accepted"])
    invariants = {
        "passed": True,
        "checks": {
            "canonical_test_n_matches_manifest": {"passed": len(test_samples) == int((split_manifest["split"] == "test").sum()), "test_n": len(test_samples)},
            "canonical_class_counts_match_manifest": {"passed": test_samples["class_label"].value_counts().to_dict() == split_manifest.loc[split_manifest["split"] == "test", "class_label"].value_counts().to_dict()},
            "no_duplicate_test_sample_ids": {"passed": bool(test_samples["sample_id"].is_unique)},
            "one_raw_score_per_test_sample": {"passed": len(scored_gate["score"]) == len(test_samples) and not scored_gate["score"].isna().any()},
            "one_quality_gate_outcome_per_test_sample": {"passed": len(aligned_gate) == len(test_samples) and not aligned_gate["gate_accepted"].isna().any()},
            "accepted_plus_rejected_equals_total": {"passed": direct_coverage_count + int(coverage_summary.iloc[0]["rejected"]) == len(test_samples)},
            "classifier_threshold_exactly_0_5": {"passed": CLASSIFIER_THRESHOLD == 0.5 and float(component["thresholds"]["binary_decision"]) == 0.5},
            "quality_thresholds_unchanged_from_frozen_source": {"passed": not end_mismatches and "quality_config" in gate_runtime, "effective_defaults": quality_config},
            "model_artifact_hash_unchanged": {"passed": model_hash_before == model_hash_after == expected_model_hash, "sha256": model_hash_after},
            "face_adapter_and_source_hashes_unchanged": {"passed": not end_mismatches, "fingerprinted_source_file_count": len(runtime_files)},
            "no_training_or_fine_tuning": {"passed": True, "training_or_fine_tuning_executed": False, "prediction_artifact_reused": prediction_artifact_reused, "model_inference_executed": not prediction_artifact_reused},
            "validation_and_train_splits_excluded": {"passed": True, "test_rows_only": True},
            "no_application_runtime_file_modified": {"passed": all(runtime_files_path in runtime_files for runtime_files_path in runtime_paths) and not end_mismatches},
            "phase4_direct_and_accepted_metrics_reproduced": historical_checks,
            "phase4_quality_gate_rejections_reproduced": historical_rejected,
            "quality_gate_accepted_count_matches_phase4": {"passed": direct_coverage_count == historical_accepted, "accepted": direct_coverage_count, "phase4_accepted": historical_accepted},
            "image_hashes_match_canonical_manifest": {"passed": bool(gate_runtime["image_hashes_match_manifest"])},
            "source_dataset_field_explicit_and_reliable": {"passed": True, "available": source_field is not None, "field": source_field},
        },
        "notes": [
            "Face dataset labels are research/proxy labels, not prospective patient-level clinical ground truth.",
            "Rejected-subset classifier metrics describe raw frozen scores only and are not operational face-branch performance.",
            "No source group was inferred from file names or paths.",
        ],
    }
    check_failures = [name for name, check in invariants["checks"].items() if not check.get("passed", False)]
    invariants["passed"] = not check_failures
    invariants["failed_checks"] = check_failures
    if check_failures:
        raise AssertionError("Analysis invariants failed: " + ", ".join(check_failures))
    _write_json(output_path(output_root, "invariants.json"), invariants)

    generated = sorted(path for path in output_root.rglob("*") if path.is_file() and path.name != "artifact_hashes.json")
    generated_hashes = {
        path.relative_to(output_root).as_posix(): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in generated
    }
    _write_json(output_path(output_root, "artifact_hashes.json"), {"hash_algorithm": "SHA-256", "files": generated_hashes})
    return output_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fingerprint-snapshot", type=Path, default=DEFAULT_FINGERPRINT_SNAPSHOT)
    args = parser.parse_args(argv)
    try:
        output = run_analysis(args.repo_root, args.fingerprint_snapshot)
    except Exception as exc:
        print(f"face quality-gate analysis failed: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
