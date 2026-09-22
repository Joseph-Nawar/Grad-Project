"""Standalone metadata pretrained-foundation-model experiment.

This module is intentionally isolated from production inference.  The staged
runner loads only requested manifest partitions and keeps test access behind
an immutable validation-evidence gate.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from rural_stroke_assist.evaluation.bootstrap import stratified_bootstrap_ci
from rural_stroke_assist.evaluation.metrics import classification_metrics, metric_dict
from rural_stroke_assist.preprocessing.metadata import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "metadata_pretrained_foundation_trial_001"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "experiments" / EXPERIMENT_ID
DEFAULT_CACHE_DIR = ROOT / "data" / "processed" / "experiments" / EXPERIMENT_ID
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "metadata_split_manifest.csv"
CANONICAL_BASELINE = ROOT / "models" / "experiments" / "metadata" / "mvp_metadata_risk_model.pkl"
HISTORICAL_PHASE4 = ROOT / "reports" / "evaluation" / "phase4" / "final_complete" / "metadata_context.json"
EXPECTED_MANIFEST_SHA256 = "750110fff761405c194b3d149918f0d49eb643baea6129f39c8e02c3900108cb"
EXPECTED_SPLIT_COUNTS = {"train": 3577, "val": 766, "test": 767}
EXPECTED_POSITIVE_COUNTS = {"train": 174, "val": 37, "test": 38}
CATEGORICAL_FEATURE_INDICES = [FEATURE_COLUMNS.index(name) for name in CATEGORICAL_FEATURES]
CANDIDATE_IDS = ("tabpfn_v2", "tabicl_v2")
TARGET_THRESHOLD = 0.5
SEED = 42
BOOTSTRAP_ITERATIONS = 1000
SELECTION_MARGIN = 0.01
TABPFN_PACKAGE_VERSION = "9.0.0"
TABICL_PACKAGE_VERSION = "2.2.0"
TABICL_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def stable_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(stable_json_bytes(payload))
    temporary.replace(path)


def load_requested_splits(path: Path, requested_splits: tuple[str, ...]) -> pd.DataFrame:
    """Stream only requested rows from the committed split manifest."""
    requested = set(requested_splits)
    if not requested:
        raise ValueError("At least one split must be requested")
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError("Manifest must contain a split column")
        for raw in reader:
            if raw.get("split") not in requested:
                continue
            row: dict[str, object] = dict(raw)
            for column in NUMERIC_FEATURES:
                if column in row:
                    row[column] = float(row[column]) if row[column] not in (None, "") else np.nan
            if TARGET_COLUMN in row and row[TARGET_COLUMN] not in (None, ""):
                row[TARGET_COLUMN] = int(float(row[TARGET_COLUMN]))
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and not set(frame["split"].dropna()).issubset(requested):
        raise ValueError("Loader returned an unrequested split")
    return frame


def validate_manifest_frame(frame: pd.DataFrame, requested_splits: tuple[str, ...]) -> None:
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, "split"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    observed = set(frame["split"].dropna())
    if not observed.issubset(set(requested_splits)):
        raise ValueError(f"Frame contains unrequested splits: {sorted(observed - set(requested_splits))}")
    if not set(frame[TARGET_COLUMN].dropna().astype(int)).issubset({0, 1}):
        raise ValueError("Target must be binary")


def prepare_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve raw numeric/categorical contract types for native candidate APIs."""
    result = frame.loc[:, FEATURE_COLUMNS].copy()
    for name in NUMERIC_FEATURES:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    for name in CATEGORICAL_FEATURES:
        result[name] = result[name].astype(object)
    return result


def calculate_binary_metrics(
    y_true: Iterable[int], scores: Iterable[float], threshold: float = TARGET_THRESHOLD,
) -> dict[str, object]:
    """Adapt the project's canonical metrics and add missing threshold metrics."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(scores), dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Binary labels and scores must have equal non-zero length")
    summary = metric_dict(classification_metrics(y, p, threshold=threshold))
    predictions = (p >= threshold).astype(int)
    summary["accuracy"] = float(accuracy_score(y, predictions))
    summary["balanced_accuracy"] = float(balanced_accuracy_score(y, predictions))
    summary["threshold"] = float(threshold)
    summary["positive_prevalence"] = float(np.mean(y))
    return summary


def _metric_value(name: str, y: Sequence[int], p: Sequence[float], threshold: float) -> float | None:
    if name in {"accuracy", "balanced_accuracy"}:
        return calculate_binary_metrics(y, p, threshold=threshold)[name]  # type: ignore[return-value]
    return metric_dict(classification_metrics(y, p, threshold=threshold)).get(name)  # type: ignore[return-value]


def stratified_bootstrap_metrics(
    y_true: Iterable[int], scores: Iterable[float], *, iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED, threshold: float = TARGET_THRESHOLD,
) -> dict[str, dict[str, object]]:
    metrics = ("roc_auc", "pr_auc", "brier", "calibration_error", "accuracy", "balanced_accuracy")
    output: dict[str, dict[str, object]] = {}
    for name in metrics:
        interval = stratified_bootstrap_ci(
            list(y_true), list(scores),
            metric=lambda y, p, metric_name=name: _metric_value(metric_name, y, p, threshold),
            iterations=iterations, seed=seed, metric_name=name,
        )
        output[name] = {
            "metric": interval.metric,
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "iterations": interval.iterations,
            "seed": interval.seed,
            "unit": interval.unit,
        }
    return output


def build_resource_protocol_payload(device: str) -> dict[str, object]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    return {
        "device": device,
        "validation_sample_rule": "first_128_validation_rows_in_manifest_order",
        "validation_sample_size": 128,
        "validation_sample_positions": list(range(128)),
        "warmup_runs": 2,
        "timed_repetitions": 5,
        "preprocessing_included": "candidate_native_inside_predict_proba_only",
        "model_load_and_context_fit_excluded": True,
        "p50_method": "numpy.median_of_five_wall_clock_milliseconds",
    }


def build_protocol_payload(device: str = "cpu") -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": "2026-09-22.v1",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_feature_indices": CATEGORICAL_FEATURE_INDICES,
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "expected_positive_counts": EXPECTED_POSITIVE_COUNTS,
        "threshold": TARGET_THRESHOLD,
        "selection_margin": SELECTION_MARGIN,
        "selection_rule": "inclusive_pr_auc_margin_then_checkpoint_size_then_p50_latency_then_peak_rss_then_candidate_id",
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": SEED, "unit": "sample-stratified"},
        "resource_protocol": build_resource_protocol_payload(device),
        "test_is_unavailable_during_validate_and_select": True,
        "candidate_ids": list(CANDIDATE_IDS),
        "candidate_declarations": {
            "tabpfn_v2": {"package": "tabpfn", "version": TABPFN_PACKAGE_VERSION, "model_version": "v2"},
            "tabicl_v2": {"package": "tabicl", "version": TABICL_PACKAGE_VERSION, "checkpoint": TABICL_CHECKPOINT},
        },
    }


def build_tabpfn_v2(device: str) -> Any:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    overrides: dict[str, object] = {
        "categorical_features_indices": CATEGORICAL_FEATURE_INDICES,
        "device": device,
        "random_state": SEED,
    }
    if device == "cpu":
        overrides["ignore_pretraining_limits"] = True
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, **overrides)


def build_tabicl_v2(device: str) -> Any:
    from tabicl import TabICLClassifier

    return TabICLClassifier(
        checkpoint_version=TABICL_CHECKPOINT,
        device=device,
        random_state=SEED,
        verbose=False,
    )


def positive_class_scores(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("Candidate model did not expose exactly one positive class")
    return probabilities[:, int(matches[0])]


def candidate_provenance(model: Any, candidate_id: str, device: str) -> dict[str, object]:
    package = "tabpfn" if candidate_id == "tabpfn_v2" else "tabicl"
    metadata: dict[str, object] = {
        "candidate_id": candidate_id,
        "package": package,
        "package_version": importlib.metadata.version(package),
        "device": device,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_feature_indices": CATEGORICAL_FEATURE_INDICES,
        "input_dtypes": {name: "numeric" for name in NUMERIC_FEATURES}
        | {name: "object/string" for name in CATEGORICAL_FEATURES},
        "model_settings": {},
        "model_weight_license": (
            "Prior Labs License (Apache 2.0 with attribution)"
            if candidate_id == "tabpfn_v2" else "Checkpoint license to be recorded from upstream artifact"
        ),
    }
    if candidate_id == "tabpfn_v2":
        metadata["model_id"] = "TabPFN v2"
        metadata["model_settings"] = {
            "model_version": "v2",
            "ignore_pretraining_limits": device == "cpu",
            "random_state": SEED,
        }
    else:
        metadata["model_id"] = "TabICLv2"
        metadata["checkpoint_version"] = TABICL_CHECKPOINT
        metadata["model_settings"] = {"checkpoint_version": TABICL_CHECKPOINT, "random_state": SEED, "verbose": False}
    for attribute in ("model_path", "checkpoint_version", "model_name"):
        value = getattr(model, attribute, None)
        if value is not None:
            metadata[attribute] = str(value)
    return metadata


def training_subset_fingerprint(frame: pd.DataFrame) -> dict[str, object]:
    ordered_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    if "split" in frame.columns:
        ordered_columns.append("split")
    ordered = frame.loc[:, ordered_columns].copy()
    rows = ordered.astype(object).where(pd.notna(ordered), None).to_dict(orient="records")
    payload = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "ordered_columns": ordered_columns,
        "dtypes": {name: str(ordered[name].dtype) for name in ordered_columns},
        "rows": rows,
    }
    return {
        "sha256": sha256_bytes(stable_json_bytes(payload)),
        "row_count": int(len(ordered)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "ordered_columns": ordered_columns,
        "dtypes": payload["dtypes"],
    }


def measure_candidate(
    model: Any,
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    X_eval: pd.DataFrame,
    candidate_id: str,
    resource_protocol: dict[str, object],
) -> dict[str, object]:
    import psutil

    process = psutil.Process(os.getpid())
    fit_start = time.perf_counter()
    model.fit(X_train, np.asarray(y_train, dtype=int))
    fit_seconds = time.perf_counter() - fit_start
    peak_rss = int(process.memory_info().rss)
    positions = list(resource_protocol["validation_sample_positions"])
    sample = X_eval.iloc[: min(len(X_eval), len(positions))]
    for _ in range(int(resource_protocol["warmup_runs"])):
        model.predict_proba(sample)
    durations_ms: list[float] = []
    for _ in range(int(resource_protocol["timed_repetitions"])):
        started = time.perf_counter()
        model.predict_proba(sample)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    whole_started = time.perf_counter()
    scores = positive_class_scores(model, X_eval)
    whole_seconds = time.perf_counter() - whole_started
    return {
        "candidate_id": candidate_id,
        "fit_seconds": fit_seconds,
        "validation_inference_seconds": whole_seconds,
        "validation_inference_p50_ms": float(np.median(durations_ms)),
        "validation_inference_repetitions_ms": durations_ms,
        "peak_rss_bytes": peak_rss,
        "resource_protocol": resource_protocol,
        "scores": scores.tolist(),
    }


def _write_predictions(path: Path, y_true: Sequence[int], scores: Sequence[float]) -> dict[str, object]:
    frame = pd.DataFrame({"y_true": list(y_true), "score": list(scores)})
    frame.to_csv(path, index=False)
    return {"path": path.name, "sha256": sha256_file(path), "row_count": int(len(frame))}


def _model_file_size(model: Any) -> int | None:
    candidates = []
    for attribute in ("model_path", "checkpoint_path"):
        value = getattr(model, attribute, None)
        if value:
            candidates.extend(value if isinstance(value, (list, tuple)) else [value])
    sizes = [Path(value).stat().st_size for value in candidates if Path(value).is_file()]
    return int(sum(sizes)) if sizes else None


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_resource_comparison(output_dir: Path, resources: list[dict[str, object]]) -> None:
    path = output_dir / "resource_comparison.csv"
    columns = ["candidate_id", "checkpoint_size_bytes", "validation_inference_p50_ms", "peak_rss_bytes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in resources:
            writer.writerow({column: item.get(column, "") for column in columns})


def _write_validation_comparison(output_dir: Path, results: list[dict[str, object]]) -> None:
    path = output_dir / "validation_comparison.csv"
    columns = ["candidate_id", "status", "validation_pr_auc", "validation_roc_auc", "n", "positive_prevalence"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            metrics = result.get("metrics", {})
            writer.writerow({
                "candidate_id": result.get("candidate_id"),
                "status": result.get("status"),
                "validation_pr_auc": metrics.get("pr_auc", "") if isinstance(metrics, dict) else "",
                "validation_roc_auc": metrics.get("roc_auc", "") if isinstance(metrics, dict) else "",
                "n": metrics.get("n", "") if isinstance(metrics, dict) else "",
                "positive_prevalence": metrics.get("positive_prevalence", "") if isinstance(metrics, dict) else "",
            })


def run_baseline_validation(frame: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    from joblib import load

    validation = frame.loc[frame["split"] == "val"]
    model = load(CANONICAL_BASELINE)
    X = prepare_candidate_frame(validation)
    y = validation[TARGET_COLUMN].astype(int).to_numpy()
    scores = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    metrics = calculate_binary_metrics(y, scores)
    prediction = _write_predictions(output_dir / "predictions_logistic_regression.csv", y, scores)
    return {
        "candidate_id": "logistic_regression",
        "status": "completed",
        "metrics": metrics,
        "bootstrap": stratified_bootstrap_metrics(y, scores),
        "prediction_artifact": prediction,
        "model_identity": {
            "model_id": "canonical_metadata_logistic_regression",
            "artifact_path": str(CANONICAL_BASELINE),
            "artifact_sha256": sha256_file(CANONICAL_BASELINE),
            "historical_test_source": str(HISTORICAL_PHASE4),
        },
    }


def run_candidate_validation(
    frame: pd.DataFrame,
    output_dir: Path,
    candidate_id: str,
) -> dict[str, object]:
    protocol = _read_json(output_dir / "protocol.json")
    device = str(protocol["resource_protocol"]["device"])
    train = frame.loc[frame["split"] == "train"]
    validation = frame.loc[frame["split"] == "val"]
    X_train = prepare_candidate_frame(train)
    X_validation = prepare_candidate_frame(validation)
    y_train = train[TARGET_COLUMN].astype(int).to_numpy()
    y_validation = validation[TARGET_COLUMN].astype(int).to_numpy()
    model = build_tabpfn_v2(device) if candidate_id == "tabpfn_v2" else build_tabicl_v2(device)
    provenance = candidate_provenance(model, candidate_id, device)
    measured = measure_candidate(
        model, X_train, y_train, X_validation, candidate_id, protocol["resource_protocol"],
    )
    scores = np.asarray(measured.pop("scores"), dtype=float)
    metrics = calculate_binary_metrics(y_validation, scores)
    prediction = _write_predictions(output_dir / f"predictions_{candidate_id}.csv", y_validation, scores)
    fingerprint = training_subset_fingerprint(train)
    checkpoint_size = _model_file_size(model)
    resource = {
        "candidate_id": candidate_id,
        "checkpoint_size_bytes": checkpoint_size if checkpoint_size is not None else "",
        "validation_inference_p50_ms": measured["validation_inference_p50_ms"],
        "peak_rss_bytes": measured["peak_rss_bytes"],
    }
    return {
        "candidate_id": candidate_id,
        "status": "completed",
        "metrics": metrics,
        "bootstrap": stratified_bootstrap_metrics(y_validation, scores),
        "prediction_artifact": prediction,
        "resource": resource,
        "provenance": provenance,
        "training_subset_fingerprint": fingerprint,
        "model_identity": provenance,
        "measured": measured,
    }


def _failure_payload(candidate_id: str, stage: str, error: Exception, output_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    failure = {
        "candidate_id": candidate_id,
        "stage": stage,
        "error_type": type(error).__name__,
        "sanitized_message": str(error).replace("\r", " ").replace("\n", " ")[:500],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "package_version": (
                importlib.metadata.version("tabpfn") if candidate_id == "tabpfn_v2"
                else importlib.metadata.version("tabicl")
            ),
        },
    }
    failure_path = output_dir / f"failure_{candidate_id}.json"
    write_json_atomic(failure_path, failure)
    terminal = {
        "candidate_id": candidate_id,
        "status": "failed",
        "model_identity": {
            "model_id": candidate_id,
            "package": "tabpfn" if candidate_id == "tabpfn_v2" else "tabicl",
        },
        "failure_artifact": failure_path.name,
        "failure_artifact_sha256": sha256_file(failure_path),
    }
    return terminal, failure


def run_validation(
    output_dir: Path,
    manifest_path: Path,
    candidate_ids: tuple[str, ...] = CANDIDATE_IDS,
    *,
    device: str = "cpu",
    loader: Callable[[Path, tuple[str, ...]], pd.DataFrame] = load_requested_splits,
    baseline_runner: Callable[[pd.DataFrame, Path], dict[str, object]] = run_baseline_validation,
    candidate_runner: Callable[[pd.DataFrame, Path, str], dict[str, object]] = run_candidate_validation,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol_payload(device)
    actual_manifest_hash = sha256_file(manifest_path) if manifest_path.exists() else None
    if actual_manifest_hash is not None and actual_manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest hash does not match the committed experiment manifest")
    protocol["manifest_sha256"] = actual_manifest_hash or protocol["manifest_sha256"]
    write_json_atomic(output_dir / "protocol.json", protocol)
    (output_dir / "protocol.sha256").write_text(sha256_file(output_dir / "protocol.json") + "\n", encoding="utf-8")

    frame = loader(manifest_path, ("train", "val"))
    validate_manifest_frame(frame, ("train", "val"))
    baseline = baseline_runner(frame, output_dir)
    write_json_atomic(output_dir / "validation_logistic_regression.json", baseline)
    results: list[dict[str, object]] = [baseline]
    resources: list[dict[str, object]] = []
    for candidate_id in candidate_ids:
        terminal_path = output_dir / f"validation_{candidate_id}.json"
        if terminal_path.exists():
            raise ValueError(f"validation evidence already exists for {candidate_id}")
        try:
            result = candidate_runner(frame, output_dir, candidate_id)
            if result.get("status") != "completed":
                raise ValueError("candidate runner returned a non-completed result without failure evidence")
            terminal = dict(result)
        except Exception as error:
            terminal, _ = _failure_payload(candidate_id, "validation", error, output_dir)
        write_json_atomic(terminal_path, terminal)
        results.append(terminal)
        if terminal.get("status") == "completed" and isinstance(terminal.get("resource"), dict):
            resources.append(terminal["resource"])
    _write_validation_comparison(output_dir, results)
    _write_resource_comparison(output_dir, resources)
    write_json_atomic(output_dir / "validation_metrics.json", {item["candidate_id"]: item for item in results})
    return {"protocol": protocol, "results": results, "candidate_ids": list(candidate_ids), "test_loaded": False}


def _resource_key(resource: dict[str, object], candidate_id: str) -> tuple[float, float, float, str]:
    def value(name: str) -> float:
        raw = resource.get(name)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return math.inf
        return number if math.isfinite(number) else math.inf
    return (
        value("checkpoint_size_bytes"),
        value("validation_inference_p50_ms"),
        value("peak_rss_bytes"),
        candidate_id,
    )


def run_selection(output_dir: Path) -> dict[str, object]:
    selection_path = output_dir / "selection_frozen.json"
    if selection_path.exists():
        raise ValueError("selection is already frozen")
    if (output_dir / "test_lock.json").exists():
        raise ValueError("test stage has already run")
    protocol_path = output_dir / "protocol.json"
    resource_path = output_dir / "resource_comparison.csv"
    if not protocol_path.exists() or not resource_path.exists():
        raise ValueError("validation evidence is incomplete")
    protocol = _read_json(protocol_path)
    resources = {
        row["candidate_id"]: row
        for row in csv.DictReader(resource_path.open("r", newline="", encoding="utf-8"))
    }
    evidence: dict[str, dict[str, object]] = {}
    completed: list[dict[str, object]] = []
    for candidate_id in CANDIDATE_IDS:
        terminal_path = output_dir / f"validation_{candidate_id}.json"
        if not terminal_path.exists():
            raise ValueError(f"missing terminal validation evidence for {candidate_id}")
        terminal = _read_json(terminal_path)
        record: dict[str, object] = {
            "status": terminal.get("status"),
            "terminal_artifact": terminal_path.name,
            "terminal_artifact_sha256": sha256_file(terminal_path),
            "model_identity": terminal.get("model_identity", {}),
        }
        if terminal.get("status") == "completed":
            metrics = terminal.get("metrics", {})
            if not isinstance(metrics, dict) or metrics.get("pr_auc") is None:
                raise ValueError(f"completed candidate lacks validation PR-AUC: {candidate_id}")
            record["metrics"] = metrics
            record["prediction_artifact"] = terminal.get("prediction_artifact", {})
            record["resource"] = terminal.get("resource", resources.get(candidate_id, {}))
            completed.append({"candidate_id": candidate_id, "pr_auc": float(metrics["pr_auc"]), "resource": record["resource"]})
        elif terminal.get("status") == "failed":
            failure_name = terminal.get("failure_artifact")
            failure_path = output_dir / str(failure_name)
            if not failure_name or not failure_path.exists():
                raise ValueError(f"failed candidate lacks failure artifact: {candidate_id}")
            record["failure_artifact"] = str(failure_name)
            record["failure_artifact_sha256"] = sha256_file(failure_path)
        else:
            raise ValueError(f"invalid terminal candidate status: {candidate_id}")
        evidence[candidate_id] = record
    if not completed:
        raise ValueError("zero completed pretrained candidates")
    best_pr_auc = max(item["pr_auc"] for item in completed)
    margin = float(protocol.get("selection_margin", SELECTION_MARGIN))
    eligible = [
        item for item in completed
        if best_pr_auc - item["pr_auc"] <= margin or math.isclose(best_pr_auc - item["pr_auc"], margin, abs_tol=1e-12)
    ]
    selected = min(eligible, key=lambda item: _resource_key(item["resource"], item["candidate_id"]))
    snapshot = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": selected["candidate_id"],
        "test_used_for_selection": False,
        "test_evaluated": False,
        "selection_rule": {
            "primary_metric": "validation_pr_auc",
            "inclusive_margin": float(protocol.get("selection_margin", SELECTION_MARGIN)),
            "eligible_condition": "best_pr_auc - candidate_pr_auc <= margin",
            "resource_tie_break": ["checkpoint_size_bytes", "validation_inference_p50_ms", "peak_rss_bytes", "candidate_id"],
        },
        "manifest_sha256": protocol["manifest_sha256"],
        "protocol_sha256": sha256_file(protocol_path),
        "resource_comparison_sha256": sha256_file(resource_path),
        "validation_evidence": evidence,
    }
    write_json_atomic(selection_path, snapshot)
    (output_dir / "selection_frozen.sha256").write_text(sha256_file(selection_path) + "\n", encoding="utf-8")
    return snapshot


def verify_selection_evidence(output_dir: Path, manifest_path: Path | None = None) -> dict[str, object]:
    selection_path = output_dir / "selection_frozen.json"
    checksum_path = output_dir / "selection_frozen.sha256"
    if not selection_path.exists() or not checksum_path.exists():
        raise ValueError("selection is not frozen")
    if checksum_path.read_text(encoding="utf-8").strip() != sha256_file(selection_path):
        raise ValueError("selection snapshot hash mismatch")
    snapshot = _read_json(selection_path)
    protocol_path = output_dir / "protocol.json"
    if snapshot.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("protocol hash mismatch")
    resource_path = output_dir / "resource_comparison.csv"
    if snapshot.get("resource_comparison_sha256") != sha256_file(resource_path):
        raise ValueError("resource comparison hash mismatch")
    if manifest_path is not None and manifest_path.exists():
        if snapshot.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("split manifest hash mismatch")
    for candidate_id, record in snapshot["validation_evidence"].items():
        terminal_path = output_dir / str(record["terminal_artifact"])
        if record["terminal_artifact_sha256"] != sha256_file(terminal_path):
            raise ValueError("validation evidence hash mismatch")
        terminal = _read_json(terminal_path)
        if terminal.get("status") != record.get("status"):
            raise ValueError("validation evidence status mismatch")
        if record["status"] == "completed":
            raw_prediction = terminal.get("prediction_artifact", {})
            if isinstance(raw_prediction, str):
                prediction_path = output_dir / raw_prediction
                prediction_hash = terminal.get("prediction_artifact_sha256")
            else:
                prediction_path = output_dir / str(raw_prediction.get("path"))
                prediction_hash = raw_prediction.get("sha256")
            if prediction_hash != sha256_file(prediction_path):
                raise ValueError("prediction evidence hash mismatch")
        else:
            failure_path = output_dir / str(record["failure_artifact"])
            if record["failure_artifact_sha256"] != sha256_file(failure_path):
                raise ValueError("failure evidence hash mismatch")
    return snapshot


def run_test(output_dir: Path, manifest_path: Path, *, device: str = "cpu", loader: Callable[[Path, tuple[str, ...]], pd.DataFrame] = load_requested_splits) -> dict[str, object]:
    if (output_dir / "test_lock.json").exists():
        raise ValueError("test stage has already run")
    snapshot = verify_selection_evidence(output_dir, manifest_path)
    frame = loader(manifest_path, ("train", "test"))
    validate_manifest_frame(frame, ("train", "test"))
    selected_id = str(snapshot["candidate_id"])
    train = frame.loc[frame["split"] == "train"]
    test = frame.loc[frame["split"] == "test"]
    X_train = prepare_candidate_frame(train)
    X_test = prepare_candidate_frame(test)
    y_train = train[TARGET_COLUMN].astype(int).to_numpy()
    y_test = test[TARGET_COLUMN].astype(int).to_numpy()
    model = build_tabpfn_v2(device) if selected_id == "tabpfn_v2" else build_tabicl_v2(device)
    model.fit(X_train, y_train)
    scores = positive_class_scores(model, X_test)
    metrics = calculate_binary_metrics(y_test, scores)
    prediction = _write_predictions(output_dir / "predictions_selected_test.csv", y_test, scores)
    result = {
        "candidate_id": selected_id,
        "status": "completed",
        "metrics": metrics,
        "bootstrap": stratified_bootstrap_metrics(y_test, scores),
        "prediction_artifact": prediction,
        "selection_sha256": sha256_file(output_dir / "selection_frozen.json"),
        "baseline_test_comparison": "historical_phase4_only",
    }
    write_json_atomic(output_dir / "selected_test_metrics.json", result)
    write_json_atomic(output_dir / "test_lock.json", {
        "status": "completed",
        "candidate_id": selected_id,
        "selection_sha256": result["selection_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    return result
