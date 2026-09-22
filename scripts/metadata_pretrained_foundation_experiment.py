"""Standalone metadata pretrained-foundation-model experiment.

This module is intentionally isolated from production inference.  The staged
runner loads only requested manifest partitions and keeps test access behind
an immutable validation-evidence gate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_bootstrap_module = importlib.import_module("rural_stroke_assist.evaluation.bootstrap")
_metrics_module = importlib.import_module("rural_stroke_assist.evaluation.metrics")
_metadata_module = importlib.import_module("rural_stroke_assist.preprocessing.metadata")
stratified_bootstrap_ci = _bootstrap_module.stratified_bootstrap_ci
classification_metrics = _metrics_module.classification_metrics
metric_dict = _metrics_module.metric_dict
CATEGORICAL_FEATURES = _metadata_module.CATEGORICAL_FEATURES
FEATURE_COLUMNS = _metadata_module.FEATURE_COLUMNS
NUMERIC_FEATURES = _metadata_module.NUMERIC_FEATURES
TARGET_COLUMN = _metadata_module.TARGET_COLUMN

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
TABPFN_SOURCE_REPOSITORY = "https://huggingface.co/Prior-Labs/TabPFN-v2-clf"
TABPFN_LICENSE = "Prior Labs License (Apache 2.0 with additional attribution requirement)"
TABPFN_LICENSE_URL = "https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt"
TABICL_SOURCE_REPOSITORY = "https://huggingface.co/jingang/TabICL"
TABICL_LICENSE = "BSD-3-Clause (official TabICL model card/repository)"
TABICL_LICENSE_URL = "https://github.com/soda-inria/tabicl/blob/main/LICENSE"


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


def _model_paths(model: Any) -> list[Path]:
    paths: list[Path] = []
    for attribute in ("model_path_", "model_path", "checkpoint_path"):
        value = getattr(model, attribute, None)
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if item is not None:
                path = Path(str(item))
                if path.is_file() and path not in paths:
                    paths.append(path)
    return paths


def _checkpoint_revision(path: Path) -> str | None:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _configured_model_identity(candidate_id: str, device: str) -> dict[str, object]:
    if candidate_id == "tabpfn_v2":
        return {
            "model_id": "TabPFN v2 classifier",
            "package": "tabpfn",
            "package_version": TABPFN_PACKAGE_VERSION,
            "model_version": "ModelVersion.V2",
            "device": device,
            "ignore_pretraining_limits": device == "cpu",
            "source_repository": TABPFN_SOURCE_REPOSITORY,
            "model_weight_license": TABPFN_LICENSE,
            "model_weight_license_url": TABPFN_LICENSE_URL,
        }
    return {
        "model_id": "TabICLv2 classifier",
        "package": "tabicl",
        "package_version": TABICL_PACKAGE_VERSION,
        "checkpoint_version": TABICL_CHECKPOINT,
        "device": device,
        "source_repository": TABICL_SOURCE_REPOSITORY,
        "model_weight_license": TABICL_LICENSE,
        "model_weight_license_url": TABICL_LICENSE_URL,
    }


def candidate_provenance(model: Any, candidate_id: str, device: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "candidate_id": candidate_id,
        **_configured_model_identity(candidate_id, device),
        "package_version": importlib.metadata.version("tabpfn" if candidate_id == "tabpfn_v2" else "tabicl"),
        "device": device,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_feature_indices": CATEGORICAL_FEATURE_INDICES,
        "input_dtypes": {name: "numeric" for name in NUMERIC_FEATURES}
        | {name: "object/string" for name in CATEGORICAL_FEATURES},
        "model_settings": {},
    }
    if candidate_id == "tabpfn_v2":
        metadata["model_settings"] = {
            "model_version": "v2",
            "ignore_pretraining_limits": device == "cpu",
            "random_state": SEED,
        }
    else:
        metadata["model_settings"] = {"checkpoint_version": TABICL_CHECKPOINT, "random_state": SEED, "verbose": False}
    paths = _model_paths(model)
    metadata["checkpoint_files"] = [
        {
            "basename": path.name,
            "path_not_committed": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "cache_revision": _checkpoint_revision(path),
        }
        for path in paths
    ]
    metadata["checkpoint_revision"] = next(
        (revision for revision in (_checkpoint_revision(path) for path in paths) if revision),
        None,
    )
    for attribute in ("checkpoint_version", "model_name"):
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
        "context_fit_seconds": fit_seconds,
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
    sizes = [path.stat().st_size for path in _model_paths(model)]
    return int(sum(sizes)) if sizes else None


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_resource_comparison(output_dir: Path, resources: list[dict[str, object]]) -> None:
    path = output_dir / "resource_comparison.csv"
    columns = [
        "candidate_id", "checkpoint_size_bytes", "model_initialization_seconds",
        "context_fit_seconds", "validation_inference_seconds",
        "validation_inference_p50_ms", "peak_rss_bytes",
    ]
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
    initialization_started = time.perf_counter()
    model = build_tabpfn_v2(device) if candidate_id == "tabpfn_v2" else build_tabicl_v2(device)
    initialization_seconds = time.perf_counter() - initialization_started
    measured = measure_candidate(
        model, X_train, y_train, X_validation, candidate_id, protocol["resource_protocol"],
    )
    measured["model_initialization_seconds"] = initialization_seconds
    provenance = candidate_provenance(model, candidate_id, device)
    scores = np.asarray(measured.pop("scores"), dtype=float)
    metrics = calculate_binary_metrics(y_validation, scores)
    prediction = _write_predictions(output_dir / f"predictions_{candidate_id}.csv", y_validation, scores)
    fingerprint = training_subset_fingerprint(train)
    checkpoint_size = _model_file_size(model)
    resource = {
        "candidate_id": candidate_id,
        "checkpoint_size_bytes": checkpoint_size if checkpoint_size is not None else "",
        "model_initialization_seconds": initialization_seconds,
        "context_fit_seconds": measured["context_fit_seconds"],
        "validation_inference_seconds": measured["validation_inference_seconds"],
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


def _failure_payload(
    candidate_id: str,
    stage: str,
    error: Exception,
    output_dir: Path,
    device: str,
) -> tuple[dict[str, object], dict[str, object]]:
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
    failure["model_identity"] = _configured_model_identity(candidate_id, device)
    failure_path = output_dir / f"failure_{candidate_id}.json"
    write_json_atomic(failure_path, failure)
    terminal = {
        "candidate_id": candidate_id,
        "status": "failed",
        "model_identity": failure["model_identity"],
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
            terminal, _ = _failure_payload(candidate_id, "validation", error, output_dir, device)
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
            record["validation_metrics_artifact_sha256"] = sha256_file(terminal_path)
            record["prediction_artifact"] = terminal.get("prediction_artifact", {})
            prediction_artifact = terminal.get("prediction_artifact", {})
            if isinstance(prediction_artifact, dict):
                record["prediction_artifact_sha256"] = prediction_artifact.get("sha256")
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
        "candidate_terminal_artifact_hashes": {
            candidate_id: record["terminal_artifact_sha256"] for candidate_id, record in evidence.items()
        },
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


def _optional_json(path: Path) -> dict[str, object]:
    return _read_json(path) if path.exists() else {}


def write_plots(output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for prediction_path in sorted(output_dir.glob("predictions_*.csv")):
        data = pd.read_csv(prediction_path)
        if data["y_true"].nunique() < 2:
            continue
        y = data["y_true"].to_numpy(dtype=int)
        p = data["score"].to_numpy(dtype=float)
        label = prediction_path.stem.removeprefix("predictions_")
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        for suffix, x, series, title, xlabel, ylabel in (
            ("roc", fpr, tpr, "ROC curve", "False positive rate", "True positive rate"),
            ("pr", recall, precision, "Precision-recall curve", "Recall", "Precision"),
        ):
            figure, axis = plt.subplots(figsize=(5, 4))
            axis.plot(x, series)
            axis.set_title(f"{label} {title}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            figure.tight_layout()
            path = plot_dir / f"{label}_{suffix}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            generated.append(path.relative_to(output_dir).as_posix())
        bins = np.linspace(0.0, 1.0, 11)
        centers: list[float] = []
        observed: list[float] = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (p >= lower) & ((p < upper) if upper < 1 else (p <= upper))
            if mask.any():
                centers.append(float(p[mask].mean()))
                observed.append(float(y[mask].mean()))
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.plot(centers, observed, marker="o")
        axis.set_title(f"{label} calibration")
        axis.set_xlabel("Mean predicted score")
        axis.set_ylabel("Observed positive frequency")
        figure.tight_layout()
        path = plot_dir / f"{label}_calibration.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        generated.append(path.relative_to(output_dir).as_posix())
    return generated


def write_provenance_manifest(output_dir: Path) -> dict[str, object]:
    protocol = _optional_json(output_dir / "protocol.json")
    start_state = _optional_json(output_dir / "implementation_start_state.json")
    candidate_evidence = {}
    for candidate_id in CANDIDATE_IDS:
        path = output_dir / f"validation_{candidate_id}.json"
        if path.exists():
            candidate_evidence[candidate_id] = _read_json(path)
    package_versions = {}
    for package in ("tabpfn", "tabicl", "scikit-learn", "numpy", "pandas", "torch", "psutil", "matplotlib"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_start_state": start_state,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": package_versions,
            "pip_check_path": "pip_check.txt",
            "pip_freeze_path": "environment_freeze.txt",
            "pip_check": (output_dir / "pip_check.txt").read_text(encoding="utf-8") if (output_dir / "pip_check.txt").exists() else None,
        },
        "protocol_sha256": sha256_file(output_dir / "protocol.json") if (output_dir / "protocol.json").exists() else None,
        "protocol": protocol,
        "manifest_sha256": protocol.get("manifest_sha256"),
        "feature_contract": {
            "target_column": TARGET_COLUMN,
            "feature_columns": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "categorical_feature_indices": CATEGORICAL_FEATURE_INDICES,
        },
        "metric_implementation": {
            "classification": "rural_stroke_assist.evaluation.metrics.classification_metrics",
            "calibration": "rural_stroke_assist.evaluation.metrics.calibration_error",
            "bootstrap": "rural_stroke_assist.evaluation.bootstrap.stratified_bootstrap_ci",
        },
        "canonical_baseline": {
            "path": str(CANONICAL_BASELINE),
            "sha256": sha256_file(CANONICAL_BASELINE) if CANONICAL_BASELINE.exists() else None,
            "historical_test_source": str(HISTORICAL_PHASE4),
        },
        "candidate_terminal_evidence": candidate_evidence,
    }
    write_json_atomic(output_dir / "provenance_manifest.json", manifest)
    return manifest


def write_artifact_hash_manifest(output_dir: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_hashes.json":
            continue
        manifest[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    write_json_atomic(output_dir / "artifact_hashes.json", manifest)
    return manifest


def _format_metric_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(output_dir: Path) -> str:
    selection = _optional_json(output_dir / "selection_frozen.json")
    selected_test = _optional_json(output_dir / "selected_test_metrics.json")
    historical = _optional_json(HISTORICAL_PHASE4)
    historical_metrics = historical.get("metrics", {}) if isinstance(historical.get("metrics", {}), dict) else {}
    lines = [
        "# Metadata Pretrained Foundation Trial 001",
        "",
        "This is a research-only evaluation of contextual/background-risk evidence from structured metadata. It does not diagnose acute stroke, change fusion, or modify production inference.",
        "",
        "## Baseline",
        "",
        "The frozen Logistic Regression artifact was replayed on validation only. Its Phase 4 test metrics are shown only as historical/previously exposed context; this trial did not freshly load or evaluate Logistic Regression on test.",
        "",
        f"Historical Phase 4 baseline test ROC-AUC: {_format_metric_value(historical_metrics.get('roc_auc'))}; PR-AUC: {_format_metric_value(historical_metrics.get('pr_auc'))}; precision: {_format_metric_value(historical_metrics.get('precision'))}.",
        "",
        "## Pretrained candidates",
        "",
        "TabPFN v2 was explicitly selected through ModelVersion.V2 with canonical categorical positions and the CPU pretraining-limit override recorded when applicable. TabICLv2 used the pinned classification checkpoint `tabicl-classifier-v2-20260212.ckpt`. Neither candidate was fine-tuned, oversampled, or broadly searched.",
        "",
        "## Validation comparison",
        "",
        "Validation PR-AUC/average precision is primary because the positive class is rare. ROC-AUC can remain high while precision and PR-AUC remain modest because most thresholded positives can be false positives under severe imbalance.",
        "",
        "| Candidate | Status | Validation PR-AUC | Validation ROC-AUC | N | Positive prevalence |",
        "|---|---|---:|---:|---:|---:|",
    ]
    comparison_path = output_dir / "validation_comparison.csv"
    if comparison_path.exists():
        for row in csv.DictReader(comparison_path.open("r", newline="", encoding="utf-8")):
            lines.append("| {candidate_id} | {status} | {validation_pr_auc} | {validation_roc_auc} | {n} | {positive_prevalence} |".format(**{key: row.get(key, "") for key in ("candidate_id", "status", "validation_pr_auc", "validation_roc_auc", "n", "positive_prevalence")}))
    lines += [
        "",
        "## Validation decision",
        "",
        f"Selected pretrained candidate: `{selection.get('candidate_id', 'not frozen')}`. The decision used validation PR-AUC first, the inclusive 0.01 absolute near-tie margin, and the predeclared resource tie-break. Selection was frozen before test access.",
        "",
        "Terminal evidence for both pretrained candidates is hash-bound in `selection_frozen.json`; failed candidates remain explicit evidence rather than being silently omitted.",
        "",
        "## Final frozen test result",
        "",
        f"Selected candidate: `{selected_test.get('candidate_id', selection.get('candidate_id', 'not evaluated'))}`. Test evaluation was limited to the frozen selected pretrained candidate after evidence verification.",
        "",
    ]
    test_metrics = selected_test.get("metrics", {}) if isinstance(selected_test.get("metrics", {}), dict) else {}
    if test_metrics:
        lines.append("Test metrics: " + "; ".join(f"{key}={_format_metric_value(value)}" for key, value in test_metrics.items() if key not in {"confusion_matrix"}) + ".")
    else:
        lines.append("Selected-candidate test metrics were not present in this fixture/output directory.")
    lines += [
        "",
        "## Calibration and class-imbalance interpretation",
        "",
        "Calibration diagnostics are reported for score behavior only. A foundation-model score is not a calibrated acute stroke probability. Improved ranking, if observed, does not change the contextual-risk interpretation or establish clinical diagnostic performance.",
        "",
        "## Engineering trade-off",
        "",
        "Measured local checkpoint size, initialization/context-fit cost, fixed-batch latency, whole-validation latency, and peak RSS are separated from upstream/documented model characteristics. Desktop measurements do not establish smartphone feasibility.",
        "",
        "## Final model-role recommendation",
        "",
        f"Reference/research metadata model: `{selection.get('candidate_id', 'selected pretrained candidate')}` subject to the validation evidence and resource costs recorded here.",
        "Lightweight edge/deployment metadata model: retain the canonical Logistic Regression alternative unless a separate production approval changes that role. No production integration is performed by this experiment.",
        "",
        "## Limitations",
        "",
        "- Severe class imbalance and only a small number of positive examples make precision, PR-AUC, calibration, and bootstrap intervals uncertain.",
        "- The target and metadata branch are proxy/contextual risk evidence, not acute stroke diagnosis or a clinical probability.",
        "- There is no paired multimodal clinical validation and no clinical probability interpretation.",
        "- The historical Phase 4 baseline test metrics were previously exposed; the new trial uses them only as labeled context.",
        "- Runtime and memory measurements are local desktop/process observations, not deployment feasibility claims.",
        "",
    ]
    report = "\n".join(lines)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "select", "test"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.stage == "validate":
        run_validation(args.output_dir, args.manifest, device=args.device)
    elif args.stage == "select":
        run_selection(args.output_dir)
    else:
        run_test(args.output_dir, args.manifest, device=args.device)
        write_plots(args.output_dir)
        write_provenance_manifest(args.output_dir)
        write_report(args.output_dir)
        write_artifact_hash_manifest(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
