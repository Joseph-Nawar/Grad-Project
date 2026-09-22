"""Standalone metadata pretrained-foundation-model experiment.

This module is intentionally isolated from production inference.  The staged
runner loads only requested manifest partitions and keeps test access behind
an immutable validation-evidence gate.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
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
