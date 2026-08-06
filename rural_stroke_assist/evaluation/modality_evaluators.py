"""Manifest-driven evaluators for the three learned branches.

These functions intentionally produce engineering evidence only.  They never
write model artifacts and they keep rejected adapter inputs out of classifier
confusion matrices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np
import pandas as pd

from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.metadata_adapter import MetadataAdapter, MetadataInput
from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
from rural_stroke_assist.inference.registry import BaselineRegistry
from rural_stroke_assist.modeling.face_inference import preprocess_face_image
from .bootstrap import group_bootstrap_ci, stratified_bootstrap_ci
from .contracts import EvaluationArtifact
from .metrics import classification_metrics, metric_dict


def _limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    return frame if not limit else frame.head(limit)


def evaluate_face(manifest: str | Path, registry: BaselineRegistry, *, split: str = "test", limit: int | None = None, bootstrap_iterations: int = 1000, seed: int = 42, output_dir: Path | None = None) -> dict[str, Any]:
    frame = _limit(pd.read_csv(manifest).query("split == @split"), limit)
    import tensorflow as tf
    model = tf.keras.models.load_model(registry.path_for("face"))
    y, scores, direct_ms = [], [], []
    for row in frame.itertuples(index=False):
        started = time.perf_counter_ns()
        scores.append(float(np.asarray(model.predict(preprocess_face_image(row.path, image_size=(160, 160)), verbose=0)).reshape(-1)[0]))
        direct_ms.append((time.perf_counter_ns() - started) / 1e6)
        y.append(int(str(row.class_label).lower() == "stroke"))
    summary = classification_metrics(y, scores)
    adapter = FaceAdapter(registry=registry)
    accepted_y, accepted_scores, reasons, states = [], [], [], []
    for row, target in zip(frame.itertuples(index=False), y):
        evidence = adapter.infer(row.path)
        states.append(evidence.quality_status.value)
        if evidence.available and evidence.score is not None:
            accepted_y.append(target); accepted_scores.append(evidence.score)
        else:
            reasons.extend(f.code for f in evidence.quality_findings)
    adapter_summary = metric_dict(classification_metrics(accepted_y, accepted_scores)) if accepted_y else None
    artifacts = _write_classification_plots(output_dir, "face", y, scores) if output_dir else []
    ci = stratified_bootstrap_ci(y, scores, lambda a, b: classification_metrics(a, b).roc_auc, iterations=bootstrap_iterations, seed=seed, metric_name="roc_auc")
    return {"total_held_out": len(frame), "processed_direct": len(y), "direct_positive_class": "Stroke", "direct": metric_dict(summary), "adapter_coverage": {"accepted": len(accepted_y), "rejected": len(y) - len(accepted_y), "coverage": len(accepted_y) / len(y) if y else 0, "quality_states": pd.Series(states).value_counts().to_dict(), "rejection_reasons": pd.Series(reasons).value_counts().to_dict()}, "adapter_aware": adapter_summary, "bootstrap": ci.__dict__, "mean_inference_ms": float(np.mean(direct_ms)), "artifacts": artifacts}


def _write_classification_plots(output_dir: Path, prefix: str, y: list[int], scores: list[float]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve
    y_array, score_array = np.asarray(y), np.asarray(scores)
    definitions = {
        "score_distribution.png": lambda: (plt.hist([score_array[y_array == k] for k in (0, 1)], label=["negative", "positive"], bins=15), plt.legend()),
        "roc.png": lambda: (plt.plot(*roc_curve(y_array, score_array)[:2], label="model"), plt.plot([0, 1], [0, 1], "--", label="chance"), plt.legend()),
        "pr.png": lambda: (plt.plot(*precision_recall_curve(y_array, score_array)[:2], label="model"), plt.axhline(float(np.mean(y_array)), linestyle="--", label="prevalence"), plt.legend()),
        "calibration.png": lambda: (plt.plot(*calibration_curve(y_array, score_array, n_bins=10), marker="o"), plt.plot([0, 1], [0, 1], "--")),
    }
    for filename, plot in definitions.items():
        plt.figure(); plot(); plt.title(f"{prefix.title()} {filename[:-4]}"); target = output_dir / f"{prefix}_{filename}"; plt.savefig(target, dpi=120, bbox_inches="tight"); plt.close(); paths.append(str(target))
    return paths


def _metadata_input(row: Any) -> MetadataInput:
    def clean(value: Any) -> Any:
        return None if pd.isna(value) else value
    values = {"age": clean(row.age), "hypertension": clean(row.hypertension), "heart_disease": clean(row.heart_disease), "avg_glucose_level": clean(row.avg_glucose_level), "bmi": clean(row.bmi), "gender": clean(row.gender), "ever_married": clean(row.ever_married), "work_type": clean(row.work_type), "residence_type": clean(row.Residence_type), "smoking_status": clean(row.smoking_status)}
    return MetadataInput.model_validate(values)


def evaluate_metadata(manifest: str | Path, registry: BaselineRegistry, *, split: str = "test", limit: int | None = None, bootstrap_iterations: int = 1000, seed: int = 42, output_dir: Path | None = None) -> dict[str, Any]:
    frame = _limit(pd.read_csv(manifest).query("split == @split"), limit)
    adapter = MetadataAdapter(registry=registry)
    y, scores, failures = [], [], []
    for row in frame.itertuples(index=False):
        y.append(int(row.stroke))
        try:
            evidence = adapter.infer(_metadata_input(row))
            if evidence.score is None: raise ValueError("metadata returned no score")
            scores.append(float(evidence.score))
        except Exception as exc:
            failures.append(type(exc).__name__)
    result = {"n": len(scores), "failures": pd.Series(failures).value_counts().to_dict(), "metrics": metric_dict(classification_metrics(y[:len(scores)], scores)) if scores else None, "artifacts": _write_classification_plots(output_dir, "metadata", y[:len(scores)], scores) if output_dir and scores else []}
    if scores:
        result["bootstrap"] = stratified_bootstrap_ci(y[:len(scores)], scores, lambda a, b: classification_metrics(a, b).roc_auc, iterations=bootstrap_iterations, seed=seed, metric_name="roc_auc").__dict__
    return result


def evaluate_speech(manifest: str | Path, registry: BaselineRegistry, *, split: str = "test", limit: int | None = None, bootstrap_iterations: int = 1000, seed: int = 42, output_dir: Path | None = None) -> dict[str, Any]:
    frame = _limit(pd.read_csv(manifest).query("split == @split"), limit)
    adapter = SpeechAdapter(registry=registry)
    y, scores, groups, failures, quality = [], [], [], [], []
    direct_y, direct_scores, direct_failures = [], [], []
    import joblib
    from rural_stroke_assist.features.speech_features import load_audio
    model = joblib.load(registry.path_for("speech")); positive_index = adapter._positive_class_index(model)
    for row in frame.itertuples(index=False):
        target = int(str(row.label).lower() == "dysarthric")
        try:
            signal, sample_rate = load_audio(row.path); features = adapter._extract_features(signal, sample_rate); direct_frame = pd.DataFrame([[features[name] for name in registry.component("speech").feature_columns]], columns=registry.component("speech").feature_columns); direct_y.append(target); direct_scores.append(float(np.asarray(model.predict_proba(direct_frame))[0, positive_index]))
        except Exception as exc:
            direct_failures.append(type(exc).__name__)
        try:
            evidence = adapter.infer(row.path)
            quality.append(evidence.quality_status.value)
            if evidence.available and evidence.score is not None:
                y.append(target); scores.append(float(evidence.score)); groups.append(str(row.speaker_id))
            else: failures.append("quality_rejected")
        except Exception as exc:
            failures.append(type(exc).__name__)
    result: dict[str, Any] = {"total_held_out": len(frame), "processed_direct": len(direct_scores), "direct_failures": pd.Series(direct_failures).value_counts().to_dict(), "direct": metric_dict(classification_metrics(direct_y, direct_scores)) if direct_scores else None, "accepted": len(scores), "rejected": len(frame) - len(scores), "coverage": len(scores) / len(frame) if len(frame) else 0, "adapter_failures": pd.Series(failures).value_counts().to_dict(), "quality_states": pd.Series(quality).value_counts().to_dict(), "adapter_aware": None, "artifacts": _write_classification_plots(output_dir, "speech", direct_y, direct_scores) if output_dir and direct_scores else []}
    if scores:
        result["adapter_aware"] = metric_dict(classification_metrics(y, scores))
        result["speaker_bootstrap"] = group_bootstrap_ci(groups, y, scores, lambda a, b: classification_metrics(a, b).roc_auc, iterations=bootstrap_iterations, seed=seed, metric_name="roc_auc").__dict__
    return result
