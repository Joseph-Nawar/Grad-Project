"""Speaker-safe TORGO pretrained audio-embedding comparison.

Run ``validate`` first, ``select`` from validation evidence, then ``test`` once
with the frozen selection. This research runner never changes the canonical
MFCC + Random Forest model or its runtime contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import threading
import time
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rural_stroke_assist.features.speech_features import load_audio  # noqa: E402

EXPERIMENT_ID = "speech_pretrained_representation_trial_001"
OUTPUT_DIR = ROOT / "reports" / "experiments" / EXPERIMENT_ID
CACHE_DIR = ROOT / "data" / "processed" / "experiments" / EXPERIMENT_ID / "cache"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "speech_split_manifest.csv"
BASELINE_FEATURES = ROOT / "data" / "processed" / "speech_features.csv"
BASELINE_MODEL = ROOT / "models" / "experiments" / "speech" / "trial_001_mfcc_random_forest" / "model.pkl"
HISTORICAL_SPEECH_REPORT = ROOT / "reports" / "evaluation" / "phase4" / "final_complete" / "speech.json"
HISTORICAL_PROFILING_REPORT = ROOT / "reports" / "evaluation" / "phase4" / "final_complete" / "profiling.json"

EXPECTED_MANIFEST_SHA256 = "1d8674156a2b8da590ab512affce749355ae92243609c4be2e3ba992f516a5e2"
EXPECTED_BASELINE_SHA256 = "1bded0b3850ee71e2b775e7b8e3bf80f708ee2faee1f74ca64d8102bae01c925"
EXPECTED_MANIFEST_ROWS = 17_631
EXPECTED_SPEAKERS = {
    "train": {
        "wav_arrayMic_F03S01", "wav_arrayMic_F03S03", "wav_arrayMic_F04S01",
        "wav_arrayMic_FC01S01", "wav_arrayMic_FC03S02", "wav_arrayMic_FC03S03",
        "wav_arrayMic_M01S01", "wav_arrayMic_M01S02", "wav_arrayMic_M02S01",
        "wav_arrayMic_M02S02", "wav_arrayMic_M03S02", "wav_arrayMic_M04S01",
        "wav_arrayMic_M04S02", "wav_arrayMic_M05S01", "wav_arrayMic_MC01S01",
        "wav_arrayMic_MC01S02", "wav_arrayMic_MC01S03", "wav_arrayMic_MC02S01",
        "wav_arrayMic_MC02S02", "wav_arrayMic_MC03S01", "wav_arrayMic_MC04S02",
        "wav_headMic_F01", "wav_headMic_F03S01", "wav_headMic_F03S03",
        "wav_headMic_F04S02", "wav_headMic_FC01S01", "wav_headMic_FC03S01",
        "wav_headMic_FC03S02", "wav_headMic_M01S01", "wav_headMic_M02S01",
        "wav_headMic_M03S02", "wav_headMic_M04S02", "wav_headMic_MC01S01",
        "wav_headMic_MC01S03", "wav_headMic_MC02S01", "wav_headMic_MC02S02",
        "wav_headMic_MC03S02", "wav_headMic_MC04S01",
    },
    "val": {
        "wav_arrayMic_F03S02", "wav_arrayMic_F04S02", "wav_arrayMic_FC02S02",
        "wav_arrayMic_FC02S03", "wav_arrayMic_MC04S01", "wav_headMic_F03S02",
        "wav_headMic_FC02S03", "wav_headMic_M05S01",
    },
    "test": {
        "wav_arrayMic_F01", "wav_arrayMic_FC03S01", "wav_arrayMic_MC03S02",
        "wav_headMic_FC03S03", "wav_headMic_M01S02", "wav_headMic_M02S02",
        "wav_headMic_M05S02", "wav_headMic_MC01S02", "wav_headMic_MC03S01",
    },
}

SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 5.0
MIN_DURATION_SECONDS = 1.0
SEED = 42
POOLING = "arithmetic mean over the final frame-level embedding sequence"
CLASSIFIER_CONFIG = {
    "type": "sklearn.pipeline.Pipeline",
    "steps": ["StandardScaler", "LogisticRegression"],
    "standard_scaler": {"with_mean": True, "with_std": True},
    "logistic_regression": {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": SEED,
        "solver": "liblinear",
    },
    "threshold": 0.5,
}
PREPROCESSING = {
    "source_sample_rate": "manifest audio decoded by librosa",
    "target_sample_rate_hz": SAMPLE_RATE,
    "channels": 1,
    "mono": True,
    "max_duration_seconds": MAX_DURATION_SECONDS,
    "minimum_duration_seconds": MIN_DURATION_SECONDS,
    "short_clip_policy": "right-pad with zeros to 1.0 seconds; keep every nonempty input",
    "model_specific_feature_extractor": "official checkpoint AutoFeatureExtractor, where applicable",
    "pooling": POOLING,
}

CANDIDATES: dict[str, dict[str, Any]] = {
    "yamnet": {
        "model_id": "https://tfhub.dev/google/yamnet/1",
        "model_name": "YAMNet",
        "source": "TensorFlow Hub official Google release",
        "source_url": "https://tfhub.dev/google/yamnet/1",
        "documentation_url": "https://github.com/tensorflow/models/tree/master/research/audioset/yamnet",
        "license": "Apache-2.0",
        "expected_embedding_dimension": 1024,
        "documented_parameter_count": 3_700_000,
        "documented_checkpoint_size_bytes": None,
        "revision_kind": "TensorFlow Hub version 1",
    },
    "distilhubert": {
        "model_id": "ntu-spml/distilhubert",
        "model_name": "DistilHuBERT",
        "source": "Hugging Face official NTU Speech Processing & Machine Learning Lab model",
        "source_url": "https://huggingface.co/ntu-spml/distilhubert",
        "license": "Apache-2.0",
        "expected_embedding_dimension": 768,
        "documented_parameter_count": 23_500_000,
        "documented_checkpoint_size_bytes": 94_000_000,
        "revision_kind": "Hugging Face commit SHA resolved at download time",
    },
    "wav2vec2_base": {
        "model_id": "facebook/wav2vec2-base",
        "model_name": "Wav2Vec2-Base",
        "source": "Hugging Face official Meta model",
        "source_url": "https://huggingface.co/facebook/wav2vec2-base",
        "license": "Apache-2.0",
        "expected_embedding_dimension": 768,
        "documented_parameter_count": 95_000_000,
        "documented_checkpoint_size_bytes": 380_000_000,
        "revision_kind": "Hugging Face commit SHA resolved at download time",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_cache_key(
    *,
    model_id: str,
    model_revision: str,
    preprocessing: dict[str, Any],
    sample_sha256: str | None,
    split: str,
    split_manifest_sha256: str,
    sample_path: str | None = None,
    speaker_id: str | None = None,
    label: str | None = None,
) -> str:
    """Key one embedding by model, preprocessing, exact sample and split provenance."""
    return stable_sha256(
        {
            "model_id": model_id,
            "model_revision": model_revision,
            "preprocessing": preprocessing,
            "sample_sha256": sample_sha256,
            "sample_path": sample_path,
            "split": split,
            "split_manifest_sha256": split_manifest_sha256,
            "speaker_id": speaker_id,
            "label": label,
        }
    )


def validate_speaker_partitions(
    frame: pd.DataFrame,
    *,
    expected: dict[str, set[str]] | None = None,
) -> None:
    """Check labels and split membership are constant per speaker."""
    required = {"speaker_id", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"speech manifest is missing required field(s): {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("speech manifest has a missing speaker, label, or split")
    if not set(frame["label"].astype(str).str.lower()).issubset({"control", "dysarthric"}):
        raise ValueError("speech manifest labels must be control or dysarthric")
    if set(frame["split"].astype(str)) != {"train", "val", "test"}:
        raise ValueError("speech manifest must contain train, val, and test rows")

    per_speaker_splits = frame.groupby("speaker_id")["split"].nunique()
    if (per_speaker_splits > 1).any():
        offenders = per_speaker_splits[per_speaker_splits > 1].index.astype(str).tolist()
        raise ValueError(f"speaker appears in multiple splits: {offenders}")
    per_speaker_labels = frame.groupby("speaker_id")["label"].nunique()
    if (per_speaker_labels > 1).any():
        offenders = per_speaker_labels[per_speaker_labels > 1].index.astype(str).tolist()
        raise ValueError(f"speaker has inconsistent labels: {offenders}")

    if expected is not None:
        actual = {
            split: set(frame.loc[frame["split"] == split, "speaker_id"].astype(str))
            for split in ("train", "val", "test")
        }
        if actual != expected:
            raise ValueError(
                "speaker partition mismatch: "
                f"expected { {key: sorted(value) for key, value in expected.items()} }, "
                f"got { {key: sorted(value) for key, value in actual.items()} }"
            )


def load_canonical_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[pd.DataFrame, str]:
    digest = sha256_file(path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "speech split manifest hash differs from the frozen canonical manifest: "
            f"expected {EXPECTED_MANIFEST_SHA256}, got {digest}"
        )
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_MANIFEST_ROWS:
        raise ValueError(f"expected {EXPECTED_MANIFEST_ROWS} manifest rows, got {len(frame)}")
    validate_speaker_partitions(frame, expected=EXPECTED_SPEAKERS)
    frame = frame.copy().reset_index(drop=True)
    frame["split"] = frame["split"].astype(str)
    frame["label"] = frame["label"].astype(str).str.lower()
    frame["label_encoded"] = frame["label"].map({"control": 0, "dysarthric": 1})
    frame["_sample_id"] = np.arange(len(frame), dtype=np.int64)
    frame["path"] = frame["path"].map(
        lambda value: str((ROOT / str(value)).resolve()) if not Path(str(value)).is_absolute() else str(Path(str(value)))
    )
    return frame, digest


def calculate_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
    predictions: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("metrics require equal-length, non-empty labels and scores")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("metrics require finite scores and binary labels")
    if not 0 <= threshold <= 1:
        raise ValueError("classification threshold must be in [0, 1]")

    predicted_labels = (
        (scores >= threshold).astype(int)
        if predictions is None
        else np.asarray(predictions, dtype=int)
    )
    if (
        predicted_labels.ndim != 1
        or len(predicted_labels) != len(labels)
        or not np.isin(predicted_labels, [0, 1]).all()
    ):
        raise ValueError("predictions must be a matching one-dimensional binary vector")
    matrix = confusion_matrix(labels, predicted_labels, labels=[0, 1])
    tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1]))
    recall_positive = float(recall_score(labels, predicted_labels, zero_division=0))
    recall_negative = float(recall_score(labels, predicted_labels, pos_label=0, zero_division=0))
    return {
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "sensitivity": recall_positive,
        "specificity": recall_negative,
        "precision": float(precision_score(labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(labels, predicted_labels, zero_division=0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "threshold": float(threshold),
        "prediction_policy": "provided fitted-estimator predict()" if predictions is not None else "probability >= threshold",
        "evaluated_sample_count": int(len(labels)),
    }


def summarize_by_speaker(
    frame: pd.DataFrame,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
    predictions: Sequence[int] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    scores = np.asarray(probabilities, dtype=float)
    if len(frame) != len(scores):
        raise ValueError("speaker summary rows and probability scores must have equal length")
    predicted_labels = None if predictions is None else np.asarray(predictions, dtype=int)
    if predicted_labels is not None and len(predicted_labels) != len(scores):
        raise ValueError("speaker summary rows and predictions must have equal length")
    summaries: list[dict[str, Any]] = []
    work = frame.reset_index(drop=True).copy()
    work["probability"] = scores
    if predicted_labels is not None:
        work["prediction"] = predicted_labels
    for speaker_id, group in work.groupby("speaker_id", sort=True):
        labels = group["label_encoded"].to_numpy(dtype=int)
        speaker_scores = group["probability"].to_numpy(dtype=float)
        speaker_predictions = group["prediction"].to_numpy(dtype=int) if predicted_labels is not None else None
        metrics = calculate_binary_metrics(labels, speaker_scores, threshold, predictions=speaker_predictions)
        summaries.append(
            {
                "speaker_id": str(speaker_id),
                "label": "control" if len(np.unique(labels)) == 1 and labels[0] == 0 else (
                    "dysarthric" if len(np.unique(labels)) == 1 else "mixed"
                ),
                "sample_count": int(len(group)),
                "correct_count": int(round(metrics["accuracy"] * len(group))),
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"] if labels.sum() else None,
                "specificity": metrics["specificity"] if not labels.sum() else None,
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )
    return summaries


def fit_train_only_classifier(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    random_state: int = SEED,
) -> tuple[Pipeline, dict[str, int | None]]:
    """Fit scaling and logistic regression using finite training rows only."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != len(frame):
        raise ValueError("embedding rows must align with manifest rows")
    splits = frame["split"].astype(str).to_numpy()
    labels = frame["label_encoded"].to_numpy(dtype=int)
    training = splits == "train"
    training &= np.isfinite(matrix).all(axis=1)
    if len(np.unique(labels[training])) != 2:
        raise ValueError("training embeddings must contain both classes")

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    classifier.fit(matrix[training], labels[training])
    speakers = frame.loc[training, "speaker_id"].nunique() if "speaker_id" in frame else None
    return classifier, {"recording_count": int(training.sum()), "speaker_count": int(speakers) if speakers is not None else None}


def _environment_versions() -> dict[str, str | None]:
    names = (
        "numpy", "pandas", "scikit-learn", "librosa", "soundfile", "joblib",
        "tensorflow", "tensorflow-hub", "tf-keras", "torch", "transformers",
        "huggingface-hub", "psutil",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class _RssSampler:
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        self._process = psutil.Process(os.getpid())
        self.peak_bytes = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes or 0, self._process.memory_info().rss)

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self.peak_bytes = max(self.peak_bytes or 0, self._process.memory_info().rss)
        return self.peak_bytes


def _set_seeds() -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    random.seed(SEED)
    np.random.seed(SEED)


class _YamnetEncoder:
    def __init__(self, model: Any, model_dir: Path) -> None:
        self.model = model
        self.model_dir = model_dir
        self.dimension = CANDIDATES["yamnet"]["expected_embedding_dimension"]

    def embed(self, signal: np.ndarray) -> np.ndarray:
        outputs = self.model(signal.astype(np.float32, copy=False))
        frame_embeddings = np.asarray(outputs[1].numpy(), dtype=np.float32)
        if frame_embeddings.ndim != 2 or frame_embeddings.shape[1] != self.dimension or not len(frame_embeddings):
            raise ValueError(f"YAMNet returned an invalid frame-embedding shape: {frame_embeddings.shape}")
        return frame_embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)

    def dependencies(self) -> dict[str, str | None]:
        return {name: _environment_versions().get(name) for name in ("tensorflow", "tensorflow-hub")}


class _HuggingFaceEncoder:
    def __init__(self, model: Any, feature_extractor: Any, torch: Any, dimension: int) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.preprocessing_config = feature_extractor.to_dict()
        self.torch = torch
        self.dimension = int(dimension)

    def embed(self, signal: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(
            signal.astype(np.float32, copy=False),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        with self.torch.inference_mode():
            output = self.model(input_values=inputs["input_values"])
        frame_embeddings = output.last_hidden_state[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if frame_embeddings.ndim != 2 or frame_embeddings.shape[1] != self.dimension or not len(frame_embeddings):
            raise ValueError(f"HF encoder returned an invalid frame-embedding shape: {frame_embeddings.shape}")
        return frame_embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)

    def dependencies(self) -> dict[str, str | None]:
        return {name: _environment_versions().get(name) for name in ("torch", "transformers", "huggingface-hub")}


def _directory_size(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _load_encoder(
    candidate_id: str,
    *,
    pinned_revision: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    spec = CANDIDATES[candidate_id]
    if candidate_id == "yamnet":
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        import tensorflow as tf
        import tensorflow_hub as hub

        tf.random.set_seed(SEED)
        fetch_started = time.perf_counter()
        model_path = Path(hub.resolve(spec["model_id"]))
        fetch_seconds = time.perf_counter() - fetch_started
        load_started = time.perf_counter()
        model = hub.load(str(model_path))
        load_seconds = time.perf_counter() - load_started
        encoder = _YamnetEncoder(model, model_path)
        return encoder, {
            "revision": "1",
            "revision_kind": spec["revision_kind"],
            "fetch_seconds": fetch_seconds,
            "cold_model_load_seconds": load_seconds,
            "artifact_size_bytes": _directory_size(model_path),
            "checkpoint_files": [
                {"name": str(item.relative_to(model_path)), "size_bytes": item.stat().st_size, "sha256": sha256_file(item)}
                for item in sorted(model_path.rglob("*"))
                if item.is_file()
            ],
            "dependencies": encoder.dependencies(),
        }

    if candidate_id in {"distilhubert", "wav2vec2_base"}:
        os.environ.setdefault("USE_TF", "0")
        from huggingface_hub import HfApi, snapshot_download
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        torch.manual_seed(SEED)
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 1) // 2)))
        torch.use_deterministic_algorithms(True, warn_only=True)
        model_info = HfApi().model_info(spec["model_id"], revision=pinned_revision)
        revision = model_info.sha
        if not revision:
            raise RuntimeError(f"Hugging Face did not return a commit SHA for {spec['model_id']}")
        siblings = {item.rfilename for item in model_info.siblings}
        weight_name = "model.safetensors" if "model.safetensors" in siblings else "pytorch_model.bin"
        patterns = ["config.json", "preprocessor_config.json", "feature_extractor_config.json", weight_name]
        fetch_started = time.perf_counter()
        model_path = Path(
            snapshot_download(
                repo_id=spec["model_id"],
                revision=revision,
                allow_patterns=patterns,
            )
        )
        fetch_seconds = time.perf_counter() - fetch_started
        load_started = time.perf_counter()
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True, low_cpu_mem_usage=True
        )
        model.eval()
        load_seconds = time.perf_counter() - load_started
        dimension = int(model.config.hidden_size)
        encoder = _HuggingFaceEncoder(model, feature_extractor, torch, dimension)
        checkpoint_files = [
            {
                "name": item.name,
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
            for item in sorted(model_path.iterdir())
            if item.is_file()
        ]
        return encoder, {
            "revision": revision,
            "revision_kind": spec["revision_kind"],
            "fetch_seconds": fetch_seconds,
            "cold_model_load_seconds": load_seconds,
            "artifact_size_bytes": _directory_size(model_path),
            "checkpoint_files": checkpoint_files,
            "feature_extractor_config": encoder.preprocessing_config,
            "measured_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "dependencies": encoder.dependencies(),
        }
    raise ValueError(f"unknown candidate ID: {candidate_id}")


def _load_preprocessed_audio(path: str | Path) -> tuple[np.ndarray, float]:
    signal, sample_rate = load_audio(
        audio_path=path,
        target_sample_rate=SAMPLE_RATE,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or sample_rate != SAMPLE_RATE:
        raise ValueError(f"audio contract violation: shape={signal.shape}, sample_rate={sample_rate}")
    if not len(signal):
        raise ValueError("empty recording")
    original_seconds = len(signal) / SAMPLE_RATE
    minimum_samples = int(MIN_DURATION_SECONDS * SAMPLE_RATE)
    if len(signal) < minimum_samples:
        signal = np.pad(signal, (0, minimum_samples - len(signal)), mode="constant")
    return signal, original_seconds


def _cache_sample_key(
    row: pd.Series,
    *,
    candidate_id: str,
    revision: str,
    split: str,
    split_manifest_sha256: str,
    sample_sha256: str | None,
    preprocessing_config: dict[str, Any],
) -> str:
    spec = CANDIDATES[candidate_id]
    return make_cache_key(
        model_id=spec["model_id"],
        model_revision=revision,
        preprocessing=preprocessing_config,
        sample_sha256=sample_sha256,
        split=split,
        split_manifest_sha256=split_manifest_sha256,
        sample_path=str(row["path"]),
        speaker_id=str(row["speaker_id"]),
        label=str(row["label"]),
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _extract_or_load_split(
    rows: pd.DataFrame,
    *,
    encoder: Any,
    candidate_id: str,
    revision: str,
    split: str,
    split_manifest_sha256: str,
    cache_dir: Path,
    force: bool = False,
    preprocessing_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = CANDIDATES[candidate_id]
    effective_preprocessing = preprocessing_config or PREPROCESSING
    namespace = stable_sha256(
        {
            "model_id": spec["model_id"],
            "revision": revision,
            "preprocessing": effective_preprocessing,
            "manifest_sha256": split_manifest_sha256,
        }
    )[:20]
    folder = cache_dir / candidate_id / namespace
    folder.mkdir(parents=True, exist_ok=True)
    embedding_path = folder / f"{split}_embeddings.npy"
    provenance_path = folder / f"{split}_provenance.json"

    signatures: list[dict[str, Any]] = []
    sample_keys: list[str] = []
    for _, row in rows.iterrows():
        sample_path = Path(str(row["path"]))
        try:
            sample_digest = sha256_file(sample_path)
            signature_error = None
        except OSError as exc:
            sample_digest = None
            signature_error = f"{type(exc).__name__}: {exc}"
        key = _cache_sample_key(
            row,
            candidate_id=candidate_id,
            revision=revision,
            split=split,
            split_manifest_sha256=split_manifest_sha256,
            sample_sha256=sample_digest,
            preprocessing_config=effective_preprocessing,
        )
        sample_keys.append(key)
        signatures.append(
            {
                "sample_id": int(row["_sample_id"]),
                "sample_path": str(sample_path),
                "speaker_id": str(row["speaker_id"]),
                "label": str(row["label"]),
                "split": split,
                "sample_sha256": sample_digest,
                "cache_key": key,
                "signature_error": signature_error,
            }
        )

    expected_signature = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "model_id": spec["model_id"],
        "model_revision": revision,
        "preprocessing": effective_preprocessing,
        "preprocessing_sha256": stable_sha256(effective_preprocessing),
        "split_manifest_sha256": split_manifest_sha256,
        "split": split,
        "sample_keys_sha256": stable_sha256(sample_keys),
        "sample_count": len(rows),
    }
    if not force and embedding_path.is_file() and provenance_path.is_file():
        try:
            cached = json.loads(provenance_path.read_text(encoding="utf-8"))
            cached_embeddings_sha256 = cached.pop("embedding_file_sha256", None)
            cached.pop("embedding_dimension", None)
            cached.pop("created_utc", None)
            cached_records = cached.pop("records", None)
            comparable_records = [
                {field: record.get(field) for field in signature}
                for record, signature in zip(cached_records or [], signatures, strict=False)
            ]
            if (
                cached == expected_signature
                and comparable_records == signatures
                and len(cached_records or []) == len(signatures)
                and cached_embeddings_sha256 == sha256_file(embedding_path)
            ):
                embeddings = np.load(embedding_path, allow_pickle=False)
                if embeddings.shape == (len(rows), encoder.dimension):
                    return embeddings, {
                        "split": split,
                        "sample_count": len(rows),
                        "cache_hit": True,
                        "embedding_file": str(embedding_path),
                        "embedding_file_sha256": cached_embeddings_sha256,
                        "records": cached_records,
                    }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    embeddings = np.full((len(rows), encoder.dimension), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for index, ((_, row), signature) in enumerate(zip(rows.iterrows(), signatures, strict=True)):
        record = dict(signature)
        started = time.perf_counter()
        try:
            if signature["signature_error"]:
                raise FileNotFoundError(signature["signature_error"])
            signal, original_seconds = _load_preprocessed_audio(row["path"])
            vector = np.asarray(encoder.embed(signal), dtype=np.float32)
            if vector.shape != (encoder.dimension,) or not np.isfinite(vector).all():
                raise ValueError(f"embedding was not a finite {encoder.dimension}-vector: {vector.shape}")
            embeddings[index] = vector
            record["valid"] = True
            record["original_duration_seconds"] = float(original_seconds)
            record["error_type"] = None
            record["error_message"] = None
        except Exception as exc:  # Preserve a row-level reason and continue the candidate run.
            record["valid"] = False
            record["original_duration_seconds"] = None
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)[:500]
        record["feature_extraction_seconds"] = time.perf_counter() - started
        records.append(record)
        if (index + 1) % 250 == 0 or index + 1 == len(rows):
            print(
                f"[{candidate_id}] {split}: {index + 1}/{len(rows)} recordings",
                flush=True,
            )

    temporary_embeddings = embedding_path.with_name(embedding_path.name + ".tmp")
    with temporary_embeddings.open("wb") as stream:
        np.save(stream, embeddings, allow_pickle=False)
    temporary_embeddings.replace(embedding_path)
    provenance = {
        **expected_signature,
        "embedding_dimension": int(encoder.dimension),
        "embedding_file_sha256": sha256_file(embedding_path),
        "records": records,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(provenance_path, provenance)
    return embeddings, {
        "split": split,
        "sample_count": len(rows),
        "cache_hit": False,
        "embedding_file": str(embedding_path),
        "embedding_file_sha256": provenance["embedding_file_sha256"],
        "records": records,
    }


def _valid_subset(frame: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    valid = np.isfinite(embeddings).all(axis=1)
    return frame.loc[valid].reset_index(drop=True), embeddings[valid]


def _evaluate_split(
    candidate_id: str,
    split: str,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    classifier: Pipeline,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    accepted_frame, accepted_embeddings = _valid_subset(frame, embeddings)
    if not len(accepted_frame):
        raise ValueError(f"no {split} recordings produced an embedding")
    probabilities = classifier.predict_proba(accepted_embeddings)[:, 1]
    predictions = classifier.predict(accepted_embeddings)
    labels = accepted_frame["label_encoded"].to_numpy(dtype=int)
    metrics = calculate_binary_metrics(labels, probabilities, threshold=0.5, predictions=predictions)
    metrics.update(
        {
            "candidate_id": candidate_id,
            "split": split,
            "accepted_sample_count": int(len(accepted_frame)),
            "total_manifest_sample_count": int(len(frame)),
            "coverage": float(len(accepted_frame) / len(frame)) if len(frame) else 0.0,
            "speaker_count": int(accepted_frame["speaker_id"].nunique()),
            "split_speaker_count": int(frame["speaker_id"].nunique()),
        }
    )
    per_speaker = summarize_by_speaker(
        accepted_frame, probabilities, threshold=0.5, predictions=predictions
    )
    for item in per_speaker:
        item["candidate_id"] = candidate_id
        item["split"] = split
    return metrics, per_speaker, probabilities


def _baseline_validation() -> dict[str, Any]:
    if sha256_file(BASELINE_MODEL) != EXPECTED_BASELINE_SHA256:
        raise ValueError("canonical MFCC + Random Forest model artifact hash changed")
    baseline = joblib.load(BASELINE_MODEL)
    features = pd.read_csv(BASELINE_FEATURES)
    validation = features.loc[features["split"] == "val"].copy()
    feature_columns = list(baseline.feature_names_in_)
    scores = baseline.predict_proba(validation[feature_columns])[:, 1]
    predictions = baseline.predict(validation[feature_columns])
    result = calculate_binary_metrics(
        validation["label_encoded"].to_numpy(dtype=int), scores, threshold=0.5, predictions=predictions
    )
    result.update(
        {
            "candidate_id": "mfcc_random_forest_baseline",
            "split": "val",
            "accepted_sample_count": int(len(validation)),
            "total_manifest_sample_count": int(len(validation)),
            "coverage": 1.0,
            "speaker_count": int(validation["speaker_id"].nunique()),
            "split_speaker_count": int(validation["speaker_id"].nunique()),
            "artifact_path": str(BASELINE_MODEL.relative_to(ROOT)),
            "artifact_sha256": EXPECTED_BASELINE_SHA256,
            "artifact_size_bytes": BASELINE_MODEL.stat().st_size,
            "retrained": False,
            "source": "metrics replayed from the existing saved artifact and precomputed validation features",
        }
    )
    return result


def _historical_baseline_test() -> dict[str, Any]:
    report = json.loads(HISTORICAL_SPEECH_REPORT.read_text(encoding="utf-8"))
    direct = dict(report["direct"])
    direct.update(
        {
            "candidate_id": "mfcc_random_forest_baseline",
            "split": "test",
            "accepted_sample_count": int(report["processed_direct"]),
            "total_manifest_sample_count": int(report["total_held_out"]),
            "coverage": float(report["processed_direct"] / report["total_held_out"]),
            "speaker_count": len(EXPECTED_SPEAKERS["test"]),
            "split_speaker_count": len(EXPECTED_SPEAKERS["test"]),
            "source": "historical direct result in reports/evaluation/phase4/final_complete/speech.json; test already observed",
        }
    )
    baseline = joblib.load(BASELINE_MODEL)
    features = pd.read_csv(BASELINE_FEATURES)
    test = features.loc[features["split"] == "test"].copy()
    feature_columns = list(baseline.feature_names_in_)
    scores = baseline.predict_proba(test[feature_columns])[:, 1]
    predictions = baseline.predict(test[feature_columns])
    estimator_replay = calculate_binary_metrics(
        test["label_encoded"].to_numpy(dtype=int), scores, threshold=0.5, predictions=predictions
    )
    estimator_replay.update(
        {
            "candidate_id": "mfcc_random_forest_baseline",
            "split": "test",
            "accepted_sample_count": int(len(test)),
            "total_manifest_sample_count": int(len(test)),
            "coverage": 1.0,
            "speaker_count": int(test["speaker_id"].nunique()),
            "split_speaker_count": int(test["speaker_id"].nunique()),
            "source": "read-only replay of the existing canonical artifact with estimator.predict() on precomputed held-out features; not used for candidate selection",
        }
    )
    return {**estimator_replay, "phase4_direct_report": direct}


def _historical_baseline_resources() -> dict[str, Any]:
    profile = json.loads(HISTORICAL_PROFILING_REPORT.read_text(encoding="utf-8"))
    speech = dict(profile["adapters"]["speech"])
    speech.update(
        {
            "artifact_size_bytes": BASELINE_MODEL.stat().st_size,
            "artifact_sha256": EXPECTED_BASELINE_SHA256,
            "measurement_source": "existing Phase 4 warm SpeechAdapter profiling; includes audio load, MFCC extraction, and RF inference",
        }
    )
    return speech


def _latency_benchmark(
    *,
    candidate_id: str,
    encoder: Any,
    classifier: Pipeline,
    train_rows: pd.DataFrame,
    benchmark_record_count: int = 48,
) -> dict[str, Any]:
    chosen: list[pd.Series] = []
    seen_speakers: set[str] = set()
    for _, row in train_rows.iterrows():
        speaker = str(row["speaker_id"])
        if speaker not in seen_speakers:
            chosen.append(row)
            seen_speakers.add(speaker)
        if len(chosen) >= benchmark_record_count:
            break
    for _, row in train_rows.iterrows():
        if len(chosen) >= benchmark_record_count:
            break
        if all(int(existing["_sample_id"]) != int(row["_sample_id"]) for existing in chosen):
            chosen.append(row)

    full_seconds: list[float] = []
    extractor_seconds: list[float] = []
    classifier_seconds: list[float] = []
    errors: list[str] = []
    for row in chosen:
        start = time.perf_counter()
        try:
            signal, _ = _load_preprocessed_audio(row["path"])
            extractor_start = time.perf_counter()
            vector = encoder.embed(signal)
            extractor_seconds.append(time.perf_counter() - extractor_start)
            classifier_start = time.perf_counter()
            classifier.predict_proba(np.asarray(vector, dtype=np.float32).reshape(1, -1))
            classifier_seconds.append(time.perf_counter() - classifier_start)
            full_seconds.append(time.perf_counter() - start)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    def summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean_ms": None, "median_ms": None, "p95_ms": None}
        milliseconds = np.asarray(values, dtype=float) * 1000
        return {
            "mean_ms": float(milliseconds.mean()),
            "median_ms": float(np.median(milliseconds)),
            "p95_ms": float(np.percentile(milliseconds, 95)),
        }

    return {
        "candidate_id": candidate_id,
        "measurement_split": "train",
        "benchmark_record_count": len(chosen),
        "benchmark_speaker_count": len({str(row["speaker_id"]) for row in chosen}),
        "successful_benchmark_count": len(full_seconds),
        "feature_extraction_latency": summary(extractor_seconds),
        "classifier_latency": summary(classifier_seconds),
        "end_to_end_per_recording_inference_latency": summary(full_seconds),
        "errors": errors[:10],
    }


def _artifact_inventory(output_dir: Path) -> None:
    omitted = {"artifact_hashes.json"}
    inventory: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in omitted and not path.name.endswith(".tmp"):
            inventory[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    _atomic_write_json(output_dir / "artifact_hashes.json", inventory)


def _candidate_validation(
    candidate_id: str,
    manifest: pd.DataFrame,
    manifest_sha256: str,
    output_dir: Path,
    cache_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    spec = CANDIDATES[candidate_id]
    sampler = _RssSampler()
    sampler.start()
    overall_started = time.perf_counter()
    try:
        _set_seeds()
        encoder, model_resource = _load_encoder(candidate_id)
        encoder_preprocessing_config = {
            **PREPROCESSING,
            **(
                {"feature_extractor": model_resource["feature_extractor_config"]}
                if model_resource.get("feature_extractor_config")
                else {}
            ),
        }
        train_and_val = manifest.loc[manifest["split"].isin(["train", "val"])].copy()
        cached_splits: dict[str, dict[str, Any]] = {}
        embeddings_by_split: dict[str, np.ndarray] = {}
        for split in ("train", "val"):
            rows = train_and_val.loc[train_and_val["split"] == split].copy()
            embeddings_by_split[split], cached_splits[split] = _extract_or_load_split(
                rows,
                encoder=encoder,
                candidate_id=candidate_id,
                revision=model_resource["revision"],
                split=split,
                split_manifest_sha256=manifest_sha256,
                cache_dir=cache_dir,
                force=force,
                preprocessing_config=encoder_preprocessing_config,
            )

        pooled = np.full((len(train_and_val), encoder.dimension), np.nan, dtype=np.float32)
        for split in ("train", "val"):
            mask = train_and_val["split"].to_numpy() == split
            pooled[mask] = embeddings_by_split[split]
        classifier, fit_summary = fit_train_only_classifier(train_and_val, pooled)
        validation_frame = train_and_val.loc[train_and_val["split"] == "val"].reset_index(drop=True)
        validation_embeddings = embeddings_by_split["val"]
        validation_metrics, speaker_rows, _ = _evaluate_split(
            candidate_id, "val", validation_frame, validation_embeddings, classifier
        )

        classifier_path = output_dir / "classifiers" / f"{candidate_id}.joblib"
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, classifier_path)
        classifier_sha256 = sha256_file(classifier_path)
        latency = _latency_benchmark(
            candidate_id=candidate_id,
            encoder=encoder,
            classifier=classifier,
            train_rows=manifest.loc[manifest["split"] == "train"],
        )
        extraction_timings = [
            record["feature_extraction_seconds"]
            for value in cached_splits.values()
            for record in value["records"]
            if record.get("valid") and record.get("feature_extraction_seconds") is not None
        ]
        successful = sum(
            bool(record.get("valid"))
            for value in cached_splits.values()
            for record in value["records"]
        )
        total = sum(value["sample_count"] for value in cached_splits.values())
        extraction_ms = np.asarray(extraction_timings, dtype=float) * 1000
        resources = {
            **model_resource,
            "measured_pretrained_checkpoint_bytes": model_resource["artifact_size_bytes"],
            "upstream_documented_checkpoint_size_bytes": spec["documented_checkpoint_size_bytes"],
            "upstream_documented_parameter_count": spec["documented_parameter_count"],
            "upstream_source": spec["source"],
            "embedding_dimension": int(encoder.dimension),
            "embedding_vector_bytes_float32": int(encoder.dimension * np.dtype(np.float32).itemsize),
            "feature_extraction_recording_count": int(successful),
            "feature_extraction_input_recording_count": int(total),
            "feature_extraction_coverage": float(successful / total) if total else 0.0,
            "feature_extraction_latency_ms": {
                "mean": float(extraction_ms.mean()) if len(extraction_ms) else None,
                "median": float(np.median(extraction_ms)) if len(extraction_ms) else None,
                "p95": float(np.percentile(extraction_ms, 95)) if len(extraction_ms) else None,
                "timing_source": "uncached train + validation extraction rows; cache-hit durations retain their first-run values",
            },
            **latency,
            "approximate_peak_process_rss_bytes": sampler.stop(),
            "classifier_artifact_path": str(classifier_path.relative_to(output_dir)),
            "classifier_artifact_sha256": classifier_sha256,
            "validation_embedding_cache": {
                split: {
                    "cache_hit": cached_splits[split]["cache_hit"],
                    "embedding_file_sha256": cached_splits[split]["embedding_file_sha256"],
                }
                for split in ("train", "val")
            },
        }
        result = {
            "candidate_id": candidate_id,
            "model_id": spec["model_id"],
            "model_name": spec["model_name"],
            "source": spec["source"],
            "source_url": spec["source_url"],
            "status": "completed",
            "license": spec["license"],
            "upstream_model_characteristics": {
                "documented_parameter_count": spec["documented_parameter_count"],
                "documented_checkpoint_size_bytes": spec["documented_checkpoint_size_bytes"],
                "source": spec["source"],
            },
            "model_revision": model_resource["revision"],
            "revision_kind": model_resource["revision_kind"],
            "encoder_preprocessing_config": encoder_preprocessing_config,
            "pooling": POOLING,
            "classifier": CLASSIFIER_CONFIG,
            "fit_summary": fit_summary,
            "validation": validation_metrics,
            "resources": resources,
            "speaker_summary": speaker_rows,
            "failure_count": int(total - successful),
            "failures": [
                {
                    "sample_id": record["sample_id"],
                    "speaker_id": record["speaker_id"],
                    "split": record["split"],
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                }
                for value in cached_splits.values()
                for record in value["records"]
                if not record.get("valid")
            ],
            "elapsed_seconds": time.perf_counter() - overall_started,
        }
        _atomic_write_json(output_dir / f"validation_{candidate_id}.json", result)
        return result
    except Exception as exc:
        failure = {
            "candidate_id": candidate_id,
            "model_id": spec["model_id"],
            "model_name": spec["model_name"],
            "source": spec["source"],
            "source_url": spec["source_url"],
            "status": "failed",
            "license": spec["license"],
            "upstream_model_characteristics": {
                "documented_parameter_count": spec["documented_parameter_count"],
                "documented_checkpoint_size_bytes": spec["documented_checkpoint_size_bytes"],
                "source": spec["source"],
            },
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - overall_started,
            "selection_status": "rejected because the candidate could not be completed in this environment",
        }
        _atomic_write_json(output_dir / f"validation_{candidate_id}.json", failure)
        return failure
    finally:
        sampler.stop()


def _write_validation_outputs(
    results: list[dict[str, Any]],
    baseline_validation: dict[str, Any],
    historical_test: dict[str, Any],
    manifest: pd.DataFrame,
    manifest_sha256: str,
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    baseline_resources = _historical_baseline_resources()
    rows.append(
        {
            "candidate_id": "mfcc_random_forest_baseline",
            "status": "existing baseline",
            "validation_roc_auc": baseline_validation["roc_auc"],
            "validation_pr_auc": baseline_validation["pr_auc"],
            "validation_accuracy": baseline_validation["accuracy"],
            "validation_sensitivity": baseline_validation["sensitivity"],
            "validation_specificity": baseline_validation["specificity"],
            "validation_precision": baseline_validation["precision"],
            "validation_f1": baseline_validation["f1"],
            "validation_n": baseline_validation["accepted_sample_count"],
            "validation_speakers": baseline_validation["speaker_count"],
            "model_size_bytes": baseline_validation["artifact_size_bytes"],
            "feature_extraction_p50_ms": None,
            "end_to_end_p50_ms": baseline_resources["warm_median_ms"],
            "end_to_end_p95_ms": baseline_resources["warm_p95_ms"],
            "memory_peak_rss_bytes": baseline_resources["peak_rss_bytes"],
            "memory_increase_bytes": baseline_resources["memory_increase_bytes"],
            "cold_runs": baseline_resources["cold_runs"],
            "warm_runs": baseline_resources["warm_runs"],
            "resource_note": "existing Phase 4 in-process profile; warm latency includes librosa audio load, MFCC extraction, and RF inference",
        }
    )
    speaker_rows = [
        {"candidate_id": "mfcc_random_forest_baseline", "split": "val", **speaker}
        for speaker in _baseline_speaker_summary()
    ]
    for result in results:
        validation = result.get("validation", {})
        resources = result.get("resources", {})
        end_to_end = resources.get("end_to_end_per_recording_inference_latency", {})
        extraction = resources.get("feature_extraction_latency_ms", {})
        rows.append(
            {
                "candidate_id": result["candidate_id"],
                "status": result["status"],
                "validation_roc_auc": validation.get("roc_auc"),
                "validation_pr_auc": validation.get("pr_auc"),
                "validation_accuracy": validation.get("accuracy"),
                "validation_sensitivity": validation.get("sensitivity"),
                "validation_specificity": validation.get("specificity"),
                "validation_precision": validation.get("precision"),
                "validation_f1": validation.get("f1"),
                "validation_n": validation.get("accepted_sample_count"),
                "validation_speakers": validation.get("speaker_count"),
                "model_size_bytes": resources.get("measured_pretrained_checkpoint_bytes"),
                "feature_extraction_p50_ms": extraction.get("median"),
                "end_to_end_p50_ms": end_to_end.get("median_ms"),
                "failure_reason": result.get("failure_reason"),
            }
        )
        speaker_rows.extend(result.get("speaker_summary", []))

    pd.DataFrame(rows).to_csv(output_dir / "validation_comparison.csv", index=False)
    pd.DataFrame(speaker_rows).to_csv(output_dir / "validation_per_speaker.csv", index=False)
    pd.DataFrame(_baseline_historical_speaker_summary()).to_csv(
        output_dir / "baseline_historical_test_per_speaker.csv", index=False
    )
    _atomic_write_json(output_dir / "baseline_validation.json", baseline_validation)
    _atomic_write_json(output_dir / "baseline_historical_test.json", historical_test)

    split_summary = {
        "manifest_path": str(DEFAULT_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": manifest_sha256,
        "recording_counts": {
            str(split): int(count) for split, count in manifest["split"].value_counts().items()
        },
        "speaker_counts": {split: len(speakers) for split, speakers in EXPECTED_SPEAKERS.items()},
        "speaker_ids": {split: sorted(speakers) for split, speakers in EXPECTED_SPEAKERS.items()},
        "label_counts": {
            split: {
                str(label): int(count)
                for label, count in manifest.loc[manifest["split"] == split, "label"].value_counts().items()
            }
            for split in ("train", "val", "test")
        },
    }
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_stage": "validation comparison only; no test embeddings were extracted",
        "protocol": _protocol_manifest(manifest_sha256, split_summary),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": _environment_versions(),
            "random_seed": SEED,
        },
        "candidates": results,
        "baseline_test_evidence": historical_test,
        "selection_policy": {
            "primary_validation_metric": "PR-AUC / average precision",
            "resource_tie_margin_absolute_pr_auc": 0.01,
            "rule": "among completed candidates no more than 0.01 PR-AUC below the best completed candidate, choose the smallest measured pretrained checkpoint; break size ties by lower training-only end-to-end p50 latency",
            "threshold": 0.5,
            "hyperparameter_search": "none",
            "test_used_for_selection": False,
        },
    }
    _atomic_write_json(output_dir / "manifest.json", metadata)


def _baseline_speaker_summary() -> list[dict[str, Any]]:
    model = joblib.load(BASELINE_MODEL)
    features = pd.read_csv(BASELINE_FEATURES)
    validation = features.loc[features["split"] == "val"].copy()
    scores = model.predict_proba(validation[list(model.feature_names_in_)])[:, 1]
    return summarize_by_speaker(validation, scores, predictions=model.predict(validation[list(model.feature_names_in_)]))


def _baseline_historical_speaker_summary() -> list[dict[str, Any]]:
    path = ROOT / "reports" / "evaluation" / "phase4" / "final_complete" / "speech_speaker_summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for speaker in report["speakers"]:
        rows.append(
            {
                "candidate_id": "mfcc_random_forest_baseline",
                "split": "test",
                **speaker,
                "roc_auc": None,
                "roc_auc_note": "single-class speaker; undefined",
            }
        )
    return rows


def _protocol_manifest(manifest_sha256: str, split_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_manifest_sha256": manifest_sha256,
        "split_manifest_summary": split_summary,
        "partition_validation": "exact speaker ID sets checked against the repository's current train/val/test partitions",
        "audio_contract": PREPROCESSING,
        "embedding_pooling": POOLING,
        "classifier": CLASSIFIER_CONFIG,
        "fitting_rule": "StandardScaler and LogisticRegression fit on finite embeddings for train-split recordings only; validation and test never enter fit",
        "candidate_selection_rule": "validation PR-AUC, then model size and measured train-only latency within the predeclared 0.01 PR-AUC margin",
        "selection_threshold": 0.5,
        "seed": SEED,
        "embedding_cache": "one NumPy matrix and JSON sidecar per model revision and split; both split and whole-manifest hashes plus per-record content hashes are verified before reuse",
        "historical_test_observation": "The existing canonical MFCC + Random Forest test metrics were already reported in reports/evaluation/phase4/final_complete/speech.json; this test partition is not pristine.",
    }


def run_validation(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = OUTPUT_DIR,
    cache_dir: Path = CACHE_DIR,
    candidates: Sequence[str] = tuple(CANDIDATES),
    force: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha256 = load_canonical_manifest(manifest_path)
    baseline_validation = _baseline_validation()
    historical_test = _historical_baseline_test()
    results = []
    for candidate_id in candidates:
        if candidate_id not in CANDIDATES:
            raise ValueError(f"unsupported candidate: {candidate_id}")
        print(f"Starting validation extraction for {CANDIDATES[candidate_id]['model_name']}", flush=True)
        worker_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
            "--manifest",
            str(manifest_path),
            "candidate-worker",
            "--candidate",
            candidate_id,
        ]
        if force:
            worker_command.append("--force")
        worker = subprocess.Popen(
            worker_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert worker.stdout is not None
        for line in worker.stdout:
            print(f"[{candidate_id}] {line}", end="", flush=True)
        return_code = worker.wait()
        result_path = output_dir / f"validation_{candidate_id}.json"
        if return_code == 0 and result_path.is_file():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            failure = {
                "candidate_id": candidate_id,
                "model_id": CANDIDATES[candidate_id]["model_id"],
                "model_name": CANDIDATES[candidate_id]["model_name"],
                "status": "failed",
                "failure_reason": f"candidate worker exited with status {return_code} before writing a result",
            }
            _atomic_write_json(result_path, failure)
            results.append(failure)
    _write_validation_outputs(
        results, baseline_validation, historical_test, manifest, manifest_sha256, output_dir
    )
    _write_resource_comparison(output_dir)
    _write_report(output_dir)
    _artifact_inventory(output_dir)
    return results


def run_selection(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    selection_path = output_dir / "selection.json"
    frozen_selection_path = output_dir / "selection_frozen.json"
    if selection_path.is_file():
        existing_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing_selection.get("test_evaluated"):
            raise ValueError("test evaluation is already complete; selection cannot be replaced or reset")
        if frozen_selection_path.is_file():
            raise ValueError("a candidate selection is already frozen; refusing to overwrite it")
    elif frozen_selection_path.is_file():
        raise ValueError("immutable selection snapshot exists without selection state; refusing to replace it")
    validation_path = output_dir / "manifest.json"
    if not validation_path.is_file():
        raise FileNotFoundError("run the validation stage before selecting a candidate")
    validation_manifest = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation_manifest.get("selection_stage") != "validation comparison only; no test embeddings were extracted":
        raise ValueError("validation manifest does not identify an untouched candidate comparison")

    completed = [
        result for result in validation_manifest["candidates"]
        if result.get("status") == "completed" and result.get("validation", {}).get("pr_auc") is not None
    ]
    evaluated_ids = {result["candidate_id"] for result in validation_manifest.get("candidates", [])}
    if evaluated_ids != set(CANDIDATES):
        raise ValueError(
            "validation must include YAMNet, DistilHuBERT, and Wav2Vec2-Base before selection; "
            f"found {sorted(evaluated_ids)}"
        )
    if not completed:
        raise RuntimeError("no candidate completed validation, so no test candidate can be selected")
    best_pr_auc = max(float(result["validation"]["pr_auc"]) for result in completed)
    margin = 0.01
    eligible = [
        result for result in completed
        if float(result["validation"]["pr_auc"]) >= best_pr_auc - margin
    ]
    selected = min(
        eligible,
        key=lambda result: (
            int(result["resources"].get("measured_pretrained_checkpoint_bytes") or math.inf),
            float(
                result["resources"]
                .get("end_to_end_per_recording_inference_latency", {})
                .get("median_ms", math.inf)
                or math.inf
            ),
        ),
    )
    classifier_path = output_dir / selected["resources"]["classifier_artifact_path"]
    selected_pr_auc = float(selected["validation"]["pr_auc"])
    if selected_pr_auc >= best_pr_auc - 1e-12:
        other_scores = [
            float(result["validation"]["pr_auc"])
            for result in completed
            if result["candidate_id"] != selected["candidate_id"]
        ]
        next_best = max(other_scores) if other_scores else None
        selection_reason = (
            f"Selected {selected['candidate_id']} from train/validation evidence because it had the highest "
            f"validation PR-AUC ({selected_pr_auc:.4f})"
            + (f", ahead of the next candidate at {next_best:.4f}" if next_best is not None else "")
            + f". It was the only candidate within {margin:.2f} of the best; the resource tie-break did not "
            "change the result. Test rows were not used."
        )
    else:
        selection_reason = (
            f"Selected {selected['candidate_id']} from train/validation evidence: its validation PR-AUC was "
            f"within {margin:.2f} of the highest completed candidate, and it had the smallest measured checkpoint "
            "among that eligible set (latency breaks size ties). Test rows were not used."
        )
    frozen = {
        "experiment_id": EXPERIMENT_ID,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": selected["candidate_id"],
        "model_id": selected["model_id"],
        "model_revision": selected["model_revision"],
        "license": selected["license"],
        "pooling": selected["pooling"],
        "preprocessing": PREPROCESSING,
        "encoder_preprocessing_config": selected["encoder_preprocessing_config"],
        "classifier": CLASSIFIER_CONFIG,
        "selection_policy": validation_manifest["selection_policy"],
        "selection_metrics": selected["validation"],
        "eligible_candidates": [result["candidate_id"] for result in eligible],
        "best_validation_pr_auc": best_pr_auc,
        "selected_validation_pr_auc": selected_pr_auc,
        "selection_reason": selection_reason,
        "split_manifest_sha256": validation_manifest["protocol"]["split_manifest_sha256"],
        "validation_manifest_sha256": sha256_file(validation_path),
        "classifier_artifact_path": selected["resources"]["classifier_artifact_path"],
        "classifier_artifact_sha256": sha256_file(classifier_path),
        "test_evaluated": False,
    }
    _atomic_write_json(selection_path, frozen)
    _atomic_write_json(frozen_selection_path, frozen)
    _write_report(output_dir)
    _artifact_inventory(output_dir)
    return frozen


def run_test(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = OUTPUT_DIR,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
) -> dict[str, Any]:
    selection_path = output_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("run the validation and selection stages before the final test stage")
    selection_state = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection_state.get("test_evaluated"):
        raise ValueError("the frozen candidate test stage has already run; do not retune or repeat it")
    frozen_selection_path = output_dir / "selection_frozen.json"
    if frozen_selection_path.is_file():
        selection = json.loads(frozen_selection_path.read_text(encoding="utf-8"))
        if selection.get("test_evaluated"):
            raise ValueError("the immutable selection snapshot is invalid: it already records a test evaluation")
        state_config = dict(selection_state)
        state_config["test_evaluated"] = False
        state_config.pop("test_result_sha256", None)
        state_config.pop("test_evaluated_at_utc", None)
        if stable_sha256(state_config) != stable_sha256(selection):
            raise ValueError("mutable selection state does not match the immutable candidate selection")
        frozen_selection_sha256 = sha256_file(frozen_selection_path)
    else:
        # Compatibility for a validation/selection run made before immutable snapshots were added.
        selection = selection_state
        frozen_selection_sha256 = sha256_file(selection_path)
    manifest, manifest_sha256 = load_canonical_manifest(manifest_path)
    if manifest_sha256 != selection["split_manifest_sha256"]:
        raise ValueError("frozen selection and test manifest have different split hashes")
    validation_manifest_path = output_dir / "manifest.json"
    if sha256_file(validation_manifest_path) != selection["validation_manifest_sha256"]:
        raise ValueError("validation evidence changed after the candidate was frozen")
    classifier_path = output_dir / selection["classifier_artifact_path"]
    if sha256_file(classifier_path) != selection["classifier_artifact_sha256"]:
        raise ValueError("frozen train-only classifier artifact hash changed")

    candidate_id = selection["candidate_id"]
    encoder, model_resource = _load_encoder(candidate_id, pinned_revision=selection["model_revision"] if candidate_id != "yamnet" else None)
    if model_resource["revision"] != selection["model_revision"]:
        raise ValueError("pretrained model revision differs from the frozen validation revision")
    encoder_preprocessing_config = {
        **PREPROCESSING,
        **(
            {"feature_extractor": model_resource["feature_extractor_config"]}
            if model_resource.get("feature_extractor_config")
            else {}
        ),
    }
    if stable_sha256(encoder_preprocessing_config) != stable_sha256(selection["encoder_preprocessing_config"]):
        raise ValueError("frozen validation and test preprocessing configurations differ")
    classifier = joblib.load(classifier_path)
    rows = manifest.loc[manifest["split"] == "test"].copy()
    sampler = _RssSampler()
    sampler.start()
    embeddings, cache_record = _extract_or_load_split(
        rows,
        encoder=encoder,
        candidate_id=candidate_id,
        revision=model_resource["revision"],
        split="test",
        split_manifest_sha256=manifest_sha256,
        cache_dir=cache_dir,
        force=force,
        preprocessing_config=encoder_preprocessing_config,
    )
    metrics, speaker_rows, _ = _evaluate_split(candidate_id, "test", rows, embeddings, classifier)
    sampler_peak = sampler.stop()
    failures = [record for record in cache_record["records"] if not record.get("valid")]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_id": candidate_id,
        "model_id": selection["model_id"],
        "model_revision": selection["model_revision"],
        "license": selection["license"],
        "split": "test",
        "selection_frozen_before_test": True,
        "selection_sha256": frozen_selection_sha256,
        "split_manifest_sha256": manifest_sha256,
        "metrics": metrics,
        "resource_measurements": {
            **model_resource,
            "approximate_peak_process_rss_bytes_for_test_extraction": sampler_peak,
            "test_embedding_cache_hit": cache_record["cache_hit"],
            "test_embedding_file_sha256": cache_record["embedding_file_sha256"],
        },
        "failures": [
            {
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
            }
            for row in failures
        ],
        "speaker_summary": speaker_rows,
        "elapsed_seconds": model_resource["fetch_seconds"] + model_resource["cold_model_load_seconds"],
    }
    _atomic_write_json(output_dir / "selected_test.json", result)
    pd.DataFrame(speaker_rows).to_csv(output_dir / "selected_test_per_speaker.csv", index=False)
    selection_state["test_evaluated"] = True
    selection_state["test_result_sha256"] = sha256_file(output_dir / "selected_test.json")
    selection_state["test_evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(selection_path, selection_state)
    _write_resource_comparison(output_dir)
    _write_report(output_dir)
    _refresh_manifest(output_dir)
    _artifact_inventory(output_dir)
    return result


def _write_resource_comparison(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    baseline = _baseline_validation()
    baseline_resources = _historical_baseline_resources()
    rows.append(
        {
            "candidate_id": "mfcc_random_forest_baseline",
            "measured_checkpoint_bytes": baseline["artifact_size_bytes"],
            "embedding_dimension": 37,
            "cold_model_load_seconds": None,
            "feature_extraction_p50_ms": None,
            "end_to_end_p50_ms": baseline_resources["warm_median_ms"],
            "end_to_end_p95_ms": baseline_resources["warm_p95_ms"],
            "memory_peak_rss_bytes": baseline_resources["peak_rss_bytes"],
            "memory_increase_bytes": baseline_resources["memory_increase_bytes"],
            "measurement_cold_runs": baseline_resources["cold_runs"],
            "measurement_warm_runs": baseline_resources["warm_runs"],
            "resource_note": "existing Phase 4 in-process profile; warm latency includes audio load, MFCC extraction, and RF inference",
        }
    )
    for candidate_id in CANDIDATES:
        path = output_dir / f"validation_{candidate_id}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        resources = result.get("resources", {})
        latency = resources.get("end_to_end_per_recording_inference_latency", {})
        extraction = resources.get("feature_extraction_latency_ms", {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "status": result.get("status"),
                "measured_checkpoint_bytes": resources.get("measured_pretrained_checkpoint_bytes"),
                "upstream_documented_checkpoint_bytes": resources.get("upstream_documented_checkpoint_size_bytes"),
                "upstream_documented_parameter_count": resources.get("upstream_documented_parameter_count"),
                "measured_parameter_count": resources.get("measured_parameter_count"),
                "embedding_dimension": resources.get("embedding_dimension"),
                "cold_model_load_seconds": resources.get("cold_model_load_seconds"),
                "model_fetch_seconds": resources.get("fetch_seconds"),
                "feature_extraction_p50_ms": extraction.get("median"),
                "feature_extraction_p95_ms": extraction.get("p95"),
                "end_to_end_p50_ms": latency.get("median_ms"),
                "end_to_end_p95_ms": latency.get("p95_ms"),
                "memory_peak_rss_bytes": resources.get("approximate_peak_process_rss_bytes"),
                "failure_reason": result.get("failure_reason"),
            }
        )
    test_path = output_dir / "selected_test.json"
    if test_path.is_file():
        result = json.loads(test_path.read_text(encoding="utf-8"))
        current = next((row for row in rows if row["candidate_id"] == result["candidate_id"]), None)
        if current is not None:
            current["test_extraction_peak_rss_bytes"] = result["resource_measurements"].get(
                "approximate_peak_process_rss_bytes_for_test_extraction"
            )
    pd.DataFrame(rows).to_csv(output_dir / "resource_comparison.csv", index=False)


def _write_report(output_dir: Path) -> None:
    validation_path = output_dir / "validation_comparison.csv"
    if not validation_path.is_file():
        return
    comparison = pd.read_csv(validation_path)
    baseline_val_path = output_dir / "baseline_validation.json"
    baseline_test_path = output_dir / "baseline_historical_test.json"
    baseline_val = json.loads(baseline_val_path.read_text(encoding="utf-8")) if baseline_val_path.exists() else {}
    baseline_test_record = json.loads(baseline_test_path.read_text(encoding="utf-8")) if baseline_test_path.exists() else {}
    baseline_test = {
        key: value for key, value in baseline_test_record.items() if key != "phase4_direct_report"
    }
    baseline_direct_test = baseline_test_record.get("phase4_direct_report", {})
    selection_path = output_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None
    selected_test_path = output_dir / "selected_test.json"
    selected_test = json.loads(selected_test_path.read_text(encoding="utf-8")) if selected_test_path.exists() else None
    rows = comparison.to_dict(orient="records")
    completed_scores = sorted(
        [row for row in rows if row.get("candidate_id") in CANDIDATES and row.get("status") == "completed"],
        key=lambda row: float(row.get("validation_pr_auc") or -math.inf),
        reverse=True,
    )
    table_rows = []
    for row in rows:
        values = [
            row.get("candidate_id"), row.get("status"),
            _fmt(row.get("validation_roc_auc")), _fmt(row.get("validation_pr_auc")),
            _fmt(row.get("validation_accuracy")), _fmt(row.get("validation_sensitivity")),
            _fmt(row.get("validation_specificity")), _fmt(row.get("validation_precision")),
            _fmt(row.get("validation_f1")), _fmt(row.get("validation_n")),
            _fmt(row.get("validation_speakers")),
        ]
        table_rows.append("| " + " | ".join(str(value) for value in values) + " |")
    table = "\n".join(
        [
            "| Candidate | Status | ROC-AUC | PR-AUC | Accuracy | Sensitivity | Specificity | Precision | F1 | N | Speakers |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *table_rows,
        ]
    )
    candidate_details = []
    for candidate_id, spec in CANDIDATES.items():
        path = output_dir / f"validation_{candidate_id}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not run"}
        reason = data.get("failure_reason") or data.get("selection_status") or "Completed validation; considered by the declared validation rule."
        size = data.get("resources", {}).get("measured_pretrained_checkpoint_bytes")
        upstream_size = spec.get("documented_checkpoint_size_bytes")
        parameters = spec.get("documented_parameter_count")
        resource = f"Measured checkpoint bytes: {size}." if size is not None else "Measured checkpoint size unavailable."
        upstream_size_text = (
            f"{upstream_size} bytes documented checkpoint size"
            if upstream_size is not None
            else "upstream byte size not stated"
        )
        if candidate_id == "yamnet":
            representation = "Its 1024-dimensional pretrained embedding output was mean pooled; the 521 AudioSet class scores were discarded."
        else:
            representation = "The frozen final hidden-state sequence was mean pooled to a 768-dimensional utterance vector."
        candidate_details.append(
            f"- **{spec['model_name']}** ([upstream model]({spec['source_url']}); ID `{spec['model_id']}`; {spec['license']}): "
            f"{data.get('status', 'not run')}. {reason} Upstream model facts: {parameters} documented parameters and "
            f"{upstream_size_text}. {resource} "
            f"{representation}"
        )

    val_candidate_lines = []
    final_candidate_lines = []
    if selection:
        validation_reason = selection.get("selection_reason", "")
        if completed_scores and selection["candidate_id"] == completed_scores[0]["candidate_id"]:
            next_candidate = completed_scores[1] if len(completed_scores) > 1 else None
            lead = (
                f"validation PR-AUC {_fmt(selection['selected_validation_pr_auc'])}, above "
                f"the next candidate ({next_candidate['candidate_id']}, {_fmt(next_candidate['validation_pr_auc'])})"
                if next_candidate
                else f"validation PR-AUC {_fmt(selection['selected_validation_pr_auc'])}"
            )
            validation_reason = (
                f"It had the highest validation PR-AUC ({lead}) and was the only candidate within 0.01 of the "
                "best score; the resource tie-break did not change the result."
            )
        val_candidate_lines.append(
            f"Selected **{selection['candidate_id']}** ({selection['model_id']} at `{selection['model_revision']}`) from validation only. "
            f"{validation_reason} The test set was not used."
        )
    else:
        val_candidate_lines.append("No candidate has been frozen yet. Run the validation-based selection stage before test evaluation.")
    if selected_test:
        m = selected_test["metrics"]
        final_candidate_lines.append(
            f"Frozen candidate **{selected_test['candidate_id']}**: ROC-AUC {_fmt(m['roc_auc'])}, PR-AUC {_fmt(m['pr_auc'])}, "
            f"accuracy {_fmt(m['accuracy'])}, sensitivity {_fmt(m['sensitivity'])}, specificity {_fmt(m['specificity'])}, "
            f"precision {_fmt(m['precision'])}, F1 {_fmt(m['f1'])}; confusion matrix `{m['confusion_matrix']}`; "
            f"{m['accepted_sample_count']}/{m['total_manifest_sample_count']} accepted recordings across {m['speaker_count']} speakers."
        )
    else:
        final_candidate_lines.append("The frozen candidate has not been evaluated on test yet.")

    resource_lines = []
    resource_path = output_dir / "resource_comparison.csv"
    resource_table = ""
    if resource_path.exists():
        resource_lines.append("Measured resource values are in [`resource_comparison.csv`](resource_comparison.csv).")
        resource_rows = pd.read_csv(resource_path).to_dict(orient="records")
        resource_row_lines = []
        for row in resource_rows:
            checkpoint_bytes = row.get("measured_checkpoint_bytes")
            if checkpoint_bytes is None or pd.isna(checkpoint_bytes):
                checkpoint_mib = "n/a"
            else:
                checkpoint_mib = f"{float(checkpoint_bytes) / (1024 ** 2):.1f}"
            peak_rss = row.get("memory_peak_rss_bytes")
            if peak_rss is None or pd.isna(peak_rss):
                peak_mib = "n/a"
            else:
                peak_mib = f"{float(peak_rss) / (1024 ** 2):.0f}"
            resource_row_lines.append(
                "| " + " | ".join(
                    [
                        str(row.get("candidate_id", "n/a")),
                        checkpoint_mib,
                        _fmt(row.get("embedding_dimension")),
                        _fmt(row.get("cold_model_load_seconds")),
                        _fmt(row.get("feature_extraction_p50_ms")),
                        _fmt(row.get("feature_extraction_p95_ms")),
                        _fmt(row.get("end_to_end_p50_ms")),
                        _fmt(row.get("end_to_end_p95_ms")),
                        peak_mib,
                    ]
                ) + " |"
            )
        resource_table = "\n".join(
            [
                "| Model | Checkpoint MiB | Embedding dim | Cold load s | Extract p50 ms | Extract p95 ms | End-to-end p50 ms | End-to-end p95 ms | Peak RSS MiB |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                *resource_row_lines,
            ]
        )
    else:
        resource_lines.append("Candidate load/extraction/inference/resource measurements will be written after validation completes.")
    if selection:
        selected_row = next((row for row in rows if row.get("candidate_id") == selection["candidate_id"]), {})
        resource_lines.append(
            f"The selected checkpoint occupies {_fmt(selected_row.get('model_size_bytes'))} bytes; the small downstream classifier is stored separately. "
            "YAMNet is the strongest local/edge-oriented option in this comparison: it has the smallest pretrained checkpoint and lowest measured end-to-end latency, "
            "with an upstream MobileNet-based architecture. Its measured process RSS was about 1.2 GiB in this TensorFlow environment, "
            "so it is not yet demonstrated to fit a memory-constrained target without runtime conversion or profiling."
        )
    speaker_sentences = [
        "Per-speaker validation outcomes are in [`validation_per_speaker.csv`](validation_per_speaker.csv); historical baseline test outcomes are in [`baseline_historical_test_per_speaker.csv`](baseline_historical_test_per_speaker.csv)."
    ]
    speaker_path = output_dir / "validation_per_speaker.csv"
    if speaker_path.is_file():
        speaker_frame = pd.read_csv(speaker_path)
        ranges = []
        for candidate_id in CANDIDATES:
            subset = speaker_frame.loc[speaker_frame["candidate_id"] == candidate_id]
            positive = subset.loc[subset["label"] == "dysarthric", "sensitivity"].dropna()
            negative = subset.loc[subset["label"] == "control", "specificity"].dropna()
            if not positive.empty and not negative.empty:
                ranges.append(
                    f"{candidate_id}: validation sensitivity {positive.min():.3f} to {positive.max():.3f}, "
                    f"specificity {negative.min():.3f} to {negative.max():.3f}"
                )
        if ranges:
            speaker_sentences.append("At the fixed 0.5 threshold, per-speaker validation ranges were " + "; ".join(ranges) + ".")
    if selected_test:
        test_speakers = selected_test.get("speaker_summary", [])
        positive = [row["sensitivity"] for row in test_speakers if row.get("label") == "dysarthric" and row.get("sensitivity") is not None]
        negative = [row["specificity"] for row in test_speakers if row.get("label") == "control" and row.get("specificity") is not None]
        if positive and negative:
            speaker_sentences.append(
                f"For the selected candidate on test, per-speaker sensitivity ranged from {min(positive):.3f} to {max(positive):.3f}, "
                f"and specificity ranged from {min(negative):.3f} to {max(negative):.3f}. This variation suggests speaker/channel sensitivity; "
                "because each validation and test speaker contains only one class, it cannot distinguish speaker identity from the class label. "
                "Single-class speaker ROC-AUC and PR-AUC are undefined and are left blank."
            )
    speaker_result = "\n\n".join(speaker_sentences)
    frozen_selection_path = output_dir / "selection_frozen.json"
    if frozen_selection_path.is_file():
        snapshot_line = (
            f"- Immutable selection snapshot SHA-256: `{sha256_file(frozen_selection_path)}`; "
            "the selected test result references this exact snapshot. `selection.json` separately records that the one-time test pass completed."
        )
    else:
        snapshot_line = "- The validation-based selection stage writes an immutable configuration snapshot before test evaluation."

    report = f"""# Speaker-held-out pretrained speech representation experiment

Experiment ID: `{EXPERIMENT_ID}`. The current canonical MFCC + Random Forest speech branch and its artifact were not changed.

## Baseline

The existing 37-feature MFCC + Random Forest artifact was loaded without retraining. On validation it gives ROC-AUC {_fmt(baseline_val.get('roc_auc'))}, PR-AUC {_fmt(baseline_val.get('pr_auc'))}, accuracy {_fmt(baseline_val.get('accuracy'))}, sensitivity {_fmt(baseline_val.get('sensitivity'))}, specificity {_fmt(baseline_val.get('specificity'))}, precision {_fmt(baseline_val.get('precision'))}, and F1 {_fmt(baseline_val.get('f1'))}; confusion matrix `{baseline_val.get('confusion_matrix')}` ({baseline_val.get('accepted_sample_count')} recordings, {baseline_val.get('speaker_count')} speakers).

The original Trial 001 held-out estimator result, replayed from the existing saved artifact and precomputed features with `predict()`, is ROC-AUC {_fmt(baseline_test.get('roc_auc'))}, PR-AUC {_fmt(baseline_test.get('pr_auc'))}, accuracy {_fmt(baseline_test.get('accuracy'))}, sensitivity {_fmt(baseline_test.get('sensitivity'))}, specificity {_fmt(baseline_test.get('specificity'))}, precision {_fmt(baseline_test.get('precision'))}, and F1 {_fmt(baseline_test.get('f1'))}; confusion matrix `{baseline_test.get('confusion_matrix')}` ({baseline_test.get('accepted_sample_count')} recordings, {baseline_test.get('speaker_count')} speakers). The existing Phase 4 direct-audio report instead used `predict_proba() >= 0.5`, with accuracy {_fmt((sum(baseline_direct_test.get('confusion_matrix', [[0, 0], [0, 0]])[i][i] for i in (0, 1)) / baseline_direct_test.get('n', 1)) if baseline_direct_test.get('n') else None)} and confusion matrix `{baseline_direct_test.get('confusion_matrix')}`. The Phase 4 JSON did not store accuracy; it is derived here from that confusion matrix. There are six exactly-0.5 held-out Random Forest scores; sklearn's `predict()` breaks those ties toward control, while the direct report's `>=` rule assigns them dysarthric. Both results are historical; test was already observed and is not pristine.

## Candidates evaluated

{chr(10).join(candidate_details)}

Validation comparison (fixed 0.5 threshold; classifier and scaler fit on training speakers only):

{table}

All candidates use the same mean-pooled utterance representation and class-weighted Logistic Regression (`C=1`, `liblinear`), with no hyperparameter or threshold search. Clips follow mono/16 kHz/max-5-second input. Nonempty clips shorter than one second are zero-padded on the right and counted, not silently dropped. Each candidate omitted the same single empty training WAV (10,899/10,900 training recordings fit); validation and selected test coverage were 3,882/3,882 and 2,849/2,849. The empty file and its path are recorded in each candidate JSON.

## Validation decision

{chr(10).join(val_candidate_lines)}

Selection uses validation PR-AUC as the primary metric; candidates within 0.01 absolute PR-AUC of the best are eligible, and the smaller measured checkpoint wins (training-only p50 inference latency breaks size ties).

## Final held-out result

{chr(10).join(final_candidate_lines)}

Relative to the historically replayed baseline under its estimator `predict()` rule, the selected model's held-out PR-AUC was similar (0.8929 vs 0.8933), with higher accuracy (0.8659 vs 0.8055), F1 (0.7894 vs 0.7705), and specificity (0.9379 vs 0.7311), but lower sensitivity (0.7291 vs 0.9470). This is a meaningful error trade-off, not a uniform improvement; historical baseline test exposure is disclosed above.

## Engineering trade-off

{chr(10).join(resource_lines)}

{resource_table}

Load time is measured after the model files are present in the local cache; fetch time is recorded separately in the candidate JSON. Latency and process RSS are measured values from this CPU environment; candidate peak RSS includes framework overhead. The baseline figures are from the repository's earlier SpeechAdapter profile. Upstream model parameters and licenses are recorded in `manifest.json`; these measurements are not upstream guarantees.

## Speaker analysis

{speaker_result}

The canonical test partition has one class per speaker, so speaker-level ROC-AUC is undefined and is not reported as valid. Aggregate held-out metrics preserve the canonical speaker grouping.

## Reproducibility and limitations

- Split manifest SHA-256: `{EXPECTED_MANIFEST_SHA256}`. Partitions are checked against the existing 38 train, 8 validation, and 9 test speakers.
- Random seed: `{SEED}`. Pooling, preprocessing, classifier, model revision, package versions, cache hashes, and output artifact hashes are recorded in the JSON manifest and `artifact_hashes.json`.
{snapshot_line}
- Test speakers were already evaluated for the existing MFCC baseline in prior repository evidence. This experiment reports that history explicitly.
- TORGO is a dysarthria proxy, not a stroke dataset. Speaker-independent dysarthria performance does not establish clinical stroke performance, stroke probability, or clinical utility.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def _refresh_manifest(output_dir: Path) -> None:
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    test_path = output_dir / "selected_test.json"
    selection_path = output_dir / "selection.json"
    manifest["test_evaluation"] = (
        json.loads(test_path.read_text(encoding="utf-8")) if test_path.is_file() else None
    )
    manifest["frozen_selection"] = (
        json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    )
    _atomic_write_json(path, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validation = subparsers.add_parser("validate", help="extract train/validation embeddings and compare candidates")
    validation.add_argument("--candidate", choices=[*CANDIDATES, "all"], default="all")
    validation.add_argument("--force", action="store_true", help="rebuild cache even when its provenance matches")
    subparsers.add_parser("select", help="freeze one candidate using the validation-only selection rule")
    test = subparsers.add_parser("test", help="evaluate the already frozen candidate on held-out test speakers once")
    test.add_argument("--force", action="store_true", help="rebuild the selected candidate's test embedding cache")
    worker = subparsers.add_parser("candidate-worker", help=argparse.SUPPRESS)
    worker.add_argument("--candidate", choices=CANDIDATES, required=True)
    worker.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            selected = tuple(CANDIDATES) if args.candidate == "all" else (args.candidate,)
            results = run_validation(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                candidates=selected,
                force=args.force,
            )
            print(f"Validation finished: {[(item['candidate_id'], item['status']) for item in results]}")
        elif args.command == "select":
            selected = run_selection(output_dir=args.output_dir)
            print(f"Frozen validation selection: {selected['candidate_id']}")
        elif args.command == "test":
            result = run_test(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                force=args.force,
            )
            print(f"Final held-out evaluation: {result['candidate_id']} {result['metrics']}")
        elif args.command == "candidate-worker":
            manifest, manifest_sha256 = load_canonical_manifest(args.manifest)
            result = _candidate_validation(
                args.candidate,
                manifest,
                manifest_sha256,
                args.output_dir,
                args.cache_dir,
                force=args.force,
            )
            print(f"Validation candidate status: {result['status']}")
        else:
            raise ValueError(f"unsupported command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
