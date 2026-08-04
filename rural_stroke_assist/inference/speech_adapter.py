"""Artifact-backed speech inference adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from rural_stroke_assist.features.speech_features import (
    extract_basic_audio_features,
    extract_mfcc_summary_features,
    load_audio,
)
from rural_stroke_assist.inference.contracts import ModalityEvidence, QualityFinding, QualityStatus
from rural_stroke_assist.inference.exceptions import ArtifactConfigurationError, FeatureContractError, InferenceFailure
from rural_stroke_assist.inference.registry import BaselineRegistry, load_baseline_registry
from rural_stroke_assist.quality.audio_quality import AudioQualityAssessor, AudioQualityAssessment, DefaultAudioQualityAssessor


class SpeechAdapter:
    def __init__(
        self,
        *,
        registry: BaselineRegistry | None = None,
        model_loader: Callable[[Path], Any] | None = None,
        audio_loader: Callable[[str | Path], tuple[np.ndarray, int]] | None = None,
        feature_extractor: Callable[[np.ndarray, int], dict[str, float]] | None = None,
        quality_assessor: AudioQualityAssessor | None = None,
    ) -> None:
        self.registry = registry or load_baseline_registry()
        self._model_loader = model_loader or self._default_model_loader
        self._audio_loader = audio_loader or load_audio
        self._feature_extractor = feature_extractor or self._extract_features
        self._quality_assessor = quality_assessor or DefaultAudioQualityAssessor()
        self._model: Any | None = None

    def _default_model_loader(self, path: Path) -> Any:
        try:
            import joblib
            return joblib.load(path)
        except Exception as exc:
            raise ArtifactConfigurationError(f"Unable to load speech model: {path}") from exc

    def _load_model(self) -> Any:
        if self._model is None:
            path = self.registry.path_for("speech")
            if not path.is_file():
                raise ArtifactConfigurationError(f"Speech artifact not found: {path}")
            try:
                self._model = self._model_loader(path)
            except ArtifactConfigurationError:
                raise
            except Exception as exc:
                raise ArtifactConfigurationError(f"Unable to load speech model: {path}") from exc
        return self._model

    @staticmethod
    def _extract_features(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
        features = extract_mfcc_summary_features(signal, sample_rate, n_mfcc=13)
        features.update(extract_basic_audio_features(signal, sample_rate))
        return features

    def _positive_class_index(self, model: Any) -> int:
        expected = self.registry.component("speech").data["class_mapping"]
        positive_keys = [key for key, value in expected.items() if value == "dysarthric"]
        if len(positive_keys) != 1 or not hasattr(model, "classes_"):
            raise FeatureContractError("Speech model does not expose a usable class mapping.")
        target = positive_keys[0]
        classes = list(model.classes_)
        for index, value in enumerate(classes):
            if str(value) == target or str(value).lower() == "dysarthric":
                return index
        raise FeatureContractError(f"Dysarthric class {target!r} is absent from model classes {classes!r}.")

    def infer(self, input_data: str | Path | None) -> ModalityEvidence:
        component = self.registry.component("speech")
        if input_data is None:
            return ModalityEvidence.unavailable(
                modality="speech", score_semantics="dysarthria_proxy_evidence", provenance=component.path,
                warning="No speech audio was provided.",
            )
        path = Path(input_data)
        if not path.is_file():
            return ModalityEvidence.unavailable(
                modality="speech", score_semantics="dysarthria_proxy_evidence", provenance=component.path,
                warning=f"Speech audio was not found: {path}", quality_status=QualityStatus.REJECT,
                findings=(QualityFinding("missing_file", "Speech audio file does not exist.", QualityStatus.REJECT),),
            )
        try:
            signal, sample_rate = self._audio_loader(path)
        except Exception as exc:
            return ModalityEvidence.unavailable(
                modality="speech", score_semantics="dysarthria_proxy_evidence", provenance=component.path,
                warning=f"Speech audio could not be decoded: {path}", quality_status=QualityStatus.REJECT,
                findings=(QualityFinding("decode_failed", str(exc), QualityStatus.REJECT),),
            )

        quality: AudioQualityAssessment = self._quality_assessor(signal, sample_rate)
        if quality.status is QualityStatus.REJECT:
            return ModalityEvidence.unavailable(
                modality="speech", score_semantics="dysarthria_proxy_evidence", provenance=component.path,
                warning="Speech audio was rejected by quality assessment.", quality_status=quality.status,
                findings=quality.findings,
            )

        try:
            features = self._feature_extractor(signal, sample_rate)
        except FeatureContractError:
            raise
        except Exception as exc:
            raise FeatureContractError("Speech feature extraction failed.") from exc
        expected = component.feature_columns
        if list(features) != expected:
            raise FeatureContractError(
                f"Speech feature order mismatch. Expected {len(expected)} columns, got {list(features)}."
            )
        frame = pd.DataFrame([[features[name] for name in expected]], columns=expected)
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise FeatureContractError("Speech features contain non-finite values.")

        model = self._load_model()
        try:
            positive_index = self._positive_class_index(model)
            probabilities = np.asarray(model.predict_proba(frame), dtype=float)
            if probabilities.shape != (1, len(model.classes_)):
                raise FeatureContractError(f"Unexpected speech probability shape: {probabilities.shape}")
            score = float(probabilities[0, positive_index])
        except FeatureContractError:
            raise
        except Exception as exc:
            raise InferenceFailure("Speech model inference failed.") from exc
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise FeatureContractError(f"Speech model returned invalid score: {score}")

        label = "dysarthric" if score >= component.thresholds.get("classification", 0.5) else "control"
        confidence = float(np.max(probabilities[0]))
        warnings = (
            "Speech score is dysarthria proxy evidence from TORGO, not stroke-specific evidence.",
            "This output is not a calibrated clinical stroke probability.",
        )
        return ModalityEvidence(
            modality="speech", available=True, score=score, score_semantics="dysarthria_proxy_evidence",
            label=label, confidence=confidence, quality_status=quality.status,
            quality_findings=quality.findings, warnings=warnings, provenance=component.path,
        )
