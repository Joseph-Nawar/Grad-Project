"""Adapters for the pinned research/reference speech and metadata pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from rural_stroke_assist.features.speech_features import load_audio
from rural_stroke_assist.inference.contracts import ModalityEvidence, QualityFinding, QualityStatus
from rural_stroke_assist.inference.exceptions import FeatureContractError, InferenceFailure
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.inference.reference_registry import ReferenceProfileRegistry
from rural_stroke_assist.inference.reference_runners import (
    DistilHuBERTRunner,
    TabPFNRunner,
)
from rural_stroke_assist.quality.audio_quality import (
    AudioQualityAssessment,
    AudioQualityAssessor,
    DefaultAudioQualityAssessor,
)


def _class_index(classes: Any, mapping: dict[str, Any], target_value: str) -> int:
    targets = [key for key, value in mapping.items() if value == target_value]
    if len(targets) != 1:
        raise FeatureContractError(f"Reference class mapping has no unique {target_value} class.")
    target = targets[0]
    values = list(classes)
    for index, value in enumerate(values):
        if str(value) == target or str(value).lower() == target_value.lower():
            return index
    raise FeatureContractError(f"Reference class {target!r} is absent from {values!r}.")


def _validate_probability_matrix(probabilities: Any, classes: Any, modality: str) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.shape != (1, len(tuple(classes))):
        raise FeatureContractError(f"Unexpected {modality} probability shape: {matrix.shape}")
    if not np.isfinite(matrix).all() or not ((matrix >= 0.0) & (matrix <= 1.0)).all():
        raise FeatureContractError(f"{modality} model returned non-finite or unbounded probabilities.")
    return matrix


class ReferenceSpeechAdapter:
    def __init__(
        self,
        *,
        registry: ReferenceProfileRegistry,
        runner: DistilHuBERTRunner,
        audio_loader: Callable[[str | Path], tuple[np.ndarray, int]] | None = None,
        quality_assessor: AudioQualityAssessor | None = None,
    ) -> None:
        self.registry = registry
        self._runner = runner
        self._audio_loader = audio_loader or load_audio
        self._quality_assessor = quality_assessor or DefaultAudioQualityAssessor()

    def infer(self, input_data: str | Path | None) -> ModalityEvidence:
        component = self.registry.component("speech")
        semantics = "dysarthria_proxy_evidence"
        if input_data is None:
            return ModalityEvidence.unavailable(
                modality="speech", score_semantics=semantics,
                provenance=component.provenance, warning="No speech audio was provided.",
            )
        path = Path(input_data)
        if not path.is_file():
            return ModalityEvidence.unavailable(
                modality="speech", score_semantics=semantics, provenance=component.provenance,
                warning=f"Speech audio was not found: {path}", quality_status=QualityStatus.REJECT,
                findings=(QualityFinding("missing_file", "Speech audio file does not exist.", QualityStatus.REJECT),),
            )
        try:
            signal, sample_rate = self._audio_loader(path)
        except Exception as exc:
            return ModalityEvidence.unavailable(
                modality="speech", score_semantics=semantics, provenance=component.provenance,
                warning=f"Speech audio could not be decoded: {path}", quality_status=QualityStatus.REJECT,
                findings=(QualityFinding("decode_failed", str(exc), QualityStatus.REJECT),),
            )
        signal = np.asarray(signal, dtype=np.float32)
        quality: AudioQualityAssessment = self._quality_assessor(signal, sample_rate)
        if quality.status is QualityStatus.REJECT:
            return ModalityEvidence.unavailable(
                modality="speech", score_semantics=semantics, provenance=component.provenance,
                warning="Speech audio was rejected by quality assessment.",
                quality_status=quality.status, findings=quality.findings,
            )
        if signal.ndim != 1 or sample_rate != int(component.data.get("sample_rate_hz", 16_000)):
            raise FeatureContractError("Reference speech audio must be mono at the pinned 16 kHz contract.")
        minimum_samples = int(float(component.data.get("minimum_duration_seconds", 1.0)) * sample_rate)
        if len(signal) < minimum_samples:
            signal = np.pad(signal, (0, minimum_samples - len(signal)), mode="constant")
        try:
            probabilities, classes = self._runner.run(signal)
            matrix = _validate_probability_matrix(probabilities, classes, "speech")
            positive_index = _class_index(classes, component.data["class_mapping"], "dysarthric")
            score = float(matrix[0, positive_index])
        except FeatureContractError:
            raise
        except Exception as exc:
            raise InferenceFailure("Reference DistilHuBERT speech inference failed.") from exc
        threshold = float(component.data.get("thresholds", {}).get("classification", 0.5))
        return ModalityEvidence(
            modality="speech", available=True, score=score, score_semantics=semantics,
            label="dysarthric" if score >= threshold else "control",
            confidence=float(np.max(matrix[0])), quality_status=quality.status,
            quality_findings=quality.findings,
            warnings=(
                "Speech score is DistilHuBERT dysarthria proxy evidence from TORGO, not stroke-specific evidence.",
                "This output is not a calibrated clinical stroke probability.",
            ),
            provenance=component.provenance,
            details={
                "profile": "pretrained_reference",
                "upstream_model_id": component.data.get("upstream_model_id", "ntu-spml/distilhubert"),
                "upstream_revision": component.data.get("upstream_revision"),
                "classifier_artifact_sha256": component.data.get("classifier_artifact_sha256"),
                "pooling": component.data.get("pooling", "final hidden-state mean pooling"),
                "score_semantics": semantics,
            },
        )


class ReferenceMetadataAdapter:
    def __init__(self, *, registry: ReferenceProfileRegistry, runner: TabPFNRunner) -> None:
        self.registry = registry
        self._runner = runner

    def infer(self, input_data: MetadataInput | None) -> ModalityEvidence:
        component = self.registry.component("metadata_context")
        semantics = "contextual_risk_evidence"
        if input_data is None:
            return ModalityEvidence.unavailable(
                modality="metadata_context", score_semantics=semantics,
                provenance=component.provenance, warning="No contextual metadata was provided.",
            )
        if not isinstance(input_data, MetadataInput):
            raise FeatureContractError("Metadata input does not match the canonical ten-feature schema.")
        columns = list(component.data["feature_columns"])
        values = input_data.model_dump(by_alias=True)
        if list(values) != columns:
            raise FeatureContractError(f"Metadata schema order mismatch: {list(values)} != {columns}")
        frame = pd.DataFrame([[values[column] for column in columns]], columns=columns)
        try:
            probabilities, classes = self._runner.run(frame)
            matrix = _validate_probability_matrix(probabilities, classes, "metadata")
            positive_index = _class_index(classes, component.data["class_mapping"], "stroke")
            score = float(matrix[0, positive_index])
        except FeatureContractError:
            raise
        except Exception as exc:
            raise InferenceFailure("Reference TabPFN metadata inference failed.") from exc
        threshold = float(component.data.get("thresholds", {}).get("classification", 0.5))
        return ModalityEvidence(
            modality="metadata_context", available=True, score=score,
            score_semantics=semantics, label="stroke" if score >= threshold else "no_stroke",
            confidence=None, quality_status=QualityStatus.PASS,
            warnings=(
                "Metadata output is TabPFN contextual/background risk evidence, not acute stroke probability.",
                "This output is not a calibrated clinical stroke probability.",
            ),
            provenance=component.provenance,
            details={
                "profile": "pretrained_reference",
                "model_version": component.data.get("model_version", "ModelVersion.V2"),
                "package_version": component.data.get("package_version", "9.0.0"),
                "checkpoint_sha256": component.data.get("checkpoint_sha256"),
                "training_context_fingerprint": component.data.get("training_context_fingerprint"),
                "categorical_feature_indices": list(component.data.get("categorical_feature_indices", (5, 6, 7, 8, 9))),
                "score_semantics": semantics,
            },
        )
