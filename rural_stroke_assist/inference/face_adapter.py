"""Artifact-backed face inference adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

from rural_stroke_assist.inference.contracts import ModalityEvidence, QualityFinding, QualityStatus
from rural_stroke_assist.inference.exceptions import ArtifactConfigurationError, FeatureContractError, InferenceFailure
from rural_stroke_assist.inference.registry import BaselineRegistry, load_baseline_registry
from rural_stroke_assist.inference.runners import FaceRunner
from rural_stroke_assist.modeling.face_inference import preprocess_face_image
from rural_stroke_assist.quality.face_quality import FaceQualityAssessor, OpenCVFaceQualityAssessor


class FaceAdapter:
    def __init__(
        self,
        *,
        registry: BaselineRegistry | None = None,
        model_loader: Callable[[Path], Any] | None = None,
        runner: FaceRunner | None = None,
        quality_assessor: FaceQualityAssessor | None = None,
    ) -> None:
        self.registry = registry or load_baseline_registry()
        self._model_loader = model_loader or self._default_model_loader
        self._runner = runner
        self._quality_assessor = quality_assessor or OpenCVFaceQualityAssessor()
        self._model: Any | None = None

    def _default_model_loader(self, path: Path) -> Any:
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(path)
        except Exception as exc:
            raise ArtifactConfigurationError(f"Unable to load face model: {path}") from exc

    def _load_model(self) -> Any:
        if self._model is None:
            path = self.registry.path_for("face")
            if not path.is_file():
                raise ArtifactConfigurationError(f"Face artifact not found: {path}")
            try:
                self._model = self._model_loader(path)
            except ArtifactConfigurationError:
                raise
            except Exception as exc:
                raise ArtifactConfigurationError(f"Unable to load face model: {path}") from exc
        return self._model

    def infer(self, input_data: str | Path | None) -> ModalityEvidence:
        component = self.registry.component("face")
        if input_data is None:
            return ModalityEvidence.unavailable(
                modality="face",
                score_semantics="visual_proxy_evidence",
                provenance=component.path,
                warning="No face image was provided.",
            )

        image_path = Path(input_data)
        if not image_path.is_file():
            return ModalityEvidence.unavailable(
                modality="face", score_semantics="visual_proxy_evidence", provenance=component.path,
                warning=f"Face image was not found: {image_path}", quality_status=QualityStatus.REJECT,
                findings=(QualityFinding("missing_file", "Face image file does not exist.", QualityStatus.REJECT),),
            )

        try:
            with Image.open(image_path) as image:
                decoded = image.convert("RGB")
                decoded.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return ModalityEvidence.unavailable(
                modality="face", score_semantics="visual_proxy_evidence", provenance=component.path,
                warning=f"Face image could not be decoded: {image_path}", quality_status=QualityStatus.REJECT,
                findings=(QualityFinding("decode_failed", str(exc), QualityStatus.REJECT),),
            )

        quality = self._quality_assessor(decoded)
        if quality.status is QualityStatus.REJECT:
            return ModalityEvidence.unavailable(
                modality="face", score_semantics="visual_proxy_evidence", provenance=component.path,
                warning="Face image was rejected by quality assessment.", quality_status=quality.status,
                findings=quality.findings,
            )

        try:
            batch = preprocess_face_image(image_path, image_size=(160, 160))
        except Exception as exc:
            return ModalityEvidence.unavailable(
                modality="face", score_semantics="visual_proxy_evidence", provenance=component.path,
                warning="Face image preprocessing failed.", quality_status=QualityStatus.REJECT,
                findings=(QualityFinding("preprocessing_failed", str(exc), QualityStatus.REJECT),),
            )

        try:
            if self._runner is not None:
                raw = np.asarray(self._runner.run(batch))
            else:
                model = self._load_model()
                raw = np.asarray(model.predict(batch, verbose=0))
            if raw.size != 1:
                raise FeatureContractError(f"Expected one sigmoid output, got shape {raw.shape}.")
            score = float(raw.reshape(-1)[0])
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise FeatureContractError(f"Face model returned invalid score: {score}")
        except FeatureContractError:
            raise
        except Exception as exc:
            raise InferenceFailure("Face model inference failed.") from exc

        label = "Stroke" if score >= component.thresholds.get("binary_decision", 0.5) else "NonStroke"
        confidence = score if label == "Stroke" else 1.0 - score
        warnings = (
            "Face score is visual proxy evidence from a non-clinical public dataset.",
            "This output is not a calibrated clinical stroke probability.",
        )
        warnings += tuple(item.message for item in quality.findings if item.status is not QualityStatus.NOT_ASSESSED)
        return ModalityEvidence(
            modality="face", available=True, score=score, score_semantics="visual_proxy_evidence",
            label=label, confidence=confidence, quality_status=quality.status,
            quality_findings=quality.findings, warnings=warnings, provenance=component.path,
        )
