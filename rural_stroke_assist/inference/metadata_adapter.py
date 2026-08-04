"""Artifact-backed contextual metadata inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from rural_stroke_assist.inference.contracts import ModalityEvidence, QualityFinding, QualityStatus
from rural_stroke_assist.inference.exceptions import ArtifactConfigurationError, FeatureContractError, InferenceFailure
from rural_stroke_assist.inference.registry import BaselineRegistry, load_baseline_registry


class MetadataInput(BaseModel):
    """Exact ten-feature input contract for the saved metadata pipeline."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    age: float | None = Field(default=None, ge=0, le=120)
    hypertension: int | None = Field(default=None, ge=0, le=1)
    heart_disease: int | None = Field(default=None, ge=0, le=1)
    avg_glucose_level: float | None = Field(default=None, ge=0)
    bmi: float | None = Field(default=None, ge=0)
    gender: Literal["Female", "Male", "Other"] | None = None
    ever_married: Literal["No", "Yes"] | None = None
    work_type: Literal["Govt_job", "Never_worked", "Private", "Self-employed", "children"] | None = None
    residence_type: Literal["Rural", "Urban"] | None = Field(default=None, alias="Residence_type")
    smoking_status: Literal["Unknown", "formerly smoked", "never smoked", "smokes"] | None = None

    def to_model_frame(self, columns: list[str]) -> pd.DataFrame:
        values = self.model_dump(by_alias=True)
        if list(values) != columns:
            raise FeatureContractError(f"Metadata schema order mismatch: {list(values)} != {columns}")
        return pd.DataFrame([[values[column] for column in columns]], columns=columns)


class MetadataAdapter:
    def __init__(
        self,
        *,
        registry: BaselineRegistry | None = None,
        model_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.registry = registry or load_baseline_registry()
        self._model_loader = model_loader or self._default_model_loader
        self._model: Any | None = None

    def _default_model_loader(self, path: Path) -> Any:
        try:
            import joblib
            return joblib.load(path)
        except Exception as exc:
            raise ArtifactConfigurationError(f"Unable to load metadata model: {path}") from exc

    def _load_model(self) -> Any:
        if self._model is None:
            path = self.registry.path_for("metadata_context")
            if not path.is_file():
                raise ArtifactConfigurationError(f"Metadata artifact not found: {path}")
            try:
                self._model = self._model_loader(path)
            except ArtifactConfigurationError:
                raise
            except Exception as exc:
                raise ArtifactConfigurationError(f"Unable to load metadata model: {path}") from exc
        return self._model

    def infer(self, input_data: MetadataInput | None) -> ModalityEvidence:
        component = self.registry.component("metadata_context")
        if input_data is None:
            return ModalityEvidence.unavailable(
                modality="metadata_context", score_semantics="contextual_risk_evidence", provenance=component.path,
                warning="No contextual metadata was provided.",
            )
        if not isinstance(input_data, MetadataInput):
            raise FeatureContractError("Metadata input does not match the canonical ten-feature schema.")
        columns = component.feature_columns
        frame = input_data.to_model_frame(columns)
        model = self._load_model()
        try:
            probabilities = np.asarray(model.predict_proba(frame), dtype=float)
            classes = list(model.classes_)
            mapping = component.data["class_mapping"]
            positive_keys = [key for key, value in mapping.items() if value == "stroke"]
            if len(positive_keys) != 1:
                raise FeatureContractError("Metadata registry has no unique stroke class mapping.")
            positive_index = next((i for i, value in enumerate(classes) if str(value) == positive_keys[0]), None)
            if positive_index is None:
                raise FeatureContractError(f"Metadata stroke class is absent from model classes: {classes!r}")
            if probabilities.shape != (1, len(classes)):
                raise FeatureContractError(f"Unexpected metadata probability shape: {probabilities.shape}")
            score = float(probabilities[0, positive_index])
        except FeatureContractError:
            raise
        except Exception as exc:
            raise InferenceFailure("Metadata model inference failed.") from exc
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise FeatureContractError(f"Metadata model returned invalid score: {score}")

        label = "stroke" if score >= component.thresholds.get("stored_mvp_threshold", 0.5) else "no_stroke"
        return ModalityEvidence(
            modality="metadata_context", available=True, score=score,
            score_semantics="contextual_risk_evidence", label=label, confidence=None,
            quality_status=QualityStatus.PASS, quality_findings=(),
            warnings=(
                "Metadata output is contextual/background risk evidence, not acute stroke probability.",
                "This output is not a calibrated clinical stroke probability.",
            ),
            provenance=component.path,
        )
