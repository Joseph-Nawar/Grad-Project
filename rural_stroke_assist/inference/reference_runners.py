"""Pinned pretrained runners for the explicit research/reference profile."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from rural_stroke_assist.inference.exceptions import ArtifactConfigurationError, FeatureContractError
from rural_stroke_assist.inference.reference_registry import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    ReferenceProfileRegistry,
)


class DistilHuBERTRunner:
    """DistilHuBERT final-hidden-state mean pooling plus frozen LR classifier."""

    def __init__(
        self,
        registry: ReferenceProfileRegistry,
        *,
        snapshot_path: Path | None = None,
        encoder_loader: Callable[[Path], tuple[Any, Any, Any]] | None = None,
        classifier_loader: Callable[[Path], Any] | None = None,
        load_on_init: bool = True,
    ) -> None:
        self.registry = registry
        self._snapshot_path = snapshot_path
        self._encoder_loader = encoder_loader or self._default_encoder_loader
        self._classifier_loader = classifier_loader or self._default_classifier_loader
        self._feature_extractor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._classifier: Any | None = None
        if load_on_init:
            self._load()

    @staticmethod
    def _default_encoder_loader(snapshot_path: Path) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel

            torch.set_num_threads(max(1, min(4, (os.cpu_count() or 1) // 2)))
            torch.use_deterministic_algorithms(True, warn_only=True)
            extractor = AutoFeatureExtractor.from_pretrained(
                str(snapshot_path), local_files_only=True
            )
            model = AutoModel.from_pretrained(
                str(snapshot_path), local_files_only=True, low_cpu_mem_usage=True
            )
            model.eval()
            return extractor, model, torch
        except Exception as exc:
            raise ArtifactConfigurationError("Unable to load pinned local DistilHuBERT resources.") from exc

    @staticmethod
    def _default_classifier_loader(path: Path) -> Any:
        try:
            import joblib

            return joblib.load(path)
        except Exception as exc:
            raise ArtifactConfigurationError(
                f"Unable to load the selected DistilHuBERT classifier: {path}"
            ) from exc

    def _load(self) -> None:
        if self._classifier is not None:
            return
        self.registry.validate_project_artifacts()
        snapshot = self._snapshot_path or self.registry.resolve_speech_snapshot(local_only=True)
        if self._snapshot_path is not None and hasattr(self.registry, "validate_speech_snapshot"):
            self.registry.validate_speech_snapshot(snapshot)
        try:
            extractor, model, torch = self._encoder_loader(snapshot)
            classifier = self._classifier_loader(self.registry.path_for("speech"))
        except ArtifactConfigurationError:
            raise
        except Exception as exc:
            raise ArtifactConfigurationError("Unable to construct the pinned DistilHuBERT runner.") from exc
        self._snapshot_path = snapshot
        self._feature_extractor = extractor
        self._model = model
        self._torch = torch
        self._classifier = classifier

    def run(self, signal: np.ndarray) -> tuple[np.ndarray, tuple[Any, ...]]:
        self._load()
        if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
            raise FeatureContractError("Reference speech runner requires a finite non-empty mono signal.")
        try:
            inputs = self._feature_extractor(
                signal.astype(np.float32, copy=False),
                sampling_rate=16_000,
                return_tensors="pt",
                padding=False,
            )
            with self._torch.inference_mode():
                output = self._model(input_values=inputs["input_values"])
            frame_embeddings = output.last_hidden_state[0].detach().cpu().numpy().astype(np.float32, copy=False)
            if frame_embeddings.ndim != 2 or not len(frame_embeddings):
                raise FeatureContractError(
                    f"DistilHuBERT returned an invalid hidden-state shape: {frame_embeddings.shape}"
                )
            embedding = frame_embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
            probabilities = np.asarray(self._classifier.predict_proba(embedding.reshape(1, -1)), dtype=float)
            classes = tuple(self._classifier.classes_)
            return probabilities, classes
        except FeatureContractError:
            raise
        except Exception as exc:
            raise ArtifactConfigurationError("Pinned DistilHuBERT inference failed.") from exc


class TabPFNRunner:
    """TabPFN V2 in-context runner using the exact frozen training context."""

    def __init__(
        self,
        registry: ReferenceProfileRegistry,
        *,
        checkpoint_path: Path | None = None,
        model_factory: Callable[[Path, dict[str, object]], Any] | None = None,
        context_loader: Callable[[Path], pd.DataFrame] | None = None,
        load_on_init: bool = True,
    ) -> None:
        self.registry = registry
        self._checkpoint_path = checkpoint_path
        self._model_factory = model_factory or self._default_model_factory
        self._context_loader = context_loader or self._default_context_loader
        self._model: Any | None = None
        if load_on_init:
            self._load()

    @staticmethod
    def _default_context_loader(path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path,
            dtype={
                **{name: "float64" for name in NUMERIC_FEATURES},
                **{name: "str" for name in CATEGORICAL_FEATURES},
                "stroke": "int64",
            },
        )

    @staticmethod
    def _default_model_factory(checkpoint_path: Path, settings: dict[str, object]) -> Any:
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion

            return TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                model_path=str(checkpoint_path),
                categorical_features_indices=list(settings["categorical_feature_indices"]),
                device="cpu",
                random_state=42,
                ignore_pretraining_limits=True,
            )
        except Exception as exc:
            raise ArtifactConfigurationError("Unable to construct the pinned TabPFN V2 runner.") from exc

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.loc[:, FEATURE_COLUMNS].copy()
        for name in NUMERIC_FEATURES:
            result[name] = pd.to_numeric(result[name], errors="coerce")
        for name in CATEGORICAL_FEATURES:
            result[name] = result[name].astype(object)
        return result

    def _load(self) -> None:
        if self._model is not None:
            return
        self.registry.validate_project_artifacts()
        checkpoint = self._checkpoint_path or self.registry.resolve_tabpfn_checkpoint()
        if self._checkpoint_path is not None and hasattr(self.registry, "validate_tabpfn_checkpoint"):
            self.registry.validate_tabpfn_checkpoint(checkpoint)
        metadata = self.registry.component("metadata_context").data
        context_path = self.registry.path_for("metadata_context")
        try:
            context = self._context_loader(context_path)
            if list(context.columns) != FEATURE_COLUMNS + ["stroke"]:
                raise ArtifactConfigurationError("TabPFN context columns do not match the frozen contract.")
            context_with_split = context.copy()
            context_with_split["split"] = pd.Series("train", index=context.index, dtype="str")
            from rural_stroke_assist.inference.reference_registry import training_context_fingerprint

            fingerprint = training_context_fingerprint(context_with_split)
            if fingerprint["sha256"] != str(metadata["training_context_fingerprint"]):
                raise ArtifactConfigurationError("TabPFN training context fingerprint does not match the frozen selection.")
            model = self._model_factory(
                checkpoint,
                {
                    "model_version": str(metadata["model_version"]),
                    "categorical_feature_indices": list(metadata["categorical_feature_indices"]),
                    "ignore_pretraining_limits": True,
                },
            )
            model.fit(self._prepare_frame(context), context["stroke"].astype(int).to_numpy())
        except ArtifactConfigurationError:
            raise
        except Exception as exc:
            raise ArtifactConfigurationError("Unable to load or fit the frozen TabPFN context.") from exc
        self._checkpoint_path = checkpoint
        self._model = model

    def run(self, frame: pd.DataFrame) -> tuple[np.ndarray, tuple[Any, ...]]:
        self._load()
        if list(frame.columns) != FEATURE_COLUMNS:
            raise FeatureContractError("Reference metadata frame does not match the canonical ten-feature order.")
        try:
            probabilities = np.asarray(self._model.predict_proba(self._prepare_frame(frame)), dtype=float)
            classes = tuple(self._model.classes_)
            return probabilities, classes
        except Exception as exc:
            raise ArtifactConfigurationError("Pinned TabPFN V2 inference failed.") from exc
