"""Fusion strategy abstraction and canonical late-fusion adapter."""

from __future__ import annotations

import math
from typing import Mapping, Protocol

from rural_stroke_assist.assessment.contracts import FusionEvidence
from rural_stroke_assist.inference.contracts import ModalityEvidence
from rural_stroke_assist.inference.registry import BaselineRegistry, load_baseline_registry
from rural_stroke_assist.modules.fusion_module import FusionInput, fuse_multimodal_scores


class FusionStrategy(Protocol):
    def fuse(self, evidence_map: Mapping[str, ModalityEvidence]) -> FusionEvidence:
        ...


class CanonicalLateFusionStrategy:
    def __init__(self, *, registry: BaselineRegistry | None = None) -> None:
        self.registry = registry or load_baseline_registry()

    def fuse(self, evidence_map: Mapping[str, ModalityEvidence]) -> FusionEvidence:
        usable = {name: item for name, item in evidence_map.items() if item.available}
        if not usable:
            raise ValueError("At least one usable modality is required for fusion.")
        if any(item.score is None or not math.isfinite(item.score) or not 0 <= item.score <= 1 for item in usable.values()):
            raise ValueError("Fusion received an invalid modality score.")
        symptom = usable.get("acute_symptoms")
        fusion_input = FusionInput(
            face_acute_score=usable.get("face").score if usable.get("face") else None,
            speech_acute_score=usable.get("speech").score if usable.get("speech") else None,
            acute_symptom_score=symptom.score if symptom else None,
            metadata_contextual_risk_score=usable.get("metadata_context").score if usable.get("metadata_context") else None,
            acute_symptom_hard_escalation=bool(symptom and symptom.details.get("hard_escalation", False)),
        )
        config = self.registry.fusion
        result = fuse_multimodal_scores(
            fusion_input,
            weights=dict(config["weights"]),
            moderate_threshold=float(config["moderate_threshold"]),
            high_threshold=float(config["high_threshold"]),
        )
        return FusionEvidence(
            evidence_score=float(result.fused_score),
            risk_band=result.risk_band,
            modality_contributions=dict(result.modality_contributions),
            normalized_weights_used=dict(result.normalized_weights_used),
            warnings=tuple(result.warnings),
            evidence_notes=tuple(result.evidence),
            provenance=self.registry.component("fusion").path,
        )
