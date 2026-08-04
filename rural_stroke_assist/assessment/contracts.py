"""Immutable application contracts for a multimodal assessment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from rural_stroke_assist.inference.contracts import ModalityEvidence, ModalityName


class AssessmentStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class StructuredAdapterFailure:
    modality: ModalityName
    error_type: str
    message: str


@dataclass(frozen=True)
class ModalityExecution:
    modality: ModalityName
    evidence: ModalityEvidence
    duration_ns: int
    failure: StructuredAdapterFailure | None = None


@dataclass(frozen=True)
class FusionEvidence:
    evidence_score: float
    risk_band: str
    modality_contributions: Mapping[str, float]
    normalized_weights_used: Mapping[str, float]
    warnings: tuple[str, ...]
    provenance: str
    evidence_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.evidence_score) or not 0 <= self.evidence_score <= 1:
            raise ValueError("Fusion evidence_score must be finite and in [0, 1].")
        for value in (*self.modality_contributions.values(), *self.normalized_weights_used.values()):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Fusion contributions and weights must be in [0, 1].")


@dataclass(frozen=True)
class ExplanationItem:
    code: str
    message: str


@dataclass(frozen=True)
class AssessmentTimings:
    modality_duration_ns: Mapping[str, int]
    fusion_duration_ns: int
    explanation_duration_ns: int
    total_duration_ns: int


@dataclass(frozen=True)
class AssessmentResult:
    status: AssessmentStatus
    modality_executions: Mapping[ModalityName, ModalityExecution]
    fusion: FusionEvidence | None
    explanations: tuple[ExplanationItem, ...]
    warnings: tuple[str, ...]
    timings: AssessmentTimings
    provenance: tuple[str, ...]
