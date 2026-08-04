"""Shared immutable contracts for modality inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, Sequence, TypeAlias


ModalityName: TypeAlias = Literal["face", "speech", "metadata_context", "acute_symptoms"]


class QualityStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    REJECT = "REJECT"
    NOT_ASSESSED = "NOT_ASSESSED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class QualityFinding:
    code: str
    message: str
    status: QualityStatus


@dataclass(frozen=True)
class ModalityEvidence:
    modality: ModalityName
    available: bool
    score: float | None
    score_semantics: str
    label: str | None
    provenance: str
    confidence: float | None = None
    quality_status: QualityStatus = QualityStatus.PASS
    quality_findings: tuple[QualityFinding, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.available:
            if self.score is None or not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
                raise ValueError("Available evidence requires a finite score in [0, 1].")
        elif self.score is not None:
            raise ValueError("Missing or unusable evidence must have score=None.")

        if self.confidence is not None:
            if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
                raise ValueError("Confidence must be a finite value in [0, 1].")

        if not self.provenance:
            raise ValueError("Evidence provenance is required.")

    @classmethod
    def unavailable(
        cls,
        *,
        modality: ModalityName,
        score_semantics: str,
        provenance: str,
        warning: str,
        quality_status: QualityStatus = QualityStatus.UNAVAILABLE,
        findings: Sequence[QualityFinding] = (),
    ) -> "ModalityEvidence":
        return cls(
            modality=modality,
            available=False,
            score=None,
            score_semantics=score_semantics,
            label=None,
            confidence=None,
            quality_status=quality_status,
            quality_findings=tuple(findings),
            warnings=(warning,),
            provenance=provenance,
        )


class ModalityAdapter(Protocol):
    def infer(self, input_data: object) -> ModalityEvidence:
        """Infer one modality without exposing framework-specific model objects."""
