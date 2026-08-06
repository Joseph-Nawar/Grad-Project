"""Small immutable contracts used by Phase 4 evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Sequence


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially supported"
    UNSUPPORTED = "unsupported"
    PENDING_HUMAN_REVIEW = "pending human review"


@dataclass(frozen=True)
class MetricSummary:
    n: int
    positive: int
    negative: int
    sensitivity: float | None
    specificity: float | None
    precision: float | None
    f1: float | None
    roc_auc: float | None
    pr_auc: float | None
    brier: float | None
    calibration_error: float | None
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class ConfidenceInterval:
    metric: str
    estimate: float | None
    lower: float | None
    upper: float | None
    iterations: int
    seed: int
    unit: str


@dataclass(frozen=True)
class EvaluationArtifact:
    path: str
    kind: str
    description: str


@dataclass(frozen=True)
class ClaimEvidence:
    claim: str
    status: ClaimStatus
    evidence: tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    suite: str
    seed: int
    config_path: str
    warnings: tuple[str, ...] = ()
    artifacts: tuple[EvaluationArtifact, ...] = ()
    results: Mapping[str, object] = field(default_factory=dict)

