"""Machine-readable contracts for Stage 5 evidence and promotion."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class PromotionDecision(StrEnum):
    ADOPTED = "ADOPTED"
    REJECTED_CONVERSION = "REJECTED_CONVERSION"
    REJECTED_PARITY = "REJECTED_PARITY"
    REJECTED_PERFORMANCE = "REJECTED_PERFORMANCE"
    REJECTED_COMPLEXITY = "REJECTED_COMPLEXITY"
    ORIGINAL_RETAINED = "ORIGINAL_RETAINED"


def _require_sha256(value: str, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest.")


def _require_finite(value: float, field: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite.")


def _require_unit(value: float, field: str) -> None:
    _require_finite(value, field)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be in [0, 1].")


@dataclass(frozen=True)
class ParityMetrics:
    sample_count: int
    label_agreement: float
    mean_absolute_score_difference: float
    median_absolute_score_difference: float
    p95_absolute_score_difference: float
    maximum_absolute_score_difference: float
    nonfinite_score_count: int
    roc_auc_drop: float
    quality_acceptance_agreement: float
    failure_semantics_agreement: float
    preprocessing_contract_unchanged: bool
    fused_risk_band_agreement: float

    def __post_init__(self) -> None:
        if self.sample_count < 0 or self.nonfinite_score_count < 0:
            raise ValueError("Parity sample and nonfinite counts must be non-negative.")
        for field in (
            "label_agreement",
            "quality_acceptance_agreement",
            "failure_semantics_agreement",
            "fused_risk_band_agreement",
        ):
            _require_unit(getattr(self, field), field)
        for field in (
            "mean_absolute_score_difference",
            "median_absolute_score_difference",
            "p95_absolute_score_difference",
            "maximum_absolute_score_difference",
            "roc_auc_drop",
        ):
            _require_finite(getattr(self, field), field)
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative.")


@dataclass(frozen=True)
class CandidateEvidence:
    modality: str
    candidate_id: str
    source_artifact: str
    source_sha256: str
    candidate_artifact: str
    candidate_sha256: str | None
    converter_name: str
    converter_version: str
    runtime_name: str
    runtime_version: str
    conversion_options: dict[str, Any]
    input_signature: str
    output_signature: str
    conversion_status: str
    failure_reason: str | None
    parity: ParityMetrics | None = None
    benchmark: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.converter_name.strip():
            raise ValueError("converter name is required for reproducibility.")
        if not self.converter_version.strip():
            raise ValueError("converter version is required for reproducibility.")
        if not self.runtime_name.strip() or not self.runtime_version.strip():
            raise ValueError("runtime name and version are required.")
        _require_sha256(self.source_sha256, "source_sha256")
        if self.candidate_sha256 is not None:
            _require_sha256(self.candidate_sha256, "candidate_sha256")
        elif self.conversion_status != "FAILED":
            raise ValueError("candidate_sha256 is required when conversion succeeds.")
        if self.conversion_status == "FAILED" and not self.failure_reason:
            raise ValueError("failed conversion evidence requires a failure reason.")


@dataclass(frozen=True)
class Stage5Registry:
    schema_version: int
    runtime_profile: str
    modalities: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported Stage 5 registry schema version")
        if self.runtime_profile not in {"original", "optimized"}:
            raise ValueError("runtime_profile must be original or optimized")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "Stage5Registry":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            schema_version=int(data["schema_version"]),
            runtime_profile=str(data["runtime_profile"]),
            modalities=dict(data["modalities"]),
        )
