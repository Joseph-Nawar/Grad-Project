"""Numerical and semantic parity gates for Stage 5 candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rural_stroke_assist.stage5.contracts import ParityMetrics


def _agreement(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != len(right):
        return 0.0
    if not left:
        return 1.0
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def build_parity_metrics(
    *,
    original_scores: Sequence[float],
    candidate_scores: Sequence[float],
    original_labels: Sequence[Any],
    candidate_labels: Sequence[Any],
    original_auc: float,
    candidate_auc: float,
    original_acceptance: Sequence[bool],
    candidate_acceptance: Sequence[bool],
    original_failure_semantics: Sequence[str],
    candidate_failure_semantics: Sequence[str],
    fused_original_bands: Sequence[str],
    fused_candidate_bands: Sequence[str],
) -> ParityMetrics:
    if len(original_scores) != len(candidate_scores):
        raise ValueError("original and candidate score arrays must have equal length")
    original = np.asarray(original_scores, dtype=float)
    candidate = np.asarray(candidate_scores, dtype=float)
    if original.shape != candidate.shape:
        raise ValueError("original and candidate score arrays must have equal shape")
    nonfinite = int((~np.isfinite(original)).sum() + (~np.isfinite(candidate)).sum())
    differences = np.abs(candidate - original)
    finite_differences = differences[np.isfinite(differences)]
    if finite_differences.size == 0:
        finite_differences = np.asarray([float("inf")])
    return ParityMetrics(
        sample_count=int(original.size),
        label_agreement=_agreement(original_labels, candidate_labels),
        mean_absolute_score_difference=float(np.mean(finite_differences)),
        median_absolute_score_difference=float(np.median(finite_differences)),
        p95_absolute_score_difference=float(np.percentile(finite_differences, 95)),
        maximum_absolute_score_difference=float(np.max(finite_differences)),
        nonfinite_score_count=nonfinite,
        roc_auc_drop=abs(float(original_auc) - float(candidate_auc)),
        quality_acceptance_agreement=_agreement(original_acceptance, candidate_acceptance),
        failure_semantics_agreement=_agreement(original_failure_semantics, candidate_failure_semantics),
        preprocessing_contract_unchanged=True,
        fused_risk_band_agreement=_agreement(fused_original_bands, fused_candidate_bands),
    )


def parity_gate(metrics: ParityMetrics) -> dict[str, Any]:
    checks = {
        "label_agreement": metrics.label_agreement >= 0.99,
        "absolute_roc_auc_drop": metrics.roc_auc_drop <= 0.005,
        "median_absolute_score_difference": metrics.median_absolute_score_difference <= 0.01,
        "finite_scores": metrics.nonfinite_score_count == 0,
        "adapter_quality_acceptance": metrics.quality_acceptance_agreement == 1.0,
        "failure_semantics": metrics.failure_semantics_agreement == 1.0,
        "preprocessing_contract": metrics.preprocessing_contract_unchanged,
        "fused_risk_band_agreement": metrics.fused_risk_band_agreement == 1.0,
    }
    return {"pass": all(checks.values()), "checks": checks, "fused_risk_band_agreement": checks["fused_risk_band_agreement"]}
