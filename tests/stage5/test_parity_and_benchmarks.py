from __future__ import annotations

import pytest

from scripts.stage5.parity import build_parity_metrics, parity_gate


def test_parity_metrics_report_required_error_statistics() -> None:
    metrics = build_parity_metrics(
        original_scores=[0.1, 0.5, 0.9, 0.3],
        candidate_scores=[0.11, 0.49, 0.9, 0.31],
        original_labels=[0, 1, 1, 0],
        candidate_labels=[0, 1, 1, 0],
        original_auc=0.9,
        candidate_auc=0.899,
        original_acceptance=[True, True, False, True],
        candidate_acceptance=[True, True, False, True],
        original_failure_semantics=["ok", "ok", "quality_reject", "ok"],
        candidate_failure_semantics=["ok", "ok", "quality_reject", "ok"],
        fused_original_bands=["LOW", "HIGH"],
        fused_candidate_bands=["LOW", "HIGH"],
    )
    assert metrics.sample_count == 4
    assert metrics.mean_absolute_score_difference == pytest.approx(0.0075)
    assert metrics.median_absolute_score_difference == pytest.approx(0.01)
    assert metrics.maximum_absolute_score_difference == pytest.approx(0.01)
    assert metrics.nonfinite_score_count == 0


def test_parity_gate_requires_all_declared_conditions() -> None:
    metrics = build_parity_metrics(
        original_scores=[0.1, 0.5],
        candidate_scores=[0.1, 0.5],
        original_labels=[0, 1],
        candidate_labels=[0, 1],
        original_auc=0.9,
        candidate_auc=0.9,
        original_acceptance=[True, True],
        candidate_acceptance=[True, True],
        original_failure_semantics=["ok", "ok"],
        candidate_failure_semantics=["ok", "ok"],
        fused_original_bands=["LOW"],
        fused_candidate_bands=["HIGH"],
    )
    result = parity_gate(metrics)
    assert result["pass"] is False
    assert result["fused_risk_band_agreement"] is False
