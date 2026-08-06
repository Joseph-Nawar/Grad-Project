import pytest

from rural_stroke_assist.evaluation.bootstrap import stratified_bootstrap_ci
from rural_stroke_assist.evaluation.metrics import calibration_error, classification_metrics


def test_classification_metrics_are_bounded_and_deterministic():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result.sensitivity == 1.0
    assert result.specificity == 1.0
    assert result.roc_auc == 1.0
    assert 0 <= result.calibration_error <= 1


def test_metrics_reject_nonfinite_scores():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0.1, float("nan")])


def test_stratified_bootstrap_repeats_exactly():
    kwargs = dict(y_true=[0, 0, 1, 1], scores=[.1, .2, .8, .9], metric=lambda y, p: classification_metrics(y, p).roc_auc, iterations=20, seed=42, metric_name="auc")
    assert stratified_bootstrap_ci(**kwargs) == stratified_bootstrap_ci(**kwargs)
