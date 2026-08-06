"""Pure binary classification and calibration metrics."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .contracts import MetricSummary


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Metric inputs must be finite.")
    return array


def calibration_error(y_true: Iterable[int], scores: Iterable[float], bins: int = 10) -> float | None:
    y, p = _finite(y_true), _finite(scores)
    if len(y) == 0:
        return None
    total = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= lower) & ((p < upper) if upper < 1 else (p <= upper))
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def classification_metrics(y_true: Iterable[int], scores: Iterable[float], threshold: float = 0.5) -> MetricSummary:
    y, p = _finite(y_true).astype(int), _finite(scores)
    if len(y) != len(p) or len(y) == 0 or not np.isin(y, [0, 1]).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Binary labels and bounded scores of equal non-zero length are required.")
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    div = lambda a, b: a / b if b else None
    return MetricSummary(
        n=len(y), positive=int(y.sum()), negative=int((y == 0).sum()),
        sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp), precision=div(tp, tp + fp),
        f1=div(2 * tp, 2 * tp + fp + fn),
        roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        pr_auc=float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        brier=float(np.mean((p - y) ** 2)), calibration_error=calibration_error(y, p),
        confusion_matrix=((tn, fp), (fn, tp)),
    )


def metric_dict(summary: MetricSummary) -> dict[str, object]:
    return {name: getattr(summary, name) for name in summary.__dataclass_fields__}
