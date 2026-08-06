"""Deterministic stratified and group bootstrap confidence intervals."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np

from .contracts import ConfidenceInterval


def _percentile_ci(values: list[float], estimate: float, metric: str, iterations: int, seed: int, unit: str) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(metric, estimate, None, None, iterations, seed, unit)
    return ConfidenceInterval(metric, estimate, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), iterations, seed, unit)


def stratified_bootstrap_ci(y_true: Sequence[int], scores: Sequence[float], metric: Callable[[Sequence[int], Sequence[float]], float | None], *, iterations: int = 1000, seed: int = 42, metric_name: str = "metric") -> ConfidenceInterval:
    y, p = np.asarray(y_true), np.asarray(scores)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    estimate = metric(y.tolist(), p.tolist())
    values: list[float] = []
    for _ in range(iterations):
        indexes = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        value = metric(y[indexes].tolist(), p[indexes].tolist())
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return _percentile_ci(values, None if estimate is None else float(estimate), metric_name, iterations, seed, "sample-stratified")


def group_bootstrap_ci(groups: Sequence[str], y_true: Sequence[int], scores: Sequence[float], metric: Callable[[Sequence[int], Sequence[float]], float | None], *, iterations: int = 1000, seed: int = 42, metric_name: str = "metric") -> ConfidenceInterval:
    unique = np.asarray(sorted(set(groups)))
    group_array, y, p = np.asarray(groups), np.asarray(y_true), np.asarray(scores)
    rng = np.random.default_rng(seed)
    estimate = metric(y.tolist(), p.tolist())
    values: list[float] = []
    for _ in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        mask = np.isin(group_array, selected)
        value = metric(y[mask].tolist(), p[mask].tolist())
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return _percentile_ci(values, None if estimate is None else float(estimate), metric_name, iterations, seed, "group")
