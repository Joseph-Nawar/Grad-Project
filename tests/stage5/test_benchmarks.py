from __future__ import annotations

import pytest

from scripts.stage5.benchmark import percentile


def test_percentile_is_deterministic_for_small_samples() -> None:
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
