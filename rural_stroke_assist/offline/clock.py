"""Injectable timing policy for deterministic synchronization tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExponentialBackoff:
    base_seconds: float = 1.0
    maximum_seconds: float = 300.0
    jitter_ratio: float = 0.2

    def delay_seconds(self, *, attempt: int, random_value: float) -> float:
        if attempt < 1:
            raise ValueError("attempt must be positive")
        if not 0 <= random_value <= 1:
            raise ValueError("random_value must be in [0, 1]")
        raw = min(self.maximum_seconds, self.base_seconds * (2 ** (attempt - 1)))
        jittered = raw * (1 + ((random_value * 2) - 1) * self.jitter_ratio)
        return min(self.maximum_seconds, round(max(0.0, jittered), 6))

