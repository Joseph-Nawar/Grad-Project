"""Conservative audio quality checks matching the accepted feature policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rural_stroke_assist.inference.contracts import QualityFinding, QualityStatus


@dataclass(frozen=True)
class AudioQualityAssessment:
    status: QualityStatus
    findings: tuple[QualityFinding, ...]


class AudioQualityAssessor(Protocol):
    def __call__(self, signal: np.ndarray, sample_rate: int) -> AudioQualityAssessment:
        ...


class DefaultAudioQualityAssessor:
    def __init__(
        self,
        *,
        min_duration_seconds: float = 0.25,
        low_rms_threshold: float = 1e-4,
        clipping_fraction_threshold: float = 0.01,
    ) -> None:
        self.min_duration_seconds = min_duration_seconds
        self.low_rms_threshold = low_rms_threshold
        self.clipping_fraction_threshold = clipping_fraction_threshold

    def __call__(self, signal: np.ndarray, sample_rate: int) -> AudioQualityAssessment:
        findings: list[QualityFinding] = []
        if sample_rate <= 0 or signal.size == 0:
            findings.append(QualityFinding("invalid_signal", "Audio contains no usable samples.", QualityStatus.REJECT))
            return AudioQualityAssessment(QualityStatus.REJECT, tuple(findings))
        if not np.isfinite(signal).all():
            findings.append(QualityFinding("non_finite_samples", "Audio contains NaN or infinite samples.", QualityStatus.REJECT))
        duration = signal.size / sample_rate
        if duration < self.min_duration_seconds:
            findings.append(QualityFinding("too_short", "Audio is shorter than the minimum usable duration.", QualityStatus.REJECT))
        rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
        if rms <= self.low_rms_threshold:
            findings.append(QualityFinding("low_energy", "Audio has near-silent energy.", QualityStatus.REJECT))
        clipping_fraction = float(np.mean(np.abs(signal) >= 0.999))
        if clipping_fraction > self.clipping_fraction_threshold:
            findings.append(QualityFinding("clipping", "Audio contains excessive clipped samples.", QualityStatus.REJECT))
        if not findings:
            return AudioQualityAssessment(QualityStatus.PASS, ())
        status = QualityStatus.REJECT if any(item.status is QualityStatus.REJECT for item in findings) else QualityStatus.WARN
        return AudioQualityAssessment(status, tuple(findings))
