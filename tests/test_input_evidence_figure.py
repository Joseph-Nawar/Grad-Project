from __future__ import annotations

import numpy as np
import pytest

from scripts.figures.generate_input_evidence_figure import (
    SYNTHETIC_METADATA_VALUES,
    SYMPTOM_FIELDS,
    clip_audio_preview,
    load_audio_preview,
)


def test_clip_audio_preview_limits_signal_to_project_display_window() -> None:
    signal = np.arange(16000 * 6, dtype=np.float32)

    clipped = clip_audio_preview(signal, sample_rate=16000, max_duration_seconds=5.0)

    assert clipped.shape == (80000,)
    np.testing.assert_array_equal(clipped, signal[:80000])


def test_clip_audio_preview_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="empty"):
        clip_audio_preview(np.array([], dtype=np.float32), sample_rate=16000)


def test_load_audio_preview_returns_signal_and_sample_rate(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "example.wav"
    audio_path.write_bytes(b"placeholder")

    def fake_load(*_args, **_kwargs):
        return np.ones(12, dtype=np.float32), 16000

    monkeypatch.setattr("scripts.figures.generate_input_evidence_figure.librosa.load", fake_load)

    signal, sample_rate = load_audio_preview(audio_path)

    assert signal.size == 12
    assert sample_rate == 16000


def test_structured_panel_fields_match_project_contract() -> None:
    assert SYMPTOM_FIELDS == [
        ("Face drooping", "face_drooping"),
        ("Arm weakness", "arm_weakness"),
        ("Speech difficulty", "speech_difficulty"),
        ("Balance / coordination loss", "balance_or_coordination_loss"),
        ("Vision disturbance", "vision_disturbance"),
    ]
    assert list(SYNTHETIC_METADATA_VALUES) == [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
    ]
