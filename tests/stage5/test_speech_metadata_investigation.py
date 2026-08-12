from __future__ import annotations

from pathlib import Path

from scripts.stage5.metadata_investigation import metadata_candidate_order
from scripts.stage5.speech_investigation import speech_candidate_order


def test_speech_investigation_converts_only_fitted_classifier_pipeline() -> None:
    assert speech_candidate_order() == ("speech-onnx-random-forest-pipeline",)


def test_metadata_prefers_complete_pipeline_before_classifier_only() -> None:
    assert metadata_candidate_order() == (
        "metadata-onnx-complete-pipeline",
        "metadata-onnx-classifier-only",
    )
