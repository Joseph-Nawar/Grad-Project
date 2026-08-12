from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.inference.contracts import QualityStatus
from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.metadata_adapter import MetadataAdapter, MetadataInput
from rural_stroke_assist.inference.registry import BaselineRegistry
from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
from rural_stroke_assist.inference.runners import LiteRTRunner, ONNXRunner
from rural_stroke_assist.quality.audio_quality import AudioQualityAssessment
from rural_stroke_assist.quality.face_quality import FaceQualityAssessment


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = BaselineRegistry.from_file(ROOT / "config" / "baseline_registry.json")


def test_face_runner_receives_frozen_preprocessed_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "face.png"
    Image.new("RGB", (32, 24), color=(120, 120, 120)).save(image_path)
    received: list[np.ndarray] = []

    class Runner:
        def run(self, tensor: np.ndarray) -> np.ndarray:
            received.append(tensor.copy())
            return np.asarray([[0.73]], dtype=np.float32)

    adapter = FaceAdapter(
        registry=REGISTRY,
        runner=Runner(),
        quality_assessor=lambda _image: FaceQualityAssessment(QualityStatus.PASS, ()),
    )

    result = adapter.infer(image_path)

    assert result.score == pytest.approx(0.73)
    assert received[0].shape == (1, 160, 160, 3)
    assert received[0].dtype == np.float32


def test_speech_runner_receives_exact_ordered_feature_frame(tmp_path: Path) -> None:
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"fixture")
    signal = np.ones(16000, dtype=np.float32) * 0.05
    received: list[pd.DataFrame] = []

    class Runner:
        classes = ("control", "dysarthric")

        def run(self, frame: pd.DataFrame):
            received.append(frame.copy())
            return np.asarray([[0.2, 0.8]], dtype=np.float64), self.classes

    adapter = SpeechAdapter(
        registry=REGISTRY,
        runner=Runner(),
        audio_loader=lambda _path: (signal, 16000),
        feature_extractor=lambda _signal, _rate: {
            name: float(index)
            for index, name in enumerate(REGISTRY.component("speech").feature_columns)
        },
        quality_assessor=lambda _signal, _rate: AudioQualityAssessment(QualityStatus.PASS, ()),
    )

    result = adapter.infer(audio_path)

    assert result.score == 0.8
    assert list(received[0].columns) == REGISTRY.component("speech").feature_columns
    assert received[0].shape == (1, 37)


def test_default_factory_remains_original_profile(monkeypatch) -> None:
    monkeypatch.delenv("RURALSTROKE_EDGE_RUNTIME_PROFILE", raising=False)
    service = create_default_assessment_service()
    assert service.runtime_profile == "original"


def test_metadata_runner_receives_existing_frozen_frame() -> None:
    received: list[pd.DataFrame] = []

    class Runner:
        classes = (0, 1)

        def run(self, frame: pd.DataFrame):
            received.append(frame.copy())
            return np.asarray([[0.35, 0.65]], dtype=np.float64), self.classes

    metadata = MetadataInput(
        age=68,
        hypertension=1,
        heart_disease=0,
        avg_glucose_level=130.0,
        bmi=None,
        gender="Male",
        ever_married="Yes",
        work_type="Private",
        residence_type="Urban",
        smoking_status="never smoked",
    )
    result = MetadataAdapter(registry=REGISTRY, runner=Runner()).infer(metadata)

    assert result.score == 0.65
    assert list(received[0].columns) == REGISTRY.component("metadata_context").feature_columns
    assert received[0].shape == (1, 10)


def test_optimized_profile_is_registry_declared_mixed_bundle(monkeypatch) -> None:
    monkeypatch.setenv("RURALSTROKE_EDGE_RUNTIME_PROFILE", "optimized")
    service = create_default_assessment_service()
    assert service.runtime_profile == "optimized"
    assert type(service.adapters["face"]._runner).__name__ == "LiteRTRunner"
    assert type(service.adapters["speech"]._runner).__name__ == "ONNXRunner"
    assert type(service.adapters["metadata_context"]._runner).__name__ == "OriginalTabularRunner"


def test_onnx_runner_preserves_exact_input_and_returns_probability_matrix(tmp_path: Path) -> None:
    received = []

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "features"})()]

        def run(self, _outputs, inputs):
            received.append(inputs)
            return [np.asarray([[0.2, 0.8]])]

    runner = ONNXRunner(tmp_path / "candidate.onnx", session_loader=lambda _path: Session())
    frame = pd.DataFrame([[1.0, 2.0]], columns=["a", "b"])
    probabilities, classes = runner.run(frame)

    assert np.array_equal(received[0]["features"], frame.to_numpy(dtype=np.float32))
    assert probabilities.shape == (1, 2)
    assert classes == (0, 1)


def test_litert_runner_forwards_exact_tensor(tmp_path: Path) -> None:
    received = []

    class Interpreter:
        def allocate_tensors(self): pass
        def get_input_details(self): return [{"index": 1}]
        def get_output_details(self): return [{"index": 2}]
        def set_tensor(self, _index, value): received.append(value)
        def invoke(self): pass
        def get_tensor(self, _index): return np.asarray([[0.73]], dtype=np.float32)

    runner = LiteRTRunner(tmp_path / "candidate.tflite", interpreter_factory=lambda _path: Interpreter())
    value = runner.run(np.ones((1, 160, 160, 3), dtype=np.float32))

    assert value.shape == (1, 1)
    assert received[0].shape == (1, 160, 160, 3)
