from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.inference.contracts import (
    ModalityEvidence,
    QualityFinding,
    QualityStatus,
)
from rural_stroke_assist.inference.exceptions import FeatureContractError, InferenceFailure
from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.metadata_adapter import MetadataInput, MetadataAdapter
from rural_stroke_assist.inference.registry import BaselineRegistry
from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
from rural_stroke_assist.inference.symptom_adapter import SymptomAdapter
from rural_stroke_assist.quality.face_quality import FaceQualityAssessment
from rural_stroke_assist.quality.audio_quality import AudioQualityAssessment


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = BaselineRegistry.from_file(ROOT / "config" / "baseline_registry.json")


def test_available_evidence_requires_bounded_score() -> None:
    with pytest.raises(ValueError):
        ModalityEvidence(
            modality="face",
            available=True,
            score=1.1,
            score_semantics="visual_proxy_evidence",
            label="Stroke",
            quality_status=QualityStatus.PASS,
            provenance="models/example.keras",
        )


def test_unavailable_evidence_has_no_score() -> None:
    evidence = ModalityEvidence.unavailable(
        modality="speech",
        score_semantics="dysarthria_proxy_evidence",
        provenance="models/example.pkl",
        warning="No audio provided.",
    )

    assert evidence.available is False
    assert evidence.score is None
    assert evidence.quality_status is QualityStatus.UNAVAILABLE


class FakeFaceModel:
    def predict(self, _batch, verbose=0):
        return np.asarray([[0.73]], dtype=np.float32)


def pass_face_quality(_image):
    return FaceQualityAssessment(
        status=QualityStatus.PASS,
        findings=(
            QualityFinding(
                code="pose_not_assessed",
                message="Pose was not assessed.",
                status=QualityStatus.NOT_ASSESSED,
            ),
        ),
    )


def reject_face_quality(_image):
    return FaceQualityAssessment(
        status=QualityStatus.REJECT,
        findings=(
            QualityFinding(
                code="multiple_faces",
                message="Multiple faces detected.",
                status=QualityStatus.REJECT,
            ),
        ),
    )


def test_face_adapter_preprocesses_and_extracts_sigmoid(tmp_path: Path) -> None:
    image_path = tmp_path / "face.png"
    Image.new("RGB", (32, 24), color=(120, 120, 120)).save(image_path)
    loads = []

    adapter = FaceAdapter(
        registry=REGISTRY,
        model_loader=lambda path: loads.append(path) or FakeFaceModel(),
        quality_assessor=pass_face_quality,
    )

    result = adapter.infer(image_path)
    second = adapter.infer(image_path)

    assert result.available is True
    assert result.score == pytest.approx(0.73, abs=1e-6)
    assert result.score_semantics == "visual_proxy_evidence"
    assert result.label == "Stroke"
    assert result.confidence == pytest.approx(0.73, abs=1e-6)
    assert result.provenance == REGISTRY.component("face").path
    assert second.score == result.score
    assert len(loads) == 1


def test_face_adapter_rejects_quality_without_inference(tmp_path: Path) -> None:
    image_path = tmp_path / "face.png"
    Image.new("RGB", (32, 24), color=(120, 120, 120)).save(image_path)
    calls = []
    adapter = FaceAdapter(
        registry=REGISTRY,
        model_loader=lambda path: calls.append(path) or FakeFaceModel(),
        quality_assessor=reject_face_quality,
    )

    result = adapter.infer(image_path)

    assert result.available is False
    assert result.score is None
    assert result.quality_status is QualityStatus.REJECT
    assert calls == []


def test_face_adapter_missing_input_is_not_zero() -> None:
    adapter = FaceAdapter(registry=REGISTRY, model_loader=lambda _path: FakeFaceModel())

    result = adapter.infer(None)

    assert result.score is None
    assert result.available is False


class FakeSpeechModel:
    classes_ = np.asarray(["control", "dysarthric"])

    def predict_proba(self, frame):
        assert list(frame.columns) == REGISTRY.component("speech").feature_columns
        return np.asarray([[0.2, 0.8]], dtype=np.float64)


def pass_audio_quality(signal, sample_rate):
    return AudioQualityAssessment(status=QualityStatus.PASS, findings=())


def test_speech_adapter_uses_exact_feature_order_and_positive_class(tmp_path: Path) -> None:
    signal = np.ones(16000, dtype=np.float32) * 0.05
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"fixture")
    adapter = SpeechAdapter(
        registry=REGISTRY,
        model_loader=lambda _path: FakeSpeechModel(),
        audio_loader=lambda _path: (signal, 16000),
        quality_assessor=pass_audio_quality,
    )

    result = adapter.infer(audio_path)

    assert result.available is True
    assert result.score == pytest.approx(0.8)
    assert result.label == "dysarthric"
    assert result.score_semantics == "dysarthria_proxy_evidence"


def test_speech_adapter_rejects_nonfinite_features(tmp_path: Path) -> None:
    signal = np.ones(16000, dtype=np.float32) * 0.05
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"fixture")

    def bad_features(*_args, **_kwargs):
        raise FeatureContractError("non-finite feature")

    adapter = SpeechAdapter(
        registry=REGISTRY,
        model_loader=lambda _path: FakeSpeechModel(),
        audio_loader=lambda _path: (signal, 16000),
        feature_extractor=bad_features,
        quality_assessor=pass_audio_quality,
    )

    with pytest.raises(FeatureContractError):
        adapter.infer(audio_path)


def test_speech_adapter_maps_decode_failure_to_unusable_evidence() -> None:
    def broken_loader(_path):
        raise ValueError("invalid audio")

    adapter = SpeechAdapter(
        registry=REGISTRY,
        model_loader=lambda _path: FakeSpeechModel(),
        audio_loader=broken_loader,
        quality_assessor=pass_audio_quality,
    )

    result = adapter.infer(Path("bad.wav"))

    assert result.available is False
    assert result.score is None
    assert result.quality_status is QualityStatus.REJECT


class FakeMetadataModel:
    classes_ = np.asarray([0, 1])
    feature_names_in_ = np.asarray(REGISTRY.component("metadata_context").feature_columns)

    def predict_proba(self, frame):
        assert list(frame.columns) == REGISTRY.component("metadata_context").feature_columns
        return np.asarray([[0.35, 0.65]], dtype=np.float64)


def test_metadata_schema_uses_exact_contract_and_contextual_semantics() -> None:
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
    adapter = MetadataAdapter(
        registry=REGISTRY,
        model_loader=lambda _path: FakeMetadataModel(),
    )

    result = adapter.infer(metadata)

    assert result.score == pytest.approx(0.65)
    assert result.label == "stroke"
    assert result.score_semantics == "contextual_risk_evidence"
    assert result.confidence is None
    assert "acute" in " ".join(result.warnings).lower()


def test_metadata_schema_rejects_unsupported_category() -> None:
    with pytest.raises(ValueError):
        MetadataInput(
            age=68,
            hypertension=1,
            heart_disease=0,
            avg_glucose_level=130.0,
            bmi=25.0,
            gender="UnknownGender",
            ever_married="Yes",
            work_type="Private",
            residence_type="Urban",
            smoking_status="never smoked",
        )


def test_symptom_adapter_delegates_existing_urgent_rules() -> None:
    adapter = SymptomAdapter(registry=REGISTRY)

    result = adapter.infer(
        AcuteStrokeSymptoms(face_drooping=True, arm_weakness=True)
    )

    assert result.available is True
    assert result.score == pytest.approx(0.85)
    assert result.label == "URGENT"
    assert result.score_semantics == "deterministic_acute_symptom_evidence"


def test_inference_failure_is_typed() -> None:
    class BrokenModel:
        def predict(self, _batch, verbose=0):
            raise RuntimeError("model failure")

    adapter = FaceAdapter(
        registry=REGISTRY,
        model_loader=lambda _path: BrokenModel(),
        quality_assessor=pass_face_quality,
    )

    image_path = ROOT / "reports" / "figures" / "face_class_distribution.png"
    with pytest.raises(InferenceFailure):
        adapter.infer(image_path)
