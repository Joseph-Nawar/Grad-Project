import numpy as np
from PIL import Image

from rural_stroke_assist.inference.contracts import QualityStatus
from rural_stroke_assist.quality.audio_quality import DefaultAudioQualityAssessor
from rural_stroke_assist.quality.face_quality import OpenCVFaceQualityAssessor


def test_face_quality_reports_no_face() -> None:
    assessor = OpenCVFaceQualityAssessor()

    result = assessor(Image.new("RGB", (160, 160), color=(120, 120, 120)))

    assert result.status is QualityStatus.REJECT
    assert any(item.code == "no_face" for item in result.findings)
    assert any(item.code == "pose_not_assessed" for item in result.findings)


def test_face_quality_rejects_small_image_dimensions() -> None:
    result = OpenCVFaceQualityAssessor()(Image.new("RGB", (80, 80), color=(120, 120, 120)))

    assert result.status is QualityStatus.REJECT
    assert any(item.code == "dimensions" for item in result.findings)


def test_face_quality_rejects_multiple_faces_and_small_face() -> None:
    class FakeCascade:
        def __init__(self, faces):
            self.faces = faces

        def detectMultiScale(self, _gray, scaleFactor, minNeighbors):
            return np.asarray(self.faces)

    assessor = OpenCVFaceQualityAssessor(min_face_size_px=80)
    assessor._cascade = FakeCascade([(10, 10, 100, 100), (120, 10, 100, 100)])
    multiple = assessor(Image.new("RGB", (240, 160), color=(120, 120, 120)))
    assert multiple.status is QualityStatus.REJECT
    assert any(item.code == "multiple_faces" for item in multiple.findings)

    assessor._cascade = FakeCascade([(10, 10, 30, 30)])
    small = assessor(Image.new("RGB", (160, 160), color=(120, 120, 120)))
    assert small.status is QualityStatus.REJECT
    assert any(item.code == "face_too_small" for item in small.findings)


def test_audio_quality_rejects_silence_short_and_clipped_inputs() -> None:
    assessor = DefaultAudioQualityAssessor()

    silence = assessor(np.zeros(16000, dtype=np.float32), 16000)
    short = assessor(np.ones(100, dtype=np.float32) * 0.1, 16000)
    clipped = assessor(np.ones(16000, dtype=np.float32), 16000)

    assert silence.status is QualityStatus.REJECT
    assert any(item.code == "low_energy" for item in silence.findings)
    assert short.status is QualityStatus.REJECT
    assert any(item.code == "too_short" for item in short.findings)
    assert clipped.status is QualityStatus.REJECT
    assert any(item.code == "clipping" for item in clipped.findings)


def test_audio_quality_rejects_nonfinite_samples() -> None:
    signal = np.ones(16000, dtype=np.float32)
    signal[0] = np.nan

    result = DefaultAudioQualityAssessor()(signal, 16000)

    assert result.status is QualityStatus.REJECT
    assert any(item.code == "non_finite_samples" for item in result.findings)
