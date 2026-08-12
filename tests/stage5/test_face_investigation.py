from __future__ import annotations

from scripts.stage5.face_investigation import face_candidate_order, quantized_attempt_is_allowed


def test_face_candidate_order_is_strict() -> None:
    assert face_candidate_order() == (
        "face-litert-fp32",
        "face-onnx-fp32",
        "face-litert-dynamic-range",
    )


def test_face_quantization_requires_fp32_viability() -> None:
    assert quantized_attempt_is_allowed({"face-litert-fp32": "PARITY_PASS", "face-onnx-fp32": "FAILED"})
    assert not quantized_attempt_is_allowed({"face-litert-fp32": "FAILED"})
    assert not quantized_attempt_is_allowed({"face-litert-fp32": "PARITY_FAIL"})
