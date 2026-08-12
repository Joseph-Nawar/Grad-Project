from __future__ import annotations

import pytest

from rural_stroke_assist.server.api.schemas.assessments import AssessmentImportRequest


def _envelope(provenance: dict) -> dict:
    return {"schema_version": 1, "case_id": "case", "assessment_id": "assessment", "request": {}, "result": {}, "provenance": provenance}


def test_promoted_edge_runtime_provenance_is_narrowly_approved() -> None:
    provenance = {
        "schema_version": 1,
        "model_ids": {"face": "canonical-face-v1", "speech": "canonical-speech-v1", "metadata_context": "canonical-metadata-v1", "acute_symptoms": "canonical-acute-v1"},
        "fusion_id": "canonical-fusion-v1",
        "profile": "optimized",
        "runtime_artifacts": {
            "face": {"artifact": "models/edge/face-trial-003-litert-fp32.tflite", "artifact_sha256": "12AAFECA62C02EC97741F89CEDDB1393684F175B32D15BFA8DD5B00F49A62F2E"},
            "speech": {"artifact": "models/edge/speech-trial-001-onnx.onnx", "artifact_sha256": "D08170F592DD1D45B7B86109077E1FC7EF14A58EC6EF51893CF04EE1B5CE7287"},
            "metadata_context": {"artifact": "models/experiments/metadata/mvp_metadata_risk_model.pkl", "artifact_sha256": "350545A6AB71A58373FD11BC2EF9F37BAADCC8420D66186EB6F4BD2B8AA1E8F5"},
        },
    }
    assert AssessmentImportRequest(envelope=_envelope(provenance), assessment_hash="a" * 64)


def test_unregistered_edge_artifact_is_rejected() -> None:
    provenance = {"schema_version": 1, "model_ids": {"face": "canonical-face-v1"}, "fusion_id": "canonical-fusion-v1", "profile": "optimized", "runtime_artifacts": {"face": {"artifact": "models/edge/fake.onnx", "artifact_sha256": "b" * 64}}}
    with pytest.raises(ValueError, match="runtime artifact"):
        AssessmentImportRequest(envelope=_envelope(provenance), assessment_hash="a" * 64)
