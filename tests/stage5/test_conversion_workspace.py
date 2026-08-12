from __future__ import annotations

from pathlib import Path

from scripts.stage5.conversion import build_conversion_record, stage5_workspace


def test_stage5_workspace_is_outside_canonical_model_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    workspace = stage5_workspace(root)
    assert workspace == root / ".stage5"
    assert "models" not in workspace.parts


def test_failed_conversion_record_keeps_reproducibility_fields() -> None:
    record = build_conversion_record(
        modality="face",
        candidate_id="face-litert-fp32",
        source_artifact="models/source.keras",
        source_sha256="a" * 64,
        candidate_artifact=None,
        converter_name="tensorflow.lite.TFLiteConverter",
        converter_version="2.19.1",
        runtime_name="tensorflow-lite",
        runtime_version="2.19.1",
        conversion_options={"precision": "fp32"},
        input_signature="(1,160,160,3)",
        output_signature="(1,1)",
        failure_reason="isolated toolchain unavailable",
    )
    assert record.conversion_status == "FAILED"
    assert record.candidate_sha256 is None
    assert record.failure_reason == "isolated toolchain unavailable"
