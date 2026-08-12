from __future__ import annotations

import json
from pathlib import Path

import pytest

from rural_stroke_assist.stage5.contracts import (
    CandidateEvidence,
    ParityMetrics,
    PromotionDecision,
    Stage5Registry,
)
from scripts.stage5.build_baseline import build_baseline


def test_parity_metrics_reject_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        ParityMetrics(
            sample_count=1,
            label_agreement=1.0,
            mean_absolute_score_difference=float("nan"),
            median_absolute_score_difference=0.0,
            p95_absolute_score_difference=0.0,
            maximum_absolute_score_difference=0.0,
            nonfinite_score_count=0,
            roc_auc_drop=0.0,
            quality_acceptance_agreement=1.0,
            failure_semantics_agreement=1.0,
            preprocessing_contract_unchanged=True,
            fused_risk_band_agreement=1.0,
        )


def test_candidate_evidence_requires_reproducible_identity_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="converter"):
        CandidateEvidence(
            modality="face",
            candidate_id="face-literrt-fp32",
            source_artifact="models/source.keras",
            source_sha256="a" * 64,
            candidate_artifact=str(tmp_path / "candidate.tflite"),
            candidate_sha256="b" * 64,
            converter_name="",
            converter_version="1.0",
            runtime_name="tflite",
            runtime_version="2.0",
            conversion_options={},
            input_signature="(1,160,160,3)",
            output_signature="(1,1)",
            conversion_status="SUCCEEDED",
            failure_reason=None,
        )


def test_registry_round_trips_explicit_original_fallback(tmp_path: Path) -> None:
    registry = Stage5Registry(
        schema_version=1,
        runtime_profile="original",
        modalities={
            "face": {
                "logical_model_id": "face-trial-003",
                "original_artifact": "models/source.keras",
                "original_sha256": "a" * 64,
                "runtime": "tensorflow",
                "artifact": "models/source.keras",
                "artifact_sha256": "a" * 64,
                "decision": PromotionDecision.ORIGINAL_RETAINED.value,
                "original_fallback": "models/source.keras",
            }
        },
    )
    path = tmp_path / "edge_runtime_registry.json"
    registry.write(path)
    loaded = Stage5Registry.read(path)
    assert loaded.runtime_profile == "original"
    assert loaded.modalities["face"]["original_fallback"] == "models/source.keras"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_baseline_builder_uses_authoritative_registry_and_phase4_metrics(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "baseline.json"

    payload = build_baseline(root, output)

    assert payload["source"] == "config/baseline_registry.json + reports/evaluation/phase4/final_complete"
    assert payload["modalities"]["face"]["source"]["sha256"] == "C969C473CD369AEFE11815208407320D559AA7683998CA0EE28C53280C7011AC"
    assert payload["modalities"]["face"]["authoritative_metrics"]["direct"]["roc_auc"] == pytest.approx(0.9817744755)
    assert payload["modalities"]["speech"]["authoritative_metrics"]["direct"]["roc_auc"] == pytest.approx(0.9426279894)
    assert payload["modalities"]["metadata_context"]["authoritative_metrics"]["metrics"]["brier"] == pytest.approx(0.1813274953)
    assert output.is_file()
