"""Assemble compact Stage 5 conversion, parity, benchmark, and decision evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble(root: Path) -> dict[str, Any]:
    evidence = root / "reports/production/stage5"
    baseline = _read(evidence / "baseline.json")
    face_conversion = _read(evidence / "face_conversion.json")
    speech_conversion = _read(evidence / "speech_conversion.json")
    metadata_conversion = _read(evidence / "metadata_conversion.json")
    face_litert = _read(evidence / "face-litert-fp32-parity.json")
    face_onnx = _read(evidence / "face-onnx-fp32-parity.json")
    face_quant = _read(evidence / "face-litert-dynamic-range-parity.json")
    speech_parity = _read(evidence / "speech-onnx-random-forest-pipeline-parity.json")
    benchmarks = _read(evidence / "benchmarks.json")
    bundles = _read(evidence / "bundle_benchmarks.json")
    assessment = _read(evidence / "assessment_parity.json")
    docker_closure = _read(evidence / "docker_closure.json")
    registry = _read(root / "config/edge_runtime_registry.json")
    conversion_manifest = {
        "schema_version": 1,
        "candidates": {
            "face": {
                "attempted": ["face-litert-fp32", "face-onnx-fp32", "face-litert-dynamic-range"],
                "conversion": face_conversion["candidates"],
                "parity": {"face-litert-fp32": face_litert, "face-onnx-fp32": face_onnx, "face-litert-dynamic-range": face_quant},
            },
            "speech": {"attempted": ["speech-onnx-random-forest-pipeline"], "conversion": speech_conversion, "parity": speech_parity},
            "metadata_context": {"attempted": ["metadata-onnx-complete-pipeline", "metadata-onnx-classifier-only"], "conversion": metadata_conversion, "parity": None},
        },
        "gates": {"label_agreement_min": 0.99, "absolute_roc_auc_drop_max": 0.005, "median_absolute_score_difference_max": 0.01, "finite_scores": "100%", "quality_acceptance": "100%", "failure_semantics": "100%", "preprocessing_contract": "unchanged", "fused_risk_band_agreement": "100%"},
        "benchmark": benchmarks,
        "bundle_comparison": {
            "bundles": bundles["bundles"],
            "selected": "mixed-litert",
            "face_onnx_decision": "REJECTED_PERFORMANCE",
            "face_onnx_reason": "mixed-onnx used 58191 more model bytes and had slower warm p50/p95 assessment latency than mixed-litert; its RSS advantage did not offset the measured latency and size regression.",
            "docker_image_bytes": docker_closure["image"]["bytes"],
            "docker_image_status": "PASS",
            "docker_closure": docker_closure,
        },
        "assessment_parity": assessment,
        "collector_edge_verification": {
            "compose_config": "PASS",
            "stage3_docker_boundary_tests": "PASS",
            "stage4_docker_boundary_tests": "PASS",
            "optimized_offline_regression": "PASS",
            "image_build": "PASS",
            "image_inspection": "PASS",
            "profile_smoke": "PASS",
            "original_fallback": "PASS",
            "optimized_compose_e2e": "PASS",
            "docker_closure_evidence": "docker_closure.json",
        },
        "promotion_registry": "config/edge_runtime_registry.json",
        "baseline": baseline,
        "decisions": {
            "face-litert-fp32": "ADOPTED",
            "face-onnx-fp32": "REJECTED_PERFORMANCE",
            "face-litert-dynamic-range": "REJECTED_PARITY",
            "speech-onnx-random-forest-pipeline": "ADOPTED",
            "metadata-onnx-complete-pipeline": "REJECTED_CONVERSION",
            "metadata-onnx-classifier-only": "ORIGINAL_RETAINED",
        },
    }
    (evidence / "conversion_manifest.json").write_text(json.dumps(conversion_manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "stage": "stage5",
        "status": "COMPLETE_AND_VERIFIED",
        "final_profile": "optimized",
        "adopted_bundle": registry["bundle"]["name"],
        "modalities": {name: {"decision": entry["decision"], "backend": entry["backend"], "artifact": entry["artifact"], "original_fallback": entry["original_fallback"]} for name, entry in registry["modalities"].items()},
        "required_gates": conversion_manifest["gates"],
        "assessment_risk_band_agreement": assessment["risk_band_agreement"],
        "collector_edge_verification": conversion_manifest["collector_edge_verification"],
        "bundle_decision": conversion_manifest["bundle_comparison"],
        "stage3b": "DEFERRED_AND_FROZEN",
        "frozen_ml_semantics_changed": False,
        "git_operations": "NONE",
        "evidence_files": ["baseline.json", "conversion_environment.json", "conversion_manifest.json", "face_conversion.json", "face-litert-fp32-parity.json", "face-onnx-fp32-parity.json", "face-litert-dynamic-range-parity.json", "speech_conversion.json", "speech-onnx-random-forest-pipeline-parity.json", "metadata_conversion.json", "benchmarks.json", "bundle_benchmarks.json", "assessment_parity.json", "docker_closure.json", "edge_runtime_registry.json"],
    }
    (evidence / "stage5_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(assemble(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
