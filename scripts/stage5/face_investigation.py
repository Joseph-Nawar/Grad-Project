"""Strictly ordered face conversion investigation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rural_stroke_assist.inference.registry import load_baseline_registry
from rural_stroke_assist.stage5.contracts import CandidateEvidence
from scripts.stage5.conversion import build_conversion_record, conversion_python, stage5_workspace


def face_candidate_order() -> tuple[str, ...]:
    return ("face-litert-fp32", "face-onnx-fp32", "face-litert-dynamic-range")


def quantized_attempt_is_allowed(statuses: dict[str, str]) -> bool:
    return statuses.get("face-litert-fp32") == "PARITY_PASS"


def _parity_pass(root: Path, candidate_id: str) -> bool:
    report = root / "reports" / "production" / "stage5" / f"{candidate_id}-parity.json"
    if not report.is_file():
        return False
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        return bool(payload["micro"]["gate"]["pass"] and payload["adapter"]["gate"]["pass"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _candidate_path(root: Path, candidate_id: str) -> Path:
    extension = ".tflite" if "litert" in candidate_id else ".onnx"
    return stage5_workspace(root) / "candidates" / "face" / f"{candidate_id}{extension}"


def _worker_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"status": "FAILED", "failure_reason": "conversion worker returned no JSON result"}


def _run_worker(root: Path, candidate_id: str, output: Path) -> dict[str, Any]:
    command = [
        str(conversion_python(root)),
        str(root / "scripts" / "stage5" / "face_conversion_worker.py"),
        "--source",
        str(load_baseline_registry().path_for("face")),
        "--candidate",
        candidate_id,
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, env=environment, timeout=900)
    payload = _worker_result(result.stdout)
    payload["command"] = command
    payload["returncode"] = result.returncode
    if result.returncode != 0 and not payload.get("failure_reason"):
        payload["failure_reason"] = result.stderr[-4000:] or "conversion worker failed"
    return payload


def investigate_face(root: Path) -> dict[str, Any]:
    registry = load_baseline_registry()
    source = registry.path_for("face")
    statuses: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for candidate_id in face_candidate_order():
        if candidate_id == "face-litert-dynamic-range" and not quantized_attempt_is_allowed(statuses):
            records.append({
                "candidate_id": candidate_id,
                "status": "NOT_ATTEMPTED",
                "reason": "LiteRT FP32 conversion or parity did not pass; strict quantization gate not met.",
            })
            continue
        output = _candidate_path(root, candidate_id)
        result = _run_worker(root, candidate_id, output)
        succeeded = result.get("status") == "SUCCEEDED" and output.is_file()
        statuses[candidate_id] = "PARITY_PASS" if succeeded and _parity_pass(root, candidate_id) else ("CONVERTED" if succeeded else "FAILED")
        converter_name = result.get("converter_name", "tensorflow.lite.TFLiteConverter" if "litert" in candidate_id else "tf2onnx")
        converter_version = result.get("converter_version", "2.19.1" if "litert" in candidate_id else "1.16.1")
        runtime_name = result.get("runtime_name", "tensorflow-lite" if "litert" in candidate_id else "onnxruntime")
        runtime_version = result.get("runtime_version", converter_version)
        record = build_conversion_record(
            modality="face",
            candidate_id=candidate_id,
            source_artifact=registry.component("face").path,
            source_sha256=_sha256(source),
            candidate_artifact=str(output) if succeeded else None,
            converter_name=converter_name,
            converter_version=converter_version,
            runtime_name=runtime_name,
            runtime_version=runtime_version,
            conversion_options=result.get("conversion_options", {}),
            input_signature=result.get("input_signature", "(batch,160,160,3) float32"),
            output_signature=result.get("output_signature", "(batch,1) float32"),
            failure_reason=None if succeeded else result.get("failure_reason", "conversion failed"),
        )
        records.append({"record": record.__dict__, "worker": result})
    payload = {"schema_version": 1, "modality": "face", "statuses": statuses, "candidates": records}
    output = root / "reports" / "production" / "stage5" / "face_conversion.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    payload = investigate_face(root)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
