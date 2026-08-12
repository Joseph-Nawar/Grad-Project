"""Metadata complete-pipeline-first ONNX investigation."""

from __future__ import annotations

from typing import Tuple
import hashlib
import json
import os
import subprocess
from pathlib import Path

from scripts.stage5.conversion import build_conversion_record, conversion_python, stage5_workspace
from rural_stroke_assist.inference.registry import load_baseline_registry


def metadata_candidate_order() -> Tuple[str, ...]:
    return ("metadata-onnx-complete-pipeline", "metadata-onnx-classifier-only")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def investigate_metadata(root: Path) -> dict:
    registry = load_baseline_registry()
    source = registry.path_for("metadata_context")
    candidate = stage5_workspace(root) / "candidates" / "metadata" / "metadata-onnx-complete-pipeline.onnx"
    command = [str(conversion_python(root)), str(root / "scripts" / "stage5" / "metadata_conversion_worker.py"), "--source", str(source), "--output", str(candidate)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=900, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    worker = {}
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            worker = value
            break
    succeeded = result.returncode == 0 and candidate.is_file() and worker.get("status") == "SUCCEEDED"
    record = build_conversion_record(
        modality="metadata_context", candidate_id="metadata-onnx-complete-pipeline",
        source_artifact=registry.component("metadata_context").path, source_sha256=_sha256(source),
        candidate_artifact=str(candidate) if succeeded else None,
        converter_name=worker.get("converter_name", "skl2onnx.convert_sklearn"), converter_version=worker.get("converter_version", "1.18.0"),
        runtime_name="onnxruntime", runtime_version=worker.get("runtime_version", "1.20.1"),
        conversion_options=worker.get("conversion_options", {"opset": 17, "zipmap": False, "complete_pipeline": True}),
        input_signature="ten named frozen metadata columns", output_signature="(batch,2) probability matrix",
        failure_reason=None if succeeded else worker.get("failure_reason", result.stderr[-4000:] or "conversion failed"),
    )
    payload = {"schema_version": 1, "modality": "metadata_context", "record": record.__dict__, "worker": worker, "command": command, "returncode": result.returncode, "classifier_only": {"status": "NOT_ATTEMPTED", "reason": "complete pipeline investigation is authoritative; preprocessing rewrite is forbidden"}}
    output = root / "reports" / "production" / "stage5" / "metadata_conversion.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    payload = investigate_metadata(root)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
