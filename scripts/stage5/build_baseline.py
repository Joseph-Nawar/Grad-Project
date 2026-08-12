"""Build compact Stage 5 baseline metadata from frozen authoritative evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_baseline(root: Path, output: Path) -> dict[str, Any]:
    registry_path = root / "config" / "baseline_registry.json"
    phase4 = root / "reports" / "evaluation" / "phase4" / "final_complete"
    registry = _read(registry_path)
    manifest_hashes = _read(phase4 / "run_manifest.json")["manifest_hashes"]
    learned = {name: value for name, value in registry["components"].items() if name in {"face", "speech", "metadata_context"}}
    modalities: dict[str, Any] = {}
    for name, component in learned.items():
        artifact = root / component["path"]
        metrics = _read(phase4 / ("metadata_context.json" if name == "metadata_context" else f"{name}.json"))
        manifest_name = {"face": "face_split", "speech": "speech_split", "metadata_context": "metadata_split"}[name]
        manifest = registry["manifests"][manifest_name]
        modalities[name] = {
            "logical_model_id": f"stage0-{name}-{Path(component['path']).stem}",
            "source": {
                "path": component["path"],
                "sha256": _sha256(artifact),
                "size_bytes": artifact.stat().st_size,
                "framework": component["framework"],
                "artifact_type": component["artifact_type"],
            },
            "contracts": {
                "input": component["input_contract"],
                "preprocessing": component["preprocessing_contract"],
                "output": component["output_contract"],
                "features": component.get("feature_columns", []),
            },
            "parity_manifest": {
                "path": manifest["path"],
                "sha256": manifest["sha256"],
                "size_bytes": manifest["size_bytes"],
                "split_policy": "existing held-out rows only; no new split",
            },
            "authoritative_metrics": metrics,
        }
    payload = {
        "schema_version": 1,
        "stage": "stage5",
        "source": "config/baseline_registry.json + reports/evaluation/phase4/final_complete",
        "environment": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "frozen_evaluation_runtime": _read(phase4 / "run_manifest.json").get("dependencies", {}),
        },
        "manifest_hashes": manifest_hashes,
        "modalities": modalities,
        "constraints": {
            "held_out_data_reused": True,
            "new_splits_created": False,
            "calibration_data_from_train_validation_only": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    output = root / "reports" / "production" / "stage5" / "baseline.json"
    build_baseline(root, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
