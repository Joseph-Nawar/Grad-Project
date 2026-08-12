"""Run face micro-parity and raw-image adapter parity in a clean process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scripts.stage5.conversion import conversion_python, stage5_workspace


def run_face_parity(root: Path, candidate_id: str) -> dict[str, Any]:
    candidate = stage5_workspace(root) / "candidates" / "face" / f"{candidate_id}{'.tflite' if 'litert' in candidate_id else '.onnx'}"
    command = [
        str(conversion_python(root)),
        str(root / "scripts" / "stage5" / "face_parity_worker.py"),
        "--root",
        str(root),
        "--candidate-id",
        candidate_id,
        "--candidate",
        str(candidate),
    ]
    environment = os.environ.copy()
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, env=environment, timeout=1800)
    payload: dict[str, Any] = {"candidate_id": candidate_id, "command": command, "returncode": result.returncode}
    for line in reversed(result.stdout.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload.update(parsed)
            break
    if result.returncode != 0:
        payload.setdefault("failure_reason", result.stderr[-4000:] or "parity worker failed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    payload = run_face_parity(root, args.candidate_id)
    output = root / "reports" / "production" / "stage5" / f"{args.candidate_id}-parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("returncode") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
