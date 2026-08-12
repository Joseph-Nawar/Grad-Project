"""Speech exact-feature and raw-audio adapter parity."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scripts.stage5.conversion import conversion_python, stage5_workspace


def run_speech_parity(root: Path) -> dict[str, Any]:
    candidate = stage5_workspace(root) / "candidates" / "speech" / "speech-onnx-random-forest-pipeline.onnx"
    command = [str(conversion_python(root)), str(root / "scripts" / "stage5" / "speech_parity_worker.py"), "--root", str(root), "--candidate", str(candidate)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=1800, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    payload = {"command": command, "returncode": result.returncode}
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payload.update(value)
            break
    if result.returncode != 0:
        payload.setdefault("failure_reason", result.stderr[-4000:] or "parity worker failed")
    return payload


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    payload = run_speech_parity(root)
    output = root / "reports" / "production" / "stage5" / "speech-onnx-random-forest-pipeline-parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("returncode") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
