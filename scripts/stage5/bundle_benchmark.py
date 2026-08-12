"""Benchmark only viable collector bundles in clean subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from scripts.stage5.conversion import conversion_python


def benchmark_bundles(root: Path, bundles: list[str]) -> dict[str, Any]:
    results = []
    for bundle in bundles:
        command = [str(conversion_python(root)), str(root / "scripts/stage5/bundle_worker.py"), "--root", str(root), "--bundle", bundle]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=1800, env={**os.environ, "PYTHONWARNINGS": "ignore", "TF_CPP_MIN_LOG_LEVEL": "2"})
        payload: dict[str, Any] = {"bundle": bundle, "command": command, "returncode": result.returncode, "process_wall_ms": (time.perf_counter() - started) * 1000}
        for line in reversed(result.stdout.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                payload.update(value)
                break
        if result.returncode != 0:
            payload.setdefault("failure_reason", result.stderr[-4000:] or "bundle benchmark failed")
        results.append(payload)
    output = root / "reports/production/stage5/bundle_benchmarks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "bundles": results}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload
