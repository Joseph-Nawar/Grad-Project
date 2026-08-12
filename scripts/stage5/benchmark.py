"""Clean-subprocess runner and bundle benchmarks."""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.stage5.conversion import conversion_python, stage5_workspace


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    return float(np.percentile(np.asarray(values, dtype=float), percent, method="linear"))


def benchmark_subprocess(root: Path, target: str) -> dict[str, Any]:
    command = [str(conversion_python(root)), str(root / "scripts" / "stage5" / "benchmark_worker.py"), "--root", str(root), "--target", target]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=1800, env={**os.environ, "PYTHONWARNINGS": "ignore", "TF_CPP_MIN_LOG_LEVEL": "2"})
    payload: dict[str, Any] = {"target": target, "command": command, "returncode": result.returncode, "process_wall_ms": (time.perf_counter() - started) * 1000}
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payload.update(value)
            break
    if result.returncode != 0:
        payload.setdefault("failure_reason", result.stderr[-4000:] or "benchmark worker failed")
    return payload


def benchmark_targets(root: Path, targets: Sequence[str]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "machine": platform.platform(),
        "python": sys.version,
        "batch_size": 1,
        "targets": [benchmark_subprocess(root, target) for target in targets],
    }
    output = root / "reports" / "production" / "stage5" / "benchmarks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload
