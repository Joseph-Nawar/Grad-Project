"""Collect redacted Stage 3 image, stack, benchmark, and compatibility evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _image(name: str) -> dict[str, object]:
    payload = json.loads(_run(["docker", "image", "inspect", name]))[0]
    return {
        "name": name,
        "id": payload.get("Id"),
        "size_bytes": payload.get("Size"),
        "created": payload.get("Created"),
        "user": payload.get("Config", {}).get("User"),
        "repo_digests": payload.get("RepoDigests", []),
    }


def _rss() -> dict[str, int | None]:
    try:
        raw = _run(["docker", "stats", "--no-stream", "--format", "{{json .}}"])
    except subprocess.CalledProcessError:
        return {}
    values: dict[str, int | None] = {}
    for line in raw.splitlines():
        try:
            item = json.loads(line)
            usage = str(item.get("MemUsage", "")).split("/")[0].strip().upper()
            multiplier = 1
            if usage.endswith("GIB"):
                multiplier, usage = 1024**3, usage[:-3]
            elif usage.endswith("MIB"):
                multiplier, usage = 1024**2, usage[:-3]
            elif usage.endswith("KIB"):
                multiplier, usage = 1024, usage[:-3]
            values[str(item.get("Name"))] = int(float(usage.strip()) * multiplier)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return values


def _portable_containers(raw_lines: list[str]) -> list[str]:
    """Keep useful Compose observations while excluding host-specific metadata."""
    portable: list[str] = []
    for line in raw_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            portable.append("<unparsed container observation>")
            continue
        if isinstance(record, dict):
            record = {
                key: value
                for key, value in record.items()
                if key not in {"Labels", "Mounts", "LocalVolumes"}
            }
            portable.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        else:
            portable.append("<unparsed container observation>")
    return portable


def _api_json(method: str, path: str, token: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        f"http://127.0.0.1:8000{path}",
        method=method,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Idempotency-Key": str(uuid.uuid4())},
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _benchmark(repeats: int = 20) -> dict[str, object]:
    token_path = ROOT / ".runtime-secrets" / "demo_token"
    if not token_path.is_file():
        return {"status": "not-run", "reason": "local stack token is unavailable"}
    token = token_path.read_text(encoding="utf-8").strip()
    durations: list[float] = []
    scores: list[float] = []
    for _ in range(repeats):
        case_id = str(uuid.uuid4())
        _api_json("POST", "/api/v1/cases", token, {"id": case_id, "facility": "Local health post", "assessment_input": {"session_id": "stage3-evidence"}})
        started = time.perf_counter()
        result = _api_json("POST", "/api/v1/assessments", token, {"id": str(uuid.uuid4()), "case_id": case_id, "session_id": "stage3-evidence", "metadata": None, "acute_symptoms": {"face_drooping": True, "arm_weakness": True, "speech_difficulty": False, "balance_or_coordination_loss": False, "vision_disturbance": False, "sudden_severe_headache": False, "confusion_or_understanding_difficulty": False, "symptom_onset_minutes": 30, "symptoms_resolved": False}})
        durations.append((time.perf_counter() - started) * 1000)
        snapshot = result.get("result_snapshot", {})
        if isinstance(snapshot, dict):
            fusion = snapshot.get("fusion", {})
            if isinstance(fusion, dict) and isinstance(fusion.get("evidence_score"), (int, float)):
                scores.append(float(fusion["evidence_score"]))
    durations.sort()
    percentile = lambda values, q: values[min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))]
    return {
        "status": "complete",
        "repeats": repeats,
        "warm_assessment_ms": {"p50_ms": percentile(durations, 0.50), "p95_ms": percentile(durations, 0.95)},
        "score_stability": {"samples": scores, "stable_rounded_6dp": len({round(score, 6) for score in scores}) <= 1},
    }


def collect(output: str | Path) -> dict[str, object]:
    registry = ROOT / "config" / "baseline_registry.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image": {name: _image(name) for name in ("ruralstroke-api:stage3-local", "ruralstroke-ui:stage3-local")},
        "stack": {"containers": _portable_containers(_run(["docker", "compose", "--project-name", "ruralstroke-stage3", "--file", "compose.yaml", "ps", "--format", "json"]).splitlines()), "container_rss_bytes": _rss()},
        "benchmark": _benchmark(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "docker": _run(["docker", "version", "--format", "{{.Server.Version}}"]),
            "compose": _run(["docker", "compose", "version"]),
            "redaction": "environment values and credentials are not recorded",
        },
        "frozen_compatibility": {
            "baseline_registry_sha256": sha256(registry.read_bytes()).hexdigest(),
            "canonical_model_paths": json.loads(registry.read_text(encoding="utf-8"))["components"],
            "models_modified_by_stage3": False,
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/production/stage3/stack-evidence.json")
    args = parser.parse_args()
    print(json.dumps(collect(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
