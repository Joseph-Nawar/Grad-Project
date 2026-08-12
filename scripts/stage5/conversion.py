"""Isolated Stage 5 conversion workspace and reproducibility records."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rural_stroke_assist.stage5.contracts import CandidateEvidence


def stage5_workspace(root: Path) -> Path:
    return root / ".stage5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def conversion_python(root: Path) -> Path:
    location = stage5_workspace(root) / "conversion-env"
    executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return location / executable


def ensure_isolated_conversion_environment(
    root: Path,
    *,
    base_python: Path | None = None,
    requirements: Path | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Create and provision a disposable converter environment without changing production."""
    workspace = stage5_workspace(root)
    workspace.mkdir(parents=True, exist_ok=True)
    python = base_python or Path(sys.executable)
    env_dir = workspace / "conversion-env"
    commands: list[dict[str, Any]] = []

    if not conversion_python(root).is_file():
        command = [str(python), "-m", "venv", str(env_dir)]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout_seconds)
        commands.append({"command": command, "returncode": result.returncode, "stderr": result.stderr[-2000:]})
        if result.returncode != 0:
            return {
                "status": "FAILED",
                "failure_reason": "venv creation failed",
                "commands": commands,
                "python": str(python),
                "machine": platform.platform(),
            }

    requirement_file = requirements or root / "stage5" / "requirements-conversion.txt"
    command = [
        str(conversion_python(root)),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "-r",
        str(requirement_file),
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout_seconds)
    commands.append({"command": command, "returncode": result.returncode, "stderr": result.stderr[-4000:]})
    if result.returncode != 0:
        return {
            "status": "FAILED",
            "failure_reason": "isolated converter dependency installation failed",
            "commands": commands,
            "python": str(conversion_python(root)),
            "machine": platform.platform(),
        }
    version = subprocess.run(
        [str(conversion_python(root)), "-c", "import sys; print(sys.version)"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    commands.append({"command": "python --version", "returncode": version.returncode, "stdout": version.stdout.strip()})
    return {
        "status": "READY",
        "commands": commands,
        "python": str(conversion_python(root)),
        "machine": platform.platform(),
    }


def build_conversion_record(
    *,
    modality: str,
    candidate_id: str,
    source_artifact: str,
    source_sha256: str,
    candidate_artifact: str | None,
    converter_name: str,
    converter_version: str,
    runtime_name: str,
    runtime_version: str,
    conversion_options: dict[str, Any],
    input_signature: str,
    output_signature: str,
    failure_reason: str | None = None,
) -> CandidateEvidence:
    candidate_sha = _sha256(Path(candidate_artifact)) if candidate_artifact else None
    return CandidateEvidence(
        modality=modality,
        candidate_id=candidate_id,
        source_artifact=source_artifact,
        source_sha256=source_sha256,
        candidate_artifact=candidate_artifact or "",
        candidate_sha256=candidate_sha,
        converter_name=converter_name,
        converter_version=converter_version,
        runtime_name=runtime_name,
        runtime_version=runtime_version,
        conversion_options=conversion_options,
        input_signature=input_signature,
        output_signature=output_signature,
        conversion_status="FAILED" if failure_reason else "SUCCEEDED",
        failure_reason=failure_reason,
    )


def write_conversion_record(record: CandidateEvidence, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
