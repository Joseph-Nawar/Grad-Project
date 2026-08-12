from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_dockerfile_has_separate_non_root_api_and_ui_runtime_targets() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim-bookworm AS api-builder" in dockerfile
    assert "FROM api-builder AS api" in dockerfile
    assert "FROM python:3.11-slim-bookworm AS ui-builder" in dockerfile
    assert "FROM ui-builder AS ui" in dockerfile
    assert dockerfile.count("USER ruralstroke") >= 2
    assert "pip uninstall" in dockerfile


def test_ui_runtime_requirements_do_not_include_ml_serving_stack() -> None:
    requirements = (ROOT / "requirements-ui-runtime.txt").read_text(encoding="utf-8").lower()

    assert "streamlit==1.39.0" in requirements
    assert "tensorflow" not in requirements
    assert "autokeras" not in requirements
    assert "boto3" not in requirements


def test_dockerignore_excludes_secrets_datasets_dev_artifacts_and_codex_workspace() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    for pattern in ("data/", "notebooks/", "reports/", ".git/", ".venv/", "runtime_data/", ".env", "../codex_workspace/"):
        assert pattern in dockerignore
    assert "models/edge/face-trial-003-litert-fp32.tflite" in dockerignore
    assert "models/edge/speech-trial-001-onnx.onnx" in dockerignore


def test_compose_has_required_services_and_no_normal_backend_host_ports() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))

    assert {"postgres", "object-store", "object-store-init", "migrations", "api", "collector", "clinician"}.issubset(compose["services"])
    assert set(compose["networks"]) == {"frontend", "backend"}
    assert set(compose["volumes"]) >= {"postgres-data", "minio-data"}
    assert "ports" not in compose["services"]["postgres"]
    assert "ports" not in compose["services"]["object-store"]
    assert compose["services"]["migrations"]["restart"] == "no"
