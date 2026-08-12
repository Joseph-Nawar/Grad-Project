from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from scripts.stack import wait_for_url


pytestmark = pytest.mark.integration
ROOT = Path(__file__).parents[2]


def _request(method: str, path: str, token: str, *, payload: object | None = None, headers: dict[str, str] | None = None) -> tuple[dict[str, object] | bytes | None, dict[str, str]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Authorization": f"Bearer {token}", **(headers or {})}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = Request(f"http://127.0.0.1:8000{path}", method=method, data=body, headers=request_headers)
    with urlopen(request, timeout=30) as response:
        content = response.read()
        if response.headers.get_content_type() == "application/json":
            return json.loads(content.decode("utf-8")), dict(response.headers.items())
        return content, dict(response.headers.items())


def test_compose_collector_assessment_submit_review_and_attachment_persist() -> None:
    token_path = ROOT / ".runtime-secrets" / "demo_token"
    if not token_path.is_file():
        pytest.fail("Run python scripts/stack.py up before the Compose E2E test.")
    token = token_path.read_text(encoding="utf-8").strip()
    case_id = str(uuid.uuid4())
    idempotency = lambda: str(uuid.uuid4())
    case, case_headers = _request(
        "POST",
        "/api/v1/cases",
        token,
        payload={"id": case_id, "facility": "Local health post", "patient_code": "demo-001", "assessment_input": {"session_id": "compose-e2e"}},
        headers={"Idempotency-Key": idempotency()},
    )
    assert isinstance(case, dict)
    attachment_request = Request(
        "http://127.0.0.1:8000/api/v1/attachments",
        method="POST",
        data=(
            f"--boundary\r\nContent-Disposition: form-data; name=case_id\r\n\r\n{case_id}\r\n"
            f"--boundary\r\nContent-Disposition: form-data; name=kind\r\n\r\naudio\r\n"
            "--boundary\r\nContent-Disposition: form-data; name=file; filename=demo.wav\r\nContent-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + b"compose-persisted-attachment\r\n--boundary--\r\n",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "multipart/form-data; boundary=boundary"},
    )
    with urlopen(attachment_request, timeout=30) as response:
        attachment = json.loads(response.read().decode("utf-8"))
    attachment_id = str(attachment["id"])
    assessment, _ = _request(
        "POST",
        "/api/v1/assessments",
        token,
        payload={
            "id": str(uuid.uuid4()),
            "case_id": case_id,
            "session_id": "compose-e2e",
            "audio_attachment_id": None,
            "face_attachment_id": None,
            "metadata": None,
            "acute_symptoms": {"face_drooping": True, "arm_weakness": True, "speech_difficulty": False, "balance_or_coordination_loss": False, "vision_disturbance": False, "sudden_severe_headache": False, "confusion_or_understanding_difficulty": False, "symptom_onset_minutes": 30, "symptoms_resolved": False},
        },
        headers={"Idempotency-Key": idempotency()},
    )
    assert isinstance(assessment, dict)
    _, assessed_headers = _request("GET", f"/api/v1/cases/{case_id}", token)
    etag = next(value for key, value in assessed_headers.items() if key.lower() == "etag")
    submitted, submit_headers = _request("POST", f"/api/v1/cases/{case_id}/submit", token, payload={"confirmed": True}, headers={"Idempotency-Key": idempotency(), "If-Match": etag})
    assert isinstance(submitted, dict) and submitted["status"] == "SUBMITTED"
    claimed, _ = _request("POST", f"/api/v1/cases/{case_id}/review-claim", token, headers={"Idempotency-Key": idempotency()})
    assert isinstance(claimed, dict) and claimed["status"] == "IN_REVIEW"
    reviewed, _ = _request("POST", f"/api/v1/cases/{case_id}/reviews", token, payload={"agree": True, "notes": "Compose persistence review", "alternative_disposition": None}, headers={"Idempotency-Key": idempotency()})
    assert isinstance(reviewed, dict)
    assert reviewed["agree"] is True

    subprocess.run(["docker", "compose", "--project-name", "ruralstroke-stage3", "--file", str(ROOT / "compose.yaml"), "restart", "api"], cwd=ROOT, check=True, env={**os.environ, "RURALSTROKE_DEMO_SECRET_DIR": str(ROOT / ".runtime-secrets")}, timeout=120)
    wait_for_url("http://127.0.0.1:8000/health/ready", timeout_seconds=120)
    persisted, _ = _request("GET", f"/api/v1/attachments/{attachment_id}", token)
    assert persisted == b"compose-persisted-attachment"
