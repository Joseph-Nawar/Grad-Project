from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess
import time

import httpx
from PIL import Image
import pytest

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.client import ApiClient
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.offline.http import ApiClientTransport
from rural_stroke_assist.offline.store import SQLiteOfflineStore
from rural_stroke_assist.offline.token import StaticTokenProvider
from rural_stroke_assist.offline.worker import SyncWorker
from rural_stroke_assist.offline.workflow import OfflineWorkflow

ROOT = Path(__file__).resolve().parents[2]


def _compose(*arguments: str) -> None:
    subprocess.run(
        [
            "docker",
            "compose",
            "--project-name",
            "ruralstroke-stage3",
            "--file",
            "compose.yaml",
            *arguments,
        ],
        cwd=ROOT,
        env={**os.environ, "RURALSTROKE_DEMO_SECRET_DIR": str(ROOT / ".runtime-secrets")},
        check=True,
        timeout=120,
    )


def _wait_for_api() -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            httpx.get("http://127.0.0.1:8000/health/ready", timeout=2).raise_for_status()
            return
        except httpx.HTTPError:
            time.sleep(2)
    raise AssertionError("Stage 3A API did not recover after the true disconnect check")


@pytest.mark.integration
def test_true_central_disconnect_preserves_local_assessment_and_syncs_after_reconnect(
    tmp_path: Path,
) -> None:
    token_file = ROOT / ".runtime-secrets" / "demo_token"
    if not token_file.is_file():
        pytest.skip("Stage 3A Compose demo token is not available")
    try:
        httpx.get("http://127.0.0.1:8000/health/ready", timeout=2).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Stage 3A central API is unavailable: {exc}")

    _compose("stop", "api")
    try:
        image = io.BytesIO()
        Image.new("RGB", (160, 160), color=(180, 160, 140)).save(image, format="PNG")
        store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
        workflow = OfflineWorkflow(
            store=store,
            attachment_store=AttachmentStore(tmp_path / "media"),
            assessment_service=create_default_assessment_service(),
        )
        workflow.initialize()
        local_case = workflow.create_draft(
            collector_identity="true-disconnect-demo",
            facility="Local health post",
            assessment_input=AssessmentInput(
                session_id="true-central-disconnect",
                metadata=MetadataInput(
                    age=60,
                    hypertension=0,
                    heart_disease=0,
                    avg_glucose_level=100,
                    bmi=24,
                    gender="Male",
                    ever_married="Yes",
                    work_type="Private",
                    Residence_type="Rural",
                    smoking_status="Unknown",
                ),
                acute_symptoms=AcuteStrokeSymptoms(face_drooping=True),
            ),
            face_bytes=image.getvalue(),
            face_filename="synthetic.png",
            face_media_type="image/png",
        )
        assessed = workflow.assess(local_case.case_id)
        workflow.queue(local_case.case_id)
        assert assessed.assessment_hash
        queued_case = store.get_case(local_case.case_id)
        assert queued_case.workflow_state.value == "QUEUED"
        assert queued_case.sync_state.value == "PENDING"
    finally:
        _compose("up", "-d", "api")
        _wait_for_api()

    token = token_file.read_text(encoding="utf-8").strip()
    client = ApiClient("http://127.0.0.1:8000", token)
    worker = SyncWorker(
        store=store,
        transport=ApiClientTransport(client),
        token_provider=StaticTokenProvider(token),
        worker_id="true-disconnect-worker",
    )
    for second in range(30):
        worker.run_once(now=f"2027-01-02T00:01:{second:02d}Z")

    final_case = store.get_case(local_case.case_id)
    assert final_case is not None
    assert final_case.sync_state.value == "SYNCED"
    assert client.get_case(local_case.case_id)["id"] == local_case.case_id
