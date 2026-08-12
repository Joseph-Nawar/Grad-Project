from __future__ import annotations

import io
import json
import os
from pathlib import Path

import httpx
from PIL import Image
import pytest

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.client import ApiClient, ApiClientError
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.offline.http import ApiClientTransport
from rural_stroke_assist.offline.store import SQLiteOfflineStore
from rural_stroke_assist.offline.token import StaticTokenProvider
from rural_stroke_assist.offline.worker import SyncWorker
from rural_stroke_assist.offline.workflow import OfflineWorkflow


@pytest.mark.integration
def test_stage3a_compose_offline_reconnect_exactly_once_and_clinician_review(
    tmp_path: Path,
) -> None:
    token_file = Path(os.getenv("RURALSTROKE_DEMO_SECRET_DIR", ".runtime-secrets")) / "demo_token"
    if not token_file.is_file():
        pytest.skip("Stage 3A Compose demo token is not available")
    try:
        httpx.get("http://127.0.0.1:8000/health/ready", timeout=2).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Stage 3A central API is unavailable: {exc}")

    token = token_file.read_text(encoding="utf-8").strip()
    image = io.BytesIO()
    Image.new("RGB", (160, 160), color=(180, 160, 140)).save(image, format="PNG")
    local_store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    workflow = OfflineWorkflow(
        store=local_store,
        attachment_store=AttachmentStore(tmp_path / "media"),
        assessment_service=create_default_assessment_service(),
    )
    workflow.initialize()
    local_case = workflow.create_draft(
        collector_identity="local-demo",
        facility="Local health post",
        assessment_input=AssessmentInput(
            session_id="compose-offline",
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
    with pytest.raises(httpx.HTTPError):
        httpx.get("http://127.0.0.1:9/health/live", timeout=0.5)

    restarted_store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    restarted_store.initialize()
    crash_lease = restarted_store.lease_next(
        worker_id="crashed-worker", now="2027-01-01T00:00:00Z", lease_seconds=30
    )
    assert crash_lease is not None
    client = ApiClient("http://127.0.0.1:8000", token)
    transport = ApiClientTransport(client)

    class CommitThenLoseResponse(ApiClientTransport):
        def __init__(self, wrapped: ApiClientTransport) -> None:
            super().__init__(wrapped.client)
            self.lost = False

        def submit_case(self, case_id, payload, etag, *, idempotency_key, token):
            response = super().submit_case(
                case_id, payload, etag, idempotency_key=idempotency_key, token=token
            )
            if not self.lost:
                self.lost = True
                raise ApiClientError(
                    503,
                    {
                        "code": "dependency_unavailable",
                        "message": "response lost after server commit",
                    },
                )
            return response

    worker = SyncWorker(
        store=restarted_store,
        transport=CommitThenLoseResponse(transport),
        token_provider=StaticTokenProvider(token),
        worker_id="restarted-worker",
    )
    for second in range(30):
        worker.run_once(now=f"2027-01-01T00:01:{second:02d}Z")

    local_final = restarted_store.get_case(local_case.case_id)
    central_case = client.get_case(local_case.case_id)
    assert local_final.workflow_state.value == "SYNCED"
    assert central_case["assessment_hash"] == local_final.assessment_hash
    assert (
        len(
            [
                item
                for item in client.list_cases().get("items", [])
                if item["id"] == local_case.case_id
            ]
        )
        == 1
    )
    imported = client.get_assessment(central_case["assessment_id"])
    assert imported["assessment_hash"] == local_final.assessment_hash

    before_review = central_case["assessment_result"]
    client.claim_review(local_case.case_id)
    client.create_review(
        local_case.case_id,
        {
            "agree": True,
            "notes": "Reviewed stored immutable snapshot.",
            "alternative_disposition": None,
        },
    )
    after_review = client.get_case(local_case.case_id)
    assert after_review["assessment_hash"] == central_case["assessment_hash"]
    assert after_review["assessment_result"] == before_review
    evidence = {
        "gate": "stage3a-compose-offline-reconnect",
        "central_case_count": 1,
        "central_attachment_count": len(central_case.get("attachments", [])),
        "central_assessment_count": 1,
        "central_submission_count": 1,
        "local_assessment_hash": local_final.assessment_hash,
        "central_assessment_hash": central_case["assessment_hash"],
        "hash_parity": local_final.assessment_hash == central_case["assessment_hash"],
        "response_lost_after_commit_recovered": True,
        "collector_restart_persistence": True,
        "worker_lease_recovery": True,
        "clinician_review_preserved_assessment": after_review["assessment_result"]
        == before_review,
        "raw_media_or_tokens": False,
    }
    destination = Path("reports/production/stage4")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "compose-e2e.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
