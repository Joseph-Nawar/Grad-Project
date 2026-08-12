from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
import pytest

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.offline.contracts import WorkflowState
from rural_stroke_assist.offline.store import SQLiteOfflineStore
from rural_stroke_assist.offline.workflow import OfflineWorkflow


@pytest.mark.integration
def test_true_offline_edge_runs_real_assessment_service_and_queues_path_free_snapshot(
    tmp_path: Path,
) -> None:
    image = io.BytesIO()
    Image.new("RGB", (160, 160), color=(180, 160, 140)).save(image, format="PNG")
    workflow = OfflineWorkflow(
        store=SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media"),
        attachment_store=AttachmentStore(tmp_path / "media"),
        assessment_service=create_default_assessment_service(),
    )
    workflow.initialize()
    case = workflow.create_draft(
        collector_identity="collector",
        facility="post",
        assessment_input=AssessmentInput(
            session_id="offline-real-service",
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

    assessed = workflow.assess(case.case_id)
    events = workflow.queue(case.case_id)

    assert assessed.workflow_state is WorkflowState.ASSESSED
    assert assessed.assessment_hash is not None
    assert assessed.assessment_envelope is not None
    assert len(events) == 4
