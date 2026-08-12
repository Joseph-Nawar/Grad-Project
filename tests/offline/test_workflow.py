from __future__ import annotations

from pathlib import Path

from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.offline.contracts import WorkflowState
from rural_stroke_assist.offline.store import SQLiteOfflineStore
from rural_stroke_assist.offline.workflow import OfflineWorkflow
from rural_stroke_assist.testsupport import fake_assessment_result


class FakeAssessmentService:
    def __init__(self) -> None:
        self.calls = 0

    def assess(self, value):
        self.calls += 1
        assert isinstance(value, AssessmentInput)
        return fake_assessment_result()


def test_local_assessment_does_not_require_network_or_access_token(tmp_path: Path) -> None:
    store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    attachment_store = AttachmentStore(tmp_path / "media")
    service = FakeAssessmentService()
    workflow = OfflineWorkflow(store=store, attachment_store=attachment_store, assessment_service=service)
    workflow.initialize()

    case = workflow.create_draft(collector_identity="collector", facility="post", assessment_input=AssessmentInput(session_id="s1"), face_bytes=b"synthetic", face_filename="capture.jpg", face_media_type="image/jpeg")
    assessed = workflow.assess(case.case_id)

    assert service.calls == 1
    assert assessed.workflow_state is WorkflowState.ASSESSED
    assert assessed.assessment_hash
    assert assessed.assessment_envelope is not None
    assert "face_image_path" not in assessed.assessment_envelope.request
    assert "speech_audio_path" not in assessed.assessment_envelope.request


def test_queue_freezes_local_workflow_and_does_not_submit_over_http(tmp_path: Path) -> None:
    store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    workflow = OfflineWorkflow(store=store, attachment_store=AttachmentStore(tmp_path / "media"), assessment_service=FakeAssessmentService())
    workflow.initialize()
    case = workflow.create_draft(collector_identity="collector", facility="post", assessment_input=AssessmentInput(session_id="s1"))
    workflow.assess(case.case_id)

    events = workflow.queue(case.case_id)

    assert len(events) == 3
    assert store.get_case(case.case_id).workflow_state is WorkflowState.QUEUED


def test_clone_resolution_creates_new_draft_and_new_attachment_ids(tmp_path: Path) -> None:
    store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    workflow = OfflineWorkflow(store=store, attachment_store=AttachmentStore(tmp_path / "media"), assessment_service=FakeAssessmentService())
    workflow.initialize()
    original = workflow.create_draft(collector_identity="collector", facility="post", assessment_input=AssessmentInput(session_id="s1"), face_bytes=b"synthetic", face_filename="capture.jpg", face_media_type="image/jpeg")
    workflow.assess(original.case_id)
    workflow.queue(original.case_id)

    cloned = workflow.clone_as_draft(original.case_id)

    assert cloned.case_id != original.case_id
    assert cloned.workflow_state is WorkflowState.DRAFT
    assert cloned.assessment_envelope is None
    assert store.list_attachments(cloned.case_id)[0].attachment_id != store.list_attachments(original.case_id)[0].attachment_id
