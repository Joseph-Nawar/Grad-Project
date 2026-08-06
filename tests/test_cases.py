from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from rural_stroke_assist.assessment.contracts import AssessmentResult, AssessmentStatus, AssessmentTimings, FusionEvidence, ModalityExecution
from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.cases.contracts import AttachmentKind, Case, CaseStatus, utc_now
from rural_stroke_assist.cases.exceptions import AttachmentValidationError, ImmutableSnapshotError, InvalidTransitionError
from rural_stroke_assist.cases.report_builder import render_json, render_markdown
from rural_stroke_assist.cases.serialization import assessment_result_to_dict
from rural_stroke_assist.cases.sqlite_repository import SQLiteCaseRepository
from rural_stroke_assist.cases.workflow_service import CaseWorkflowService
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.contracts import ModalityEvidence
from rural_stroke_assist.inference.metadata_adapter import MetadataInput


def result() -> AssessmentResult:
    evidence = ModalityEvidence(
        modality="acute_symptoms", available=True, score=0.85,
        score_semantics="deterministic_acute_symptom_evidence", label="URGENT",
        provenance="rural_stroke_assist/modules/acute_symptom_module.py",
    )
    execution = ModalityExecution("acute_symptoms", evidence, 10)
    return AssessmentResult(
        status=AssessmentStatus.PARTIAL, modality_executions={"acute_symptoms": execution},
        fusion=FusionEvidence(0.85, "URGENT", {"acute_symptoms": 0.85}, {"acute_symptoms": 1.0}, (), "rural_stroke_assist/modules/fusion_module.py"),
        explanations=(), warnings=("warning",), timings=AssessmentTimings({"acute_symptoms": 10}, 10, 10, 30), provenance=("source",),
    )


def assessment_input() -> AssessmentInput:
    return AssessmentInput(
        session_id="session", metadata=MetadataInput(age=60, hypertension=0, heart_disease=0, avg_glucose_level=100, bmi=24, gender="Male", ever_married="Yes", work_type="Private", Residence_type="Rural", smoking_status="Unknown"),
        acute_symptoms=AcuteStrokeSymptoms(face_drooping=True),
    )


class FakeAssessmentService:
    def __init__(self) -> None:
        self.calls = 0

    def assess(self, value: AssessmentInput) -> AssessmentResult:
        self.calls += 1
        return result()


def service(tmp_path: Path) -> tuple[CaseWorkflowService, FakeAssessmentService, SQLiteCaseRepository]:
    repository = SQLiteCaseRepository(tmp_path / "cases.sqlite3")
    fake = FakeAssessmentService()
    workflow = CaseWorkflowService(repository=repository, attachment_store=AttachmentStore(tmp_path / "runtime"), assessment_service=fake)
    workflow.initialize()
    return workflow, fake, repository


def test_case_lifecycle_and_review_rules(tmp_path: Path) -> None:
    workflow, fake, _ = service(tmp_path)
    case = workflow.create_draft(collector_identity="collector", facility="post", assessment_input=assessment_input(), face_bytes=b"face", face_media_type="image/jpeg")
    assert case.status is CaseStatus.DRAFT
    assessed = workflow.assess(case.case_id, actor="collector")
    assert assessed.status is CaseStatus.ASSESSED
    with pytest.raises(InvalidTransitionError):
        workflow.submit(case.case_id, actor="collector", confirmed=False)
    submitted = workflow.submit(case.case_id, actor="collector", confirmed=True)
    assert submitted.status is CaseStatus.SUBMITTED
    reviewing = workflow.begin_review(case.case_id, actor="clinician")
    assert reviewing.status is CaseStatus.IN_REVIEW
    with pytest.raises(InvalidTransitionError):
        workflow.review(case.case_id, clinician_name="Dr", medical_centre="Centre", agree=False, notes="override")
    reviewed = workflow.review(case.case_id, clinician_name="Dr", medical_centre="Centre", agree=False, notes="Different urgency is appropriate.", alternative_disposition="Urgent transfer")
    assert reviewed.status is CaseStatus.REVIEWED_OVERRIDDEN
    assert fake.calls == 1


def test_submitted_assessment_snapshot_is_immutable(tmp_path: Path) -> None:
    workflow, _, repository = service(tmp_path)
    case = workflow.create_draft(collector_identity="collector", facility="post", assessment_input=assessment_input())
    assessed = workflow.assess(case.case_id, actor="collector")
    submitted = workflow.submit(assessed.case_id, actor="collector", confirmed=True)
    changed = submitted.with_update(assessment_result={"changed": True})
    with pytest.raises(ImmutableSnapshotError):
        repository.save(changed)


def test_attachment_store_uses_managed_names_and_rejects_bad_inputs(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path / "runtime", max_bytes=10)
    reference = store.save("case-id", AttachmentKind.FACE, b"123", filename="patient-name.png", media_type="image/png")
    assert "patient-name" not in reference.relative_path
    assert store.resolve(reference).is_file()
    with pytest.raises(AttachmentValidationError):
        store.save("case-id", AttachmentKind.FACE, b"123", filename="bad.exe", media_type="application/octet-stream")
    with pytest.raises(AttachmentValidationError):
        store.save("case-id", AttachmentKind.FACE, b"12345678901", filename="a.jpg", media_type="image/jpeg")


def test_sqlite_repository_has_deterministic_order_and_concurrent_writes(tmp_path: Path) -> None:
    workflow, _, repository = service(tmp_path)
    ids: list[str] = []
    lock = threading.Lock()

    def create() -> None:
        case = workflow.create_draft(collector_identity="collector", facility="post", assessment_input=assessment_input())
        with lock:
            ids.append(case.case_id)

    threads = [threading.Thread(target=create) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(repository.list()) == 4
    assert [case.case_id for case in repository.list()] == [case.case_id for case in repository.list()]


def test_serialization_preserves_result_and_reports() -> None:
    snapshot = assessment_result_to_dict(result())
    assert snapshot["fusion"]["evidence_score"] == 0.85
    assert "provenance" in snapshot
    case = Case("id", None, "collector", "post", {}, (), snapshot, utc_now(), utc_now(), None, CaseStatus.ASSESSED)
    assert json.loads(render_json(case))["assessment_result"]["fusion"]["risk_band"] == "URGENT"
    assert "not a clinical diagnosis" in render_markdown(case)
