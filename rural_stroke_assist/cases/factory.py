"""Default local workflow construction."""

from pathlib import Path

from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.cases.sqlite_repository import SQLiteCaseRepository
from rural_stroke_assist.cases.workflow_service import CaseWorkflowService


def create_default_workflow_service(runtime_dir: str | Path = "runtime_data") -> CaseWorkflowService:
    from rural_stroke_assist.assessment.factory import create_default_assessment_service

    root = Path(runtime_dir)
    repository = SQLiteCaseRepository(root / "cases.sqlite3")
    attachments = AttachmentStore(root)
    service = CaseWorkflowService(repository=repository, attachment_store=attachments, assessment_service=create_default_assessment_service())
    service.initialize()
    return service


def create_review_workflow_service(runtime_dir: str | Path = "runtime_data") -> CaseWorkflowService:
    """Construct the clinician-side workflow without model-backed adapters."""
    root = Path(runtime_dir)
    repository = SQLiteCaseRepository(root / "cases.sqlite3")
    attachments = AttachmentStore(root)

    class ReadOnlyAssessmentService:
        def assess(self, _value: object) -> None:
            raise RuntimeError("Clinician workflow cannot run assessments.")

    service = CaseWorkflowService(repository=repository, attachment_store=attachments, assessment_service=ReadOnlyAssessmentService())
    service.initialize()
    return service
