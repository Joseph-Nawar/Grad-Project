from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.cases.contracts import CaseStatus
from rural_stroke_assist.cases.sqlite_repository import SQLiteCaseRepository
from rural_stroke_assist.cases.workflow_service import CaseWorkflowService
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.inference.registry import BaselineRegistry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = BaselineRegistry.from_file(ROOT / "config" / "baseline_registry.json")


@pytest.mark.integration
def test_real_assessment_persist_submit_review_reload(tmp_path: Path) -> None:
    face = pd.read_csv(REGISTRY.manifest_path("face_split"))
    speech = pd.read_csv(REGISTRY.manifest_path("speech_split"))
    metadata = pd.read_csv(REGISTRY.manifest_path("metadata_split"))
    row = metadata.loc[metadata["split"] == "test"].iloc[0]
    face_path = Path(face.loc[face["split"] == "test", "path"].iloc[0])
    audio_path = Path(speech.loc[speech["split"] == "test", "path"].iloc[0])
    workflow = CaseWorkflowService(
        repository=SQLiteCaseRepository(tmp_path / "cases.sqlite3"),
        attachment_store=AttachmentStore(tmp_path / "runtime"),
        assessment_service=create_default_assessment_service(),
    )
    workflow.initialize()
    case = workflow.create_draft(
        collector_identity="collector", facility="post", assessment_input=AssessmentInput(
            session_id="real", metadata=MetadataInput(age=row["age"], hypertension=row["hypertension"], heart_disease=row["heart_disease"], avg_glucose_level=row["avg_glucose_level"], bmi=row["bmi"], gender=row["gender"], ever_married=row["ever_married"], work_type=row["work_type"], Residence_type=row["Residence_type"], smoking_status=row["smoking_status"]), acute_symptoms=AcuteStrokeSymptoms(face_drooping=True, arm_weakness=True),
        ), face_bytes=face_path.read_bytes(), face_filename=face_path.name, face_media_type="image/jpeg", audio_bytes=audio_path.read_bytes(), audio_filename=audio_path.name, audio_media_type="audio/wav",
    )
    assessed = workflow.assess(case.case_id, actor="collector")
    assert assessed.status is CaseStatus.ASSESSED
    assert assessed.assessment_result and assessed.assessment_result["fusion"]
    submitted = workflow.submit(case.case_id, actor="collector", confirmed=True)
    workflow.begin_review(submitted.case_id, actor="clinician")
    reviewed = workflow.review(submitted.case_id, clinician_name="Specialist", medical_centre="Centre", agree=True, notes="Reviewed submitted evidence and agree with proposed urgency.")
    reloaded = workflow.get(reviewed.case_id)
    assert reloaded.status is CaseStatus.REVIEWED_AGREED
    assert reloaded.assessment_result == assessed.assessment_result
    assert reloaded.clinician_review is not None
