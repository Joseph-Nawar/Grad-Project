from __future__ import annotations

from pathlib import Path
from io import BytesIO
import wave

import pytest
from PIL import Image

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.cases.contracts import CaseStatus
from rural_stroke_assist.cases.sqlite_repository import SQLiteCaseRepository
from rural_stroke_assist.cases.workflow_service import CaseWorkflowService
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.testsupport import fake_assessment_result


ROOT = Path(__file__).resolve().parents[1]


def run_app(path: Path, tmp_path: Path) -> AppTest:
    import os

    os.environ["RURAL_STROKE_RUNTIME_DIR"] = str(tmp_path / path.stem)
    streamlit.cache_resource.clear()
    at = AppTest.from_file(str(path), default_timeout=30)
    at.run()
    return at


def test_collector_app_starts_with_guided_collection(tmp_path: Path) -> None:
    at = run_app(ROOT / "apps" / "collector_app.py", tmp_path)
    assert not at.exception
    assert at.title[0].value == "RuralStroke-Assist · Collector"
    assert any("minimum information" in item.value for item in at.info)
    assert any(item.label == "Start or save draft" for item in at.button)


def test_clinician_app_has_empty_queue_state(tmp_path: Path) -> None:
    at = run_app(ROOT / "apps" / "clinician_app.py", tmp_path)
    assert not at.exception
    assert at.title[0].value == "RuralStroke-Assist · Clinician review"
    assert any("No submitted cases" in item.value for item in at.info)


def test_clinician_app_displays_seeded_review_queue(tmp_path: Path) -> None:
    runtime = tmp_path / "clinician_app"
    workflow = CaseWorkflowService(
        repository=SQLiteCaseRepository(runtime / "cases.sqlite3"),
        attachment_store=AttachmentStore(runtime),
        assessment_service=type("FakeService", (), {"assess": lambda self, value: fake_assessment_result()})(),
    )
    workflow.initialize()
    case = workflow.create_draft(
        collector_identity="collector", facility="post", assessment_input=AssessmentInput(
            session_id="seed", metadata=MetadataInput(age=60, hypertension=0, heart_disease=0, avg_glucose_level=100, bmi=24, gender="Male", ever_married="Yes", work_type="Private", Residence_type="Rural", smoking_status="Unknown"),
        )
    )
    workflow.assess(case.case_id, actor="collector")
    workflow.submit(case.case_id, actor="collector", confirmed=True)
    at = run_app(ROOT / "apps" / "clinician_app.py", tmp_path)
    assert not at.exception
    assert any(item.label == "Begin review" for item in at.button)


def test_clinician_app_renders_seeded_media_and_humanized_context(tmp_path: Path) -> None:
    runtime = tmp_path / "clinician_app"
    workflow = CaseWorkflowService(
        repository=SQLiteCaseRepository(runtime / "cases.sqlite3"),
        attachment_store=AttachmentStore(runtime),
        assessment_service=type("FakeService", (), {"assess": lambda self, value: fake_assessment_result()})(),
    )
    workflow.initialize()
    face_stream = BytesIO()
    Image.new("RGB", (8, 8), "white").save(face_stream, format="PNG")
    audio_stream = BytesIO()
    with wave.open(audio_stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 1600)
    case = workflow.create_draft(
        collector_identity="collector", facility="post", assessment_input=AssessmentInput(
            session_id="seed", metadata=MetadataInput(age=60, hypertension=0, heart_disease=0, avg_glucose_level=100, bmi=24, gender="Male", ever_married="Yes", work_type="Govt_job", Residence_type="Rural", smoking_status="Unknown"),
        ), face_bytes=face_stream.getvalue(), face_filename="submitted.png", face_media_type="image/png", audio_bytes=audio_stream.getvalue(), audio_filename="submitted.wav", audio_media_type="audio/wav",
    )
    workflow.assess(case.case_id, actor="collector")
    workflow.submit(case.case_id, actor="collector", confirmed=True)
    at = run_app(ROOT / "apps" / "clinician_app.py", tmp_path)
    assert not at.exception
    assert any("Case " in item.value for item in at.markdown)
    assert any("Facial image evidence" in item.value for item in at.markdown)
    assert any("Speech recording evidence" in item.value for item in at.markdown)
    assert any("Government employment" in item.value for item in at.markdown)
    assert any(item.label == "Developer data" for item in at.expander)
    assert any(item.label == "Begin review" for item in at.button)


def test_clinician_app_review_form_is_reachable(tmp_path: Path) -> None:
    runtime = tmp_path / "clinician_app"
    workflow = CaseWorkflowService(
        repository=SQLiteCaseRepository(runtime / "cases.sqlite3"),
        attachment_store=AttachmentStore(runtime),
        assessment_service=type("FakeService", (), {"assess": lambda self, value: fake_assessment_result()})(),
    )
    workflow.initialize()
    case = workflow.create_draft(collector_identity="collector", facility="post", assessment_input=AssessmentInput(session_id="seed", metadata=MetadataInput(age=60, hypertension=0, heart_disease=0, avg_glucose_level=100, bmi=24, gender="Male", ever_married="Yes", work_type="Private", Residence_type="Rural", smoking_status="Unknown")))
    workflow.assess(case.case_id, actor="collector")
    workflow.submit(case.case_id, actor="collector", confirmed=True)
    at = run_app(ROOT / "apps" / "clinician_app.py", tmp_path)
    next(item for item in at.button if item.label == "Begin review").click()
    at.run()
    assert not at.exception
    assert any(item.label == "Decision" for item in at.radio)
    assert any(item.label == "Record clinician review" for item in at.button)


def test_collector_app_can_submit_existing_assessed_case(tmp_path: Path) -> None:
    runtime = tmp_path / "collector_app"
    workflow = CaseWorkflowService(
        repository=SQLiteCaseRepository(runtime / "cases.sqlite3"),
        attachment_store=AttachmentStore(runtime),
        assessment_service=type("FakeService", (), {"assess": lambda self, value: fake_assessment_result()})(),
    )
    workflow.initialize()
    case = workflow.create_draft(collector_identity="collector", facility="post", assessment_input=AssessmentInput(session_id="seed", metadata=MetadataInput(age=60, hypertension=0, heart_disease=0, avg_glucose_level=100, bmi=24, gender="Male", ever_married="Yes", work_type="Private", Residence_type="Rural", smoking_status="Unknown")))
    workflow.assess(case.case_id, actor="collector")
    at = run_app(ROOT / "apps" / "collector_app.py", tmp_path)
    next(item for item in at.button if item.label == "Resume selected case").click()
    at.run()
    review_checkbox = next(item for item in at.checkbox if "assessment snapshot" in item.label)
    review_checkbox.set_value(True)
    next(item for item in at.button if item.label == "Submit case for clinician review").click()
    at.run()
    assert not at.exception
    assert any("Submitted case" in item.value for item in at.success)

