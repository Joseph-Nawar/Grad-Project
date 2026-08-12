"""Offline-first Streamlit collector edge application."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import httpx
import streamlit as st

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.client import ApiClient
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.offline.http import ApiClientTransport
from rural_stroke_assist.offline.store import SQLiteOfflineStore
from rural_stroke_assist.offline.token import FileTokenProvider, StaticTokenProvider
from rural_stroke_assist.offline.worker import SyncWorker
from rural_stroke_assist.offline.workflow import OfflineWorkflow

st.set_page_config(page_title="RuralStroke-Assist - Collector Edge", page_icon="🩺", layout="wide")


@st.cache_resource
def get_workflow() -> OfflineWorkflow:
    database = Path(os.getenv("RURALSTROKE_OFFLINE_DB", "/var/lib/ruralstroke/collector.sqlite3"))
    media = Path(os.getenv("RURALSTROKE_OFFLINE_MEDIA", "/var/lib/ruralstroke/media"))
    workflow = OfflineWorkflow(
        store=SQLiteOfflineStore(database, media),
        attachment_store=AttachmentStore(media),
        assessment_service=create_default_assessment_service(),
    )
    workflow.initialize()
    return workflow


def _central_online() -> bool:
    url = os.getenv("RURALSTROKE_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        return httpx.get(f"{url}/health/live", timeout=0.5).is_success
    except httpx.HTTPError:
        return False


def _metadata(*, disabled: bool = False) -> MetadataInput:
    return MetadataInput(
        age=float(st.number_input("Age", min_value=0, max_value=120, value=60, disabled=disabled)),
        hypertension=int(st.checkbox("Hypertension", disabled=disabled)),
        heart_disease=int(st.checkbox("Heart disease", disabled=disabled)),
        avg_glucose_level=float(
            st.number_input("Average glucose", min_value=0.0, value=100.0, disabled=disabled)
        ),
        bmi=float(st.number_input("BMI", min_value=0.0, value=25.0, disabled=disabled)),
        gender=st.selectbox("Gender", ["Female", "Male", "Other"], disabled=disabled),
        ever_married=st.selectbox("Ever married", ["No", "Yes"], disabled=disabled),
        work_type=st.selectbox(
            "Work type",
            ["Govt_job", "Never_worked", "Private", "Self-employed", "children"],
            disabled=disabled,
        ),
        Residence_type=st.selectbox("Residence type", ["Rural", "Urban"], disabled=disabled),
        smoking_status=st.selectbox(
            "Smoking status",
            ["Unknown", "formerly smoked", "never smoked", "smokes"],
            disabled=disabled,
        ),
    )


def _run_sync(workflow: OfflineWorkflow) -> None:
    client = ApiClient()
    token_file = os.getenv("RURALSTROKE_API_TOKEN_FILE")
    token_provider = (
        FileTokenProvider(Path(token_file)) if token_file else StaticTokenProvider(client.token)
    )
    worker = SyncWorker(
        store=workflow.store,
        transport=ApiClientTransport(client),
        token_provider=token_provider,
        worker_id=f"streamlit-{os.getpid()}",
    )
    for _ in range(8):
        outcome = worker.run_once(
            now=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        if outcome.event_id is None:
            break


def main() -> None:
    st.title("RuralStroke-Assist - Collector Edge")
    st.caption("Local-first collection and assessment; central synchronization is separate.")
    workflow = get_workflow()
    workflow.store.cleanup_synchronized_media(
        grace_seconds=float(os.getenv("RURALSTROKE_MEDIA_GRACE_SECONDS", "86400")),
        now=datetime.now(timezone.utc).timestamp(),
    )
    if _central_online():
        st.success("Central API online; local work remains usable if connectivity changes.")
    else:
        st.info("Offline: local case creation and assessment remain available.")
    counts = workflow.store.sync_counts()
    st.caption(
        f"Pending {counts.get('PENDING', 0)} | Retrying {counts.get('RETRY_WAIT', 0)} | Auth blocked {counts.get('BLOCKED_AUTH', 0)} | Conflicts {counts.get('CONFLICT', 0)} | Failed {counts.get('DEAD_LETTER', 0)}"
    )

    st.session_state.setdefault("case_id", None)
    case = workflow.store.get_case(st.session_state.case_id) if st.session_state.case_id else None
    locked = case is not None and case.workflow_state.value != "DRAFT"
    with st.form("local_case"):
        collector = st.text_input(
            "Collector identity", value="Rural health volunteer", disabled=locked
        )
        facility = st.text_input("Facility", value="Local health post", disabled=locked)
        patient_code = st.text_input("Optional pseudonymous patient code", disabled=locked)
        metadata = _metadata(disabled=locked)
        symptoms = AcuteStrokeSymptoms(
            face_drooping=st.checkbox("Face drooping or asymmetry", disabled=locked),
            arm_weakness=st.checkbox("Arm weakness or drift", disabled=locked),
            speech_difficulty=st.checkbox("Speech difficulty", disabled=locked),
            balance_or_coordination_loss=st.checkbox(
                "Balance or coordination loss", disabled=locked
            ),
            vision_disturbance=st.checkbox("Vision disturbance", disabled=locked),
            sudden_severe_headache=st.checkbox("Sudden severe headache", disabled=locked),
            confusion_or_understanding_difficulty=st.checkbox(
                "Confusion or difficulty understanding", disabled=locked
            ),
            symptoms_resolved=st.checkbox("Symptoms have resolved", disabled=locked),
        )
        face = st.file_uploader("Face image", type=["jpg", "jpeg", "png"], disabled=locked)
        audio = st.file_uploader("Speech recording", type=["wav", "mp3", "ogg"], disabled=locked)
        save = st.form_submit_button("Save local draft", disabled=locked)
    if save:
        case = workflow.create_draft(
            collector_identity=collector,
            facility=facility,
            patient_code=patient_code or None,
            assessment_input=AssessmentInput(
                session_id="collector-edge", metadata=metadata, acute_symptoms=symptoms
            ),
            face_bytes=face.getvalue() if face else None,
            audio_bytes=audio.getvalue() if audio else None,
            face_filename=face.name if face else None,
            audio_filename=audio.name if audio else None,
            face_media_type=face.type if face else None,
            audio_media_type=audio.type if audio else None,
        )
        st.session_state.case_id = case.case_id
        st.success("Local draft saved.")

    if case:
        st.write(
            f"Local workflow: `{case.workflow_state.value}` | Synchronization: `{case.sync_state.value}`"
        )
    if st.button(
        "Restore authentication and retry",
        disabled=case is None or case.sync_state.value != "BLOCKED_AUTH",
    ):
        _run_sync(workflow)
        st.rerun()
    if st.button(
        "Clone as new local draft",
        disabled=case is None or case.workflow_state.value not in {"CONFLICT", "DEAD_LETTER"},
    ):
        cloned = workflow.clone_as_draft(case.case_id)
        st.session_state.case_id = cloned.case_id
        st.rerun()
    if st.button("Assess locally", disabled=case is None or case.workflow_state.value != "DRAFT"):
        workflow.assess(case.case_id)
        st.rerun()
    if st.button(
        "Queue for sync", disabled=case is None or case.workflow_state.value != "ASSESSED"
    ):
        workflow.queue(case.case_id)
        st.rerun()
    if st.button(
        "Retry synchronization", disabled=case is None or case.workflow_state.value != "QUEUED"
    ):
        _run_sync(workflow)
        st.rerun()


if __name__ == "__main__":
    main()
