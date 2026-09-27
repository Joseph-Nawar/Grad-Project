"""Offline-first Streamlit collector edge application."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
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
from rural_stroke_assist.ui.common import apply_theme, render_assessment_result, render_preassessment_quality, render_product_header


LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title="RuralStroke-Triage Edge Collector", page_icon="🩺", layout="wide")
apply_theme(st)


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


def _metadata() -> MetadataInput:
    st.markdown("<div class='rsa-section-kicker'>Contextual / background risk</div>", unsafe_allow_html=True)
    first, second = st.columns(2)
    with first:
        age = st.number_input("Age", min_value=0, max_value=120, value=60)
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart disease")
        glucose = st.number_input("Average glucose", min_value=0.0, value=100.0)
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    with second:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        married = st.selectbox("Ever married", ["No", "Yes"])
        work = st.selectbox("Work type", ["Govt_job", "Never_worked", "Private", "Self-employed", "children"])
        residence = st.selectbox("Residence type", ["Rural", "Urban"])
        smoking = st.selectbox("Smoking status", ["Unknown", "formerly smoked", "never smoked", "smokes"])
    return MetadataInput(
        age=float(age), hypertension=int(hypertension), heart_disease=int(heart_disease),
        avg_glucose_level=float(glucose), bmi=float(bmi), gender=gender,
        ever_married=married, work_type=work, Residence_type=residence,
        smoking_status=smoking,
    )


def _symptoms() -> AcuteStrokeSymptoms:
    st.markdown("<div class='rsa-section-kicker'>Deterministic acute symptom evidence</div>", unsafe_allow_html=True)
    columns = st.columns(3)
    fields = (
        ("face_drooping", "Face drooping or asymmetry"),
        ("arm_weakness", "Arm weakness or drift"),
        ("speech_difficulty", "Speech difficulty"),
        ("balance_or_coordination_loss", "Balance or coordination loss"),
        ("vision_disturbance", "Vision disturbance"),
        ("sudden_severe_headache", "Sudden severe headache"),
        ("confusion_or_understanding_difficulty", "Confusion or difficulty understanding"),
    )
    values = {}
    for index, (key, label) in enumerate(fields):
        with columns[index % len(columns)]:
            values[key] = st.checkbox(label)
    known = st.checkbox("Onset time is known")
    onset = st.number_input("Minutes since symptom onset", min_value=0, value=0) if known else None
    values["symptoms_resolved"] = st.checkbox("Symptoms have resolved")
    return AcuteStrokeSymptoms(**values, symptom_onset_minutes=onset)


def _run_sync(workflow: OfflineWorkflow) -> None:
    client = ApiClient()
    token_file = os.getenv("RURALSTROKE_API_TOKEN_FILE")
    provider = FileTokenProvider(Path(token_file)) if token_file else StaticTokenProvider(client.token)
    worker = SyncWorker(
        store=workflow.store,
        transport=ApiClientTransport(client),
        token_provider=provider,
        worker_id=f"streamlit-{os.getpid()}",
    )
    for _ in range(8):
        outcome = worker.run_once(now=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        if outcome.event_id is None:
            break


def _render_new_local_case(workflow: OfflineWorkflow) -> None:
    st.markdown("## Capture evidence locally")
    st.caption("Cases and media are saved to the local edge store before synchronization.")
    with st.form("local_case", clear_on_submit=False):
        top = st.columns(3)
        with top[0]:
            collector = st.text_input("Collector identity", value="Rural health volunteer")
        with top[1]:
            facility = st.text_input("Facility", value="Local health post")
        with top[2]:
            patient_code = st.text_input("Pseudonymous case code", placeholder="e.g. EDGE-014")
        metadata = _metadata()
        st.divider()
        symptoms = _symptoms()
        st.divider()
        face_col, speech_col = st.columns(2, gap="large")
        with face_col:
            st.markdown("### Visual evidence")
            face = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
            with st.expander("Optional camera capture"):
                camera = st.camera_input("Capture face image")
            face = face or camera
        with speech_col:
            st.markdown("### Speech evidence")
            st.caption("Dysarthria-related speech evidence from the English sample branch.")
            audio = st.file_uploader("Upload a speech recording", type=["wav", "mp3", "ogg"])
        render_preassessment_quality(
            st,
            face.getvalue() if face else None,
            audio.getvalue() if audio else None,
        )
        confirm = st.checkbox("I reviewed the case information and selected evidence.")
        save = st.form_submit_button("Save local draft", type="primary")
    if not save:
        return
    if not collector.strip() or not facility.strip():
        st.error("Enter collector identity and facility before saving locally.")
        return
    if not confirm:
        st.error("Review the case information and selected evidence before saving locally.")
        return
    try:
        case = workflow.create_draft(
            collector_identity=collector,
            facility=facility,
            patient_code=patient_code or None,
            assessment_input=AssessmentInput(session_id="collector-edge", metadata=metadata, acute_symptoms=symptoms),
            face_bytes=face.getvalue() if face else None,
            audio_bytes=audio.getvalue() if audio else None,
            face_filename=face.name if face else None,
            audio_filename=audio.name if audio else None,
            face_media_type=face.type if face else None,
            audio_media_type=audio.type if audio else None,
        )
        st.session_state.case_id = case.case_id
        st.rerun()
    except Exception as exc:
        LOGGER.exception("Local draft save failed")
        st.error("The local draft could not be saved. Check the selected files and available local storage.")


def main() -> None:
    workflow = get_workflow()
    workflow.store.cleanup_synchronized_media(
        grace_seconds=float(os.getenv("RURALSTROKE_MEDIA_GRACE_SECONDS", "86400")),
        now=datetime.now(timezone.utc).timestamp(),
    )
    online = _central_online()
    render_product_header(
        st,
        role="edge",
        active_step=1 if not st.session_state.get("case_id") else 2,
        connectivity=("Central service online" if online else "Offline — assessment and local storage remain available", "online" if online else "offline"),
    )
    counts = workflow.store.sync_counts()
    with st.sidebar:
        st.markdown("### Local sync queue")
        st.metric("Pending", counts.get("PENDING", 0))
        st.metric("Retrying", counts.get("RETRY_WAIT", 0))
        st.caption(
            f"Auth blocked {counts.get('BLOCKED_AUTH', 0)} · Conflicts {counts.get('CONFLICT', 0)} · Failed {counts.get('DEAD_LETTER', 0)}"
        )
    st.session_state.setdefault("case_id", None)
    case = workflow.store.get_case(st.session_state.case_id) if st.session_state.case_id else None
    if case is None:
        _render_new_local_case(workflow)
        return

    st.markdown(f"## Local case {case.case_id[:8].upper()}")
    st.caption(f"Workflow · {case.workflow_state.value.replace('_', ' ').title()} · Synchronization · {case.sync_state.value.replace('_', ' ').title()}")
    if case.sync_state.value == "SYNCED":
        st.success("Synchronized successfully")
    elif not online:
        st.info("Offline — assessment and local storage remain available. Sync will wait for reconnection.")
    elif case.sync_state.value == "RETRY_WAIT":
        st.info("Central service is online. Synchronization is queued for the next retry; the local assessment remains saved.")
    elif case.sync_state.value in {"PENDING", "SYNCING"}:
        st.info("Local assessment is saved. Synchronization is waiting for the central service.")

    actions = st.columns([1, 1, 2])
    if case.workflow_state.value == "DRAFT":
        with actions[0]:
            if st.button("Assess locally", type="primary", use_container_width=True):
                try:
                    with st.spinner("Analyzing multimodal evidence locally… Visual, speech, symptom and contextual branches are being evaluated."):
                        workflow.assess(case.case_id)
                    st.rerun()
                except Exception:
                    LOGGER.exception("Local assessment failed")
                    st.error("Local assessment could not be completed. Review the input quality and try again.")
    if case.workflow_state.value == "ASSESSED":
        with actions[0]:
            if st.button("Queue for synchronization", type="primary", use_container_width=True):
                workflow.queue(case.case_id)
                st.rerun()
    if case.workflow_state.value == "QUEUED":
        if online:
            with actions[1]:
                if st.button("Synchronize now", type="primary", use_container_width=True):
                    try:
                        _run_sync(workflow)
                        st.rerun()
                    except Exception:
                        LOGGER.exception("Synchronization attempt failed")
                        st.error("Synchronization did not finish. The local case remains saved for retry.")
        else:
            with actions[1]:
                if st.button("Recheck connection", use_container_width=True):
                    try:
                        if _central_online():
                            _run_sync(workflow)
                            st.rerun()
                        st.info("The central service is still offline. The local assessment remains saved.")
                    except Exception:
                        LOGGER.exception("Synchronization retry failed")
                        st.error("Synchronization did not finish. The local assessment remains saved for retry.")
    if case.assessment_envelope is not None:
        st.markdown("### Saved local assessment")
        render_assessment_result(
            st,
            case.assessment_envelope.result,
            runtime_provenance=case.assessment_envelope.provenance,
            include_limitations=False,
            compact=False,
        )
    with st.expander("Local case context", expanded=False):
        st.write(f"**Facility:** {case.facility}")
        st.write(f"**Pseudonymous case code:** {case.patient_code or 'Not provided'}")
        st.json(case.assessment_input)
    if case.workflow_state.value in {"SYNCED", "DEAD_LETTER", "CONFLICT"}:
        if st.button("Start another local assessment"):
            st.session_state.case_id = None
            st.rerun()


if __name__ == "__main__":
    main()
