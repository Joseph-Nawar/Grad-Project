"""Streamlit collector for the Stage 2 API."""

from __future__ import annotations

import streamlit as st

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.client import ApiClient, ApiClientError
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.ui.common import (
    apply_theme,
    render_assessment_result,
    render_preassessment_quality,
    render_product_header,
)
from rural_stroke_assist.ui.oidc import streamlit_api_client
from rural_stroke_assist.ui.payloads import build_case_assessment_input_payload
from rural_stroke_assist.ui.presentation import short_case_reference


st.set_page_config(page_title="RuralStroke-Triage Collector", page_icon="🩺", layout="wide")
apply_theme(st)


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def _metadata_form() -> MetadataInput:
    st.markdown("<div class='rsa-section-kicker'>Contextual / background risk</div>", unsafe_allow_html=True)
    row_one = st.columns(4)
    with row_one[0]:
        age = st.number_input("Age", min_value=0, max_value=120, value=60)
    with row_one[1]:
        hypertension = st.checkbox("Hypertension")
    with row_one[2]:
        heart_disease = st.checkbox("Heart disease")
    with row_one[3]:
        avg_glucose = st.number_input("Average glucose", min_value=0.0, value=100.0)

    row_two = st.columns(4)
    with row_two[0]:
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    with row_two[1]:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    with row_two[2]:
        ever_married = st.selectbox("Ever married", ["No", "Yes"])
    with row_two[3]:
        residence = st.selectbox("Residence type", ["Rural", "Urban"])

    row_three = st.columns(2)
    with row_three[0]:
        work_type = st.selectbox("Work type", ["Govt_job", "Never_worked", "Private", "Self-employed", "children"])
    with row_three[1]:
        smoking = st.selectbox("Smoking status", ["Unknown", "formerly smoked", "never smoked", "smokes"])
    return MetadataInput(
        age=age,
        hypertension=int(hypertension),
        heart_disease=int(heart_disease),
        avg_glucose_level=avg_glucose,
        bmi=bmi,
        gender=gender,
        ever_married=ever_married,
        work_type=work_type,
        Residence_type=residence,
        smoking_status=smoking,
    )


def _symptoms_form() -> AcuteStrokeSymptoms:
    st.markdown("<div class='rsa-section-kicker'>Deterministic acute symptom evidence</div>", unsafe_allow_html=True)
    columns = st.columns(3)
    symptoms: dict[str, bool] = {}
    fields = (
        ("face_drooping", "Face drooping or asymmetry"),
        ("arm_weakness", "Arm weakness or drift"),
        ("speech_difficulty", "Speech difficulty"),
        ("balance_or_coordination_loss", "Balance or coordination loss"),
        ("vision_disturbance", "Vision disturbance"),
        ("sudden_severe_headache", "Sudden severe headache"),
        ("confusion_or_understanding_difficulty", "Confusion or difficulty understanding"),
    )
    for index, (field, label) in enumerate(fields):
        with columns[index % len(columns)]:
            symptoms[field] = st.checkbox(label)
    onset_known = st.checkbox("Onset time is known")
    onset_minutes = st.number_input("Minutes since symptom onset", min_value=0, value=0) if onset_known else None
    resolved = st.checkbox("Symptoms have resolved")
    return AcuteStrokeSymptoms(**symptoms, symptom_onset_minutes=onset_minutes, symptoms_resolved=resolved)


def _build_input(metadata: MetadataInput, symptoms: AcuteStrokeSymptoms) -> AssessmentInput:
    return AssessmentInput(session_id="collector-session", metadata=metadata, acute_symptoms=symptoms)


def _error(exc: ApiClientError, action: str) -> None:
    if exc.status_code == 422:
        st.error(f"{action} could not be validated. Check the entered case details and try again.")
    elif exc.status_code >= 500:
        st.error(f"{action} is temporarily unavailable. Keep this draft and retry when the service is ready.")
    else:
        st.error(exc.payload.get("message", f"{action} could not be completed."))


def _start_case_form(client: ApiClient) -> None:
    st.markdown("## Capture a new assessment")
    st.caption("Use a pseudonymous code only. Contextual risk is background information; acute symptoms are evaluated separately.")
    with st.form("collection_form", clear_on_submit=False):
        context_tab, symptom_tab, media_tab = st.tabs(["Case & context", "Acute symptoms", "Media evidence"])
        with context_tab:
            st.markdown("<div class='rsa-section-kicker'>Case context</div>", unsafe_allow_html=True)
            identity_cols = st.columns(3)
            with identity_cols[0]:
                collector = st.text_input("Collector identity", value="Rural health volunteer")
            with identity_cols[1]:
                facility = st.text_input("Facility", value="Local health post")
            with identity_cols[2]:
                patient_code = st.text_input("Pseudonymous case code", placeholder="e.g. DEMO-014")
            metadata = _metadata_form()

        with symptom_tab:
            st.caption("These symptom inputs are evaluated as a deterministic acute-evidence branch.")
            symptoms = _symptoms_form()

        with media_tab:
            st.markdown("<div class='rsa-section-kicker'>Multimodal capture</div>", unsafe_allow_html=True)
            face_col, speech_col = st.columns(2, gap="large")
            with face_col:
                st.markdown("### Visual evidence")
                face_upload = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"], key="collector_face_upload")
                with st.expander("Optional camera capture"):
                    face_camera = st.camera_input("Capture face image", key="collector_face_camera")
                selected_face = face_upload or face_camera
            with speech_col:
                st.markdown("### Speech evidence")
                st.caption("Read the supplied English sentence once at a comfortable pace.")
                st.code("The quick brown fox jumps over the lazy dog.", language=None)
                st.info("This branch estimates dysarthria-related speech evidence from English TORGO recordings. Other languages and the sentence itself have not been clinically validated.")
                audio_upload = st.file_uploader("Upload a speech recording", type=["wav", "mp3", "ogg"], key="collector_audio_upload")
            render_preassessment_quality(
                st,
                selected_face.getvalue() if selected_face else None,
                audio_upload.getvalue() if audio_upload else None,
            )

        reviewed = st.checkbox("I reviewed the case information and selected evidence.")
        action_cols = st.columns([1, 1, 3])
        with action_cols[0]:
            save_draft = st.form_submit_button("Save draft")
        with action_cols[1]:
            run_assessment = st.form_submit_button("Run multimodal assessment", type="primary")

    if not (save_draft or run_assessment):
        return
    if not collector.strip() or not facility.strip():
        st.error("Enter collector identity and facility before saving this case.")
        return
    if not reviewed:
        st.error("Review the case information and selected evidence before continuing.")
        return

    try:
        assessment_input = _build_input(metadata, symptoms)
        case = client.create_case(
            case_id=None,
            facility=facility,
            patient_code=patient_code or None,
            assessment_input=build_case_assessment_input_payload(assessment_input),
        )
        st.session_state.current_case = case
        st.session_state.attachment_ids = {}
        if selected_face:
            uploaded = client.upload_attachment(
                case["id"], "face", selected_face.getvalue(),
                getattr(selected_face, "name", "capture.jpg"),
                getattr(selected_face, "type", "image/jpeg"),
            )
            st.session_state.attachment_ids["face"] = str(uploaded["id"])
        if audio_upload:
            uploaded = client.upload_attachment(
                case["id"], "audio", audio_upload.getvalue(),
                getattr(audio_upload, "name", "recording.wav"),
                getattr(audio_upload, "type", "audio/wav"),
            )
            st.session_state.attachment_ids["audio"] = str(uploaded["id"])
        if save_draft:
            st.success(f"Draft saved: {short_case_reference(case['id'])}. Run the assessment when ready.")
            st.rerun()
        _run_assessment(client, case)
    except ApiClientError as exc:
        _error(exc, "Case creation, media upload or assessment")


def _assessment_payload(case: dict[str, object]) -> dict[str, object]:
    input_data = case.get("assessment_input") or {}
    if not isinstance(input_data, dict):
        input_data = {}
    payload: dict[str, object] = {
        "case_id": case["id"],
        "face_attachment_id": st.session_state.get("attachment_ids", {}).get("face"),
        "audio_attachment_id": st.session_state.get("attachment_ids", {}).get("audio"),
    }
    if input_data.get("metadata") is not None:
        payload["metadata"] = input_data["metadata"]
    if input_data.get("acute_symptoms") is not None:
        payload["acute_symptoms"] = input_data["acute_symptoms"]
    if input_data.get("session_id"):
        payload["session_id"] = input_data["session_id"]
    return payload


def _run_assessment(client: ApiClient, case: dict[str, object]) -> None:
    with st.spinner("Analyzing multimodal evidence… Visual, speech, symptom and contextual branches are being evaluated."):
        client.create_assessment(_assessment_payload(case))
        st.session_state.current_case = client.get_case(str(case["id"]))
    st.success("Assessment completed and stored as an immutable snapshot.")
    st.rerun()


def _resume_drafts(client: ApiClient) -> None:
    with st.expander("Resume an existing draft", expanded=False):
        try:
            cases = client.list_cases().get("items", [])
            drafts = [item for item in cases if item.get("status") == "DRAFT"]
        except ApiClientError as exc:
            _error(exc, "Draft list")
            return
        if not drafts:
            st.caption("No saved drafts are available.")
            return
        selected = st.selectbox("Saved draft", drafts, format_func=lambda item: short_case_reference(item["id"]))
        if st.button("Open saved draft"):
            resumed = client.get_case(selected["id"])
            st.session_state.current_case = resumed
            st.session_state.attachment_ids = {
                str(item["kind"]): str(item["id"])
                for item in resumed.get("attachments", [])
            }
            st.rerun()


def _render_active_case(client: ApiClient, case: dict[str, object]) -> None:
    status = str(case.get("status", "UNKNOWN"))
    active_step = 1 if status == "DRAFT" else 2 if status == "ASSESSED" else 3
    render_product_header(st, role="collector", active_step=active_step)
    st.markdown(f"## {short_case_reference(str(case['id']))} · {status.replace('_', ' ').title()}")
    attachments = case.get("attachments") or []
    st.caption(f"{len(attachments)} media attachment(s) linked to this case · Review the case state before continuing.")

    if status == "DRAFT":
        left, right = st.columns([3, 1])
        with left:
            st.info("Draft saved. The assessment has not run yet.")
        with right:
            if st.button("Run multimodal assessment", type="primary", use_container_width=True):
                try:
                    _run_assessment(client, case)
                except ApiClientError as exc:
                    _error(exc, "Assessment")
    elif case.get("assessment_result"):
        render_assessment_result(st, case["assessment_result"])

    if status == "ASSESSED":
        st.markdown("### Send this immutable assessment snapshot for clinician review")
        confirmed = st.checkbox("I confirm the inputs and assessment snapshot are ready for clinician review.")
        if st.button("Submit to clinician review", type="primary", disabled=not confirmed):
            try:
                current, etag = client.get_case_with_etag(str(case["id"]))
                updated = client.submit_case(str(case["id"]), {"confirmed": True}, etag or f'"{current["version"]}"')
                st.session_state.current_case = updated
                st.success("Submitted to the clinician review queue.")
                st.rerun()
            except ApiClientError as exc:
                _error(exc, "Submission")
    elif status in {"SUBMITTED", "IN_REVIEW", "REVIEWED_AGREED", "REVIEWED_OVERRIDDEN"}:
        st.success("This assessment snapshot is available in the clinician review workflow.")

    if st.button("Start another assessment"):
        st.session_state.current_case = None
        st.session_state.attachment_ids = {}
        st.rerun()


def main() -> None:
    client = streamlit_api_client(st)
    if client is None:
        render_product_header(st, role="collector", active_step=1)
        st.info("Sign in with the assigned demonstration account to continue.")
        return
    st.session_state.setdefault("current_case", None)
    st.session_state.setdefault("attachment_ids", {})
    case = st.session_state.current_case
    if case is not None:
        try:
            case = client.get_case(str(case["id"]))
            st.session_state.current_case = case
            attachment_ids = dict(st.session_state.get("attachment_ids", {}))
            for attachment in case.get("attachments", []):
                attachment_ids.setdefault(str(attachment["kind"]), str(attachment["id"]))
            st.session_state.attachment_ids = attachment_ids
        except ApiClientError as exc:
            render_product_header(st, role="collector", active_step=1)
            _error(exc, "Case retrieval")
            return
        _render_active_case(client, case)
        return

    render_product_header(st, role="collector", active_step=1)
    _start_case_form(client)
    _resume_drafts(client)


if __name__ == "__main__":
    main()
