"""Streamlit collector client for the Stage 2 API."""

from __future__ import annotations

import streamlit as st

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.client import ApiClient, ApiClientError
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.ui.common import apply_theme, render_assessment_result, render_preassessment_quality
from rural_stroke_assist.ui.presentation import short_case_reference


st.set_page_config(page_title="RuralStroke-Assist Collector", page_icon="🩺", layout="wide")
apply_theme(st)


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def _metadata_form() -> MetadataInput:
    return MetadataInput(age=st.number_input("Age", min_value=0, max_value=120, value=60), hypertension=int(st.checkbox("Hypertension")), heart_disease=int(st.checkbox("Heart disease")), avg_glucose_level=st.number_input("Average glucose", min_value=0.0, value=100.0), bmi=st.number_input("BMI", min_value=0.0, value=25.0), gender=st.selectbox("Gender", ["Female", "Male", "Other"]), ever_married=st.selectbox("Ever married", ["No", "Yes"]), work_type=st.selectbox("Work type", ["Govt_job", "Never_worked", "Private", "Self-employed", "children"]), Residence_type=st.selectbox("Residence type", ["Rural", "Urban"]), smoking_status=st.selectbox("Smoking status", ["Unknown", "formerly smoked", "never smoked", "smokes"]))


def _build_input(metadata: MetadataInput, symptoms: AcuteStrokeSymptoms) -> AssessmentInput:
    return AssessmentInput(session_id="collector-session", metadata=metadata, acute_symptoms=symptoms)


def _error(exc: ApiClientError) -> None:
    st.error(exc.payload.get("message", "The API request failed."))


def main() -> None:
    client = get_client()
    st.title("RuralStroke-Assist · Collector")
    st.caption("Guided collection for research screening and triage support")
    st.info("Collect only the minimum information needed for this local demonstration. Do not enter names, national IDs, or addresses.")
    st.session_state.setdefault("current_case", None)
    st.session_state.setdefault("attachment_ids", {})

    with st.form("collection_form"):
        collector = st.text_input("Collector identity", value="Rural health volunteer")
        facility = st.text_input("Facility", value="Local health post")
        patient_code = st.text_input("Optional pseudonymous patient code")
        st.subheader("Contextual metadata")
        metadata = _metadata_form()
        st.subheader("FAST / BE-FAST symptoms")
        onset_known = st.checkbox("Onset time is known")
        onset_minutes = st.number_input("Minutes since symptom onset", min_value=0, value=0) if onset_known else None
        symptoms = AcuteStrokeSymptoms(face_drooping=st.checkbox("Face drooping or asymmetry"), arm_weakness=st.checkbox("Arm weakness or drift"), speech_difficulty=st.checkbox("Speech difficulty"), balance_or_coordination_loss=st.checkbox("Balance or coordination loss"), vision_disturbance=st.checkbox("Vision disturbance"), sudden_severe_headache=st.checkbox("Sudden severe headache"), confusion_or_understanding_difficulty=st.checkbox("Confusion or difficulty understanding"), symptom_onset_minutes=onset_minutes, symptoms_resolved=st.checkbox("Symptoms have resolved"))
        st.subheader("Face and speech")
        st.caption("Sample sentence: ‘The quick brown fox jumps over the lazy dog.’")
        st.info("The speech branch was developed using English TORGO recordings. Use the provided English sentence for this demonstration; Arabic and other languages have not been validated for this branch. The sentence itself is not clinically validated.")
        face_upload = st.camera_input("Capture face image") or st.file_uploader("Or upload a face image", type=["jpg", "jpeg", "png"])
        audio_upload = st.file_uploader("Upload a speech recording", type=["wav", "mp3", "ogg"])
        render_preassessment_quality(st, face_upload.getvalue() if face_upload else None, audio_upload.getvalue() if audio_upload else None)
        confirm_inputs = st.checkbox("I reviewed the collected inputs")
        start = st.form_submit_button("Start or save draft")

    if start:
        try:
            assessment_input = _build_input(metadata, symptoms)
            case = client.create_case(case_id=None, facility=facility, patient_code=patient_code or None, assessment_input=assessment_input.model_dump(mode="json"))
            attachment_ids: dict[str, str] = {}
            if face_upload:
                attachment_ids["face"] = str(client.upload_attachment(case["id"], "face", face_upload.getvalue(), getattr(face_upload, "name", "capture.jpg"), getattr(face_upload, "type", "image/jpeg"))["id"])
            if audio_upload:
                attachment_ids["audio"] = str(client.upload_attachment(case["id"], "audio", audio_upload.getvalue(), getattr(audio_upload, "name", "recording.wav"), getattr(audio_upload, "type", "audio/wav"))["id"])
            st.session_state.current_case = case
            st.session_state.attachment_ids = attachment_ids
            st.success(f"Draft saved: {short_case_reference(case['id'])}")
            if not confirm_inputs:
                st.warning("Input review is not confirmed; review is required before submission.")
        except ApiClientError as exc:
            _error(exc)

    case = st.session_state.current_case
    if case is None:
        st.subheader("Resume a draft")
        try:
            cases = client.list_cases().get("items", [])
        except ApiClientError:
            cases = []
            st.info("No submitted cases are available; start the API to resume a draft.")
        if cases:
            selected = st.selectbox("Existing case", cases, format_func=lambda item: short_case_reference(item["id"]))
            if st.button("Resume selected case"):
                st.session_state.current_case = client.get_case(selected["id"])
                st.rerun()
        return

    try:
        case = client.get_case(case["id"])
        st.session_state.current_case = case
    except ApiClientError as exc:
        _error(exc)
        return
    st.divider()
    st.write(f"{short_case_reference(case['id'])} · Status `{case['status']}`")
    if case["status"] == "DRAFT":
        st.warning("Run the assessment after confirming that the inputs are correct.")
        if st.button("Run assessment", type="primary"):
            try:
                payload = {"case_id": case["id"], "face_attachment_id": st.session_state.attachment_ids.get("face"), "audio_attachment_id": st.session_state.attachment_ids.get("audio"), "metadata": metadata.model_dump(by_alias=True, mode="json"), "acute_symptoms": symptoms.model_dump(mode="json")}
                client.create_assessment(payload)
                st.session_state.current_case = client.get_case(case["id"])
                st.success("Assessment completed and stored as an immutable snapshot.")
                st.rerun()
            except ApiClientError as exc:
                _error(exc)
    if case.get("assessment_result"):
        render_assessment_result(st, case["assessment_result"])
    if case["status"] == "ASSESSED":
        confirm_submit = st.checkbox("I confirm these inputs and the assessment snapshot are ready for clinician review.")
        if st.button("Submit case for clinician review"):
            try:
                client.submit_case(case["id"], {"confirmed": confirm_submit}, f'"{case["version"]}"')
                st.success(f"Submitted case {case['id']}.")
                st.session_state.current_case = client.get_case(case["id"])
                st.rerun()
            except ApiClientError as exc:
                _error(exc)


if __name__ == "__main__":
    main()
