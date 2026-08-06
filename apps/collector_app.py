"""Streamlit collector application for local RuralStroke-Assist workflow."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rural_stroke_assist.assessment.exceptions import AssessmentError
from rural_stroke_assist.cases.contracts import CaseStatus
from rural_stroke_assist.cases.factory import create_default_workflow_service
from rural_stroke_assist.cases.report_builder import render_html, render_json, render_markdown
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.ui.common import apply_theme, render_assessment_result, render_preassessment_quality
from rural_stroke_assist.ui.presentation import short_case_reference


st.set_page_config(page_title="RuralStroke-Assist Collector", page_icon="🩺", layout="wide")
apply_theme(st)


@st.cache_resource
def get_workflow():
    return create_default_workflow_service(os.environ.get("RURAL_STROKE_RUNTIME_DIR", "runtime_data"))


def _metadata_form() -> MetadataInput:
    return MetadataInput(
        age=st.number_input("Age", min_value=0, max_value=120, value=60),
        hypertension=int(st.checkbox("Hypertension")),
        heart_disease=int(st.checkbox("Heart disease")),
        avg_glucose_level=st.number_input("Average glucose", min_value=0.0, value=100.0),
        bmi=st.number_input("BMI", min_value=0.0, value=25.0),
        gender=st.selectbox("Gender", ["Female", "Male", "Other"]),
        ever_married=st.selectbox("Ever married", ["No", "Yes"]),
        work_type=st.selectbox("Work type", ["Govt_job", "Never_worked", "Private", "Self-employed", "children"]),
        Residence_type=st.selectbox("Residence type", ["Rural", "Urban"]),
        smoking_status=st.selectbox("Smoking status", ["Unknown", "formerly smoked", "never smoked", "smokes"]),
    )


def _build_input(metadata: MetadataInput, face_path: Path | None, audio_path: Path | None, symptoms: AcuteStrokeSymptoms) -> AssessmentInput:
    return AssessmentInput(session_id="collector-session", metadata=metadata, face_image_path=face_path, speech_audio_path=audio_path, acute_symptoms=symptoms)


def main() -> None:
    workflow = get_workflow()
    st.title("RuralStroke-Assist · Collector")
    st.caption("Guided collection for research screening and triage support")
    st.info("Collect only the minimum information needed for this local demonstration. Do not enter names, national IDs, or addresses.")
    if "current_case_id" not in st.session_state:
        st.session_state.current_case_id = None

    with st.form("collection_form"):
        collector = st.text_input("Collector identity", value="Rural health volunteer")
        facility = st.text_input("Facility", value="Local health post")
        patient_code = st.text_input("Optional pseudonymous patient code")
        st.subheader("Contextual metadata")
        metadata = _metadata_form()
        st.subheader("FAST / BE-FAST symptoms")
        onset_known = st.checkbox("Onset time is known")
        onset_minutes = st.number_input("Minutes since symptom onset", min_value=0, value=0) if onset_known else None
        symptoms = AcuteStrokeSymptoms(
            face_drooping=st.checkbox("Face drooping or asymmetry"),
            arm_weakness=st.checkbox("Arm weakness or drift"),
            speech_difficulty=st.checkbox("Speech difficulty"),
            balance_or_coordination_loss=st.checkbox("Balance or coordination loss"),
            vision_disturbance=st.checkbox("Vision disturbance"),
            sudden_severe_headache=st.checkbox("Sudden severe headache"),
            confusion_or_understanding_difficulty=st.checkbox("Confusion or difficulty understanding"),
            symptom_onset_minutes=onset_minutes,
            symptoms_resolved=st.checkbox("Symptoms have resolved"),
        )
        st.subheader("Face and speech")
        st.caption("Sample sentence: ‘The quick brown fox jumps over the lazy dog.’")
        st.info("The speech branch was developed using English TORGO recordings. Use the provided English sentence for this demonstration; Arabic and other languages have not been validated for this branch. The sentence itself is not clinically validated.")
        face_upload = st.camera_input("Capture face image")
        if face_upload is None:
            face_upload = st.file_uploader("Or upload a face image", type=["jpg", "jpeg", "png"])
        audio_upload = st.file_uploader("Upload a speech recording", type=["wav", "mp3", "ogg"])
        render_preassessment_quality(st, face_upload.getvalue() if face_upload else None, audio_upload.getvalue() if audio_upload else None)
        confirm_inputs = st.checkbox("I reviewed the collected inputs")
        start = st.form_submit_button("Start or save draft")

    if start:
        face_bytes = face_upload.getvalue() if face_upload else None
        audio_bytes = audio_upload.getvalue() if audio_upload else None
        assessment_input = _build_input(metadata, None, None, symptoms)
        case = workflow.create_draft(collector_identity=collector, facility=facility, patient_code=patient_code or None, assessment_input=assessment_input, face_bytes=face_bytes, audio_bytes=audio_bytes, face_filename=getattr(face_upload, "name", None), audio_filename=getattr(audio_upload, "name", None), face_media_type=getattr(face_upload, "type", None), audio_media_type=getattr(audio_upload, "type", None))
        st.session_state.current_case_id = case.case_id
        st.success(f"Draft saved: {short_case_reference(case.case_id)}")
        if not confirm_inputs:
            st.warning("Input review is not confirmed; review is required before submission.")

    case_id = st.session_state.current_case_id
    if not case_id:
        st.subheader("Resume a draft")
        drafts = workflow.list_cases(statuses=(CaseStatus.DRAFT, CaseStatus.ASSESSED, CaseStatus.SUBMITTED, CaseStatus.REVIEWED_AGREED, CaseStatus.REVIEWED_OVERRIDDEN))
        if drafts:
            selected = st.selectbox("Existing case", [item.case_id for item in drafts])
            if st.button("Resume selected case"):
                st.session_state.current_case_id = selected
                st.rerun()
        return

    case = workflow.get(case_id)
    st.divider()
    st.write(f"{short_case_reference(case.case_id)} · Status `{case.status.value}`")
    if case.status is CaseStatus.DRAFT:
        st.warning("Run the assessment after confirming that the inputs are correct.")
        if st.button("Run assessment", type="primary"):
            try:
                case = workflow.assess(case.case_id, actor=case.collector_identity)
                st.success("Assessment completed and stored as an immutable snapshot.")
            except (AssessmentError, ValueError) as exc:
                st.error(str(exc))
    if case.assessment_result:
        render_assessment_result(st, case.assessment_result)
    if case.status is CaseStatus.ASSESSED:
        confirm_submit = st.checkbox("I confirm these inputs and the assessment snapshot are ready for clinician review.")
        if st.button("Submit case for clinician review"):
            try:
                case = workflow.submit(case.case_id, actor=case.collector_identity, confirmed=confirm_submit)
                st.success(f"Submitted case {case.case_id} at {case.submitted_at}.")
            except Exception as exc:
                st.error(str(exc))
    if case.status in {CaseStatus.SUBMITTED, CaseStatus.IN_REVIEW, CaseStatus.REVIEWED_AGREED, CaseStatus.REVIEWED_OVERRIDDEN}:
        st.download_button("Download JSON report", render_json(case), file_name=f"case-{case.case_id}.json", mime="application/json")
        st.download_button("Download Markdown report", render_markdown(case), file_name=f"case-{case.case_id}.md", mime="text/markdown")
        st.download_button("Download HTML report", render_html(case), file_name=f"case-{case.case_id}.html", mime="text/html")
    if case.clinician_review:
        st.success(f"Clinician review outcome: {case.clinician_review.decision}")
        st.write(case.clinician_review.notes)


if __name__ == "__main__":
    main()
