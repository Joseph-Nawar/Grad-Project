"""Streamlit clinician review application for submitted local cases."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import streamlit as st

from rural_stroke_assist.cases.contracts import CaseStatus
from rural_stroke_assist.cases.factory import create_review_workflow_service
from rural_stroke_assist.cases.report_builder import render_html, render_json, render_markdown
from rural_stroke_assist.ui.common import apply_theme, render_assessment_result, render_case_context, render_case_header, render_limitations, render_submitted_media
from rural_stroke_assist.ui.presentation import short_case_reference


st.set_page_config(page_title="RuralStroke-Assist Clinician", page_icon="🩺", layout="wide")
apply_theme(st)


@st.cache_resource
def get_workflow():
    return create_review_workflow_service(os.environ.get("RURAL_STROKE_RUNTIME_DIR", "runtime_data"))


def main() -> None:
    workflow = get_workflow()
    st.title("RuralStroke-Assist · Clinician review")
    st.caption("Review submitted research evidence and record a triage-urgency opinion")
    statuses = (CaseStatus.SUBMITTED, CaseStatus.IN_REVIEW, CaseStatus.REVIEWED_AGREED, CaseStatus.REVIEWED_OVERRIDDEN)
    cases = workflow.list_cases(statuses=statuses)
    status_filter = st.selectbox("Filter status", ["ALL", *[item.value for item in statuses]])
    if status_filter != "ALL":
        cases = [case for case in cases if case.status.value == status_filter]
    facilities = ["ALL", *sorted({case.facility for case in cases})]
    facility_filter = st.selectbox("Filter facility", facilities)
    if facility_filter != "ALL":
        cases = [case for case in cases if case.facility == facility_filter]
    bands = ["ALL", "LOW", "MODERATE", "HIGH", "URGENT"]
    band_filter = st.selectbox("Filter evidence band", bands)
    if band_filter != "ALL":
        cases = [case for case in cases if ((case.assessment_result or {}).get("fusion") or {}).get("risk_band") == band_filter]
    time_filter = st.selectbox("Filter time", ["ALL", "LAST_24_HOURS", "LAST_7_DAYS"])
    if time_filter != "ALL":
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24 if time_filter == "LAST_24_HOURS" else 24 * 7)
        cases = [case for case in cases if datetime.fromisoformat(case.created_at.replace("Z", "+00:00")) >= cutoff]
    if not cases:
        st.info("No submitted cases are available for review.")
        return
    selected_case = st.selectbox("Case queue", cases, format_func=lambda item: short_case_reference(item.case_id))
    case = workflow.get(selected_case.case_id)
    render_case_header(st, case)
    render_case_context(st, case)
    render_submitted_media(st, workflow, case)
    st.subheader("Stored assessment summary")
    if case.assessment_result:
        render_assessment_result(st, case.assessment_result, include_limitations=False)
    st.download_button("Download JSON report", render_json(case), file_name=f"case-{case.case_id}.json", mime="application/json")
    st.download_button("Download Markdown report", render_markdown(case), file_name=f"case-{case.case_id}.md", mime="text/markdown")
    st.download_button("Download HTML report", render_html(case), file_name=f"case-{case.case_id}.html", mime="text/html")
    if case.status is CaseStatus.SUBMITTED and st.button("Begin review", type="primary"):
        workflow.begin_review(case.case_id, actor="clinician")
        st.rerun()
    if case.status is CaseStatus.IN_REVIEW:
        with st.form("review_form"):
            clinician = st.text_input("Clinician name")
            centre = st.text_input("Medical centre")
            decision = st.radio("Decision", ["Agree with proposed triage urgency", "Override proposed triage urgency"])
            notes = st.text_area("Review notes")
            alternative = st.text_input("Alternative disposition or urgency (required for override)")
            save_review = st.form_submit_button("Record clinician review")
        if save_review:
            agree = decision.startswith("Agree")
            try:
                reviewed = workflow.review(case.case_id, clinician_name=clinician, medical_centre=centre, agree=agree, notes=notes, alternative_disposition=alternative or None)
                st.success(f"Review recorded: {reviewed.status.value}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if case.clinician_review:
        st.success(f"Recorded outcome: {case.clinician_review.decision}")
        st.write(case.clinician_review.notes)
    if case.assessment_result:
        render_limitations(st, case.assessment_result)


if __name__ == "__main__":
    main()
