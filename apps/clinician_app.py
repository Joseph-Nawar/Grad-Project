"""Streamlit clinician review client for the Stage 2 API."""

from __future__ import annotations

import streamlit as st

from rural_stroke_assist.client import ApiClient, ApiClientError
from rural_stroke_assist.ui.common import apply_theme, render_assessment_result
from rural_stroke_assist.ui.presentation import build_context_rows, humanize_value, short_case_reference


st.set_page_config(page_title="RuralStroke-Assist Clinician", page_icon="🩺", layout="wide")
apply_theme(st)


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def _error(exc: ApiClientError) -> None:
    st.error(exc.payload.get("message", "The API request failed."))


def _render_context(case: dict[str, object]) -> None:
    payload = case.get("assessment_input") or {}
    rows, symptom_rows = build_context_rows(payload)
    st.subheader("Collection context")
    st.write(f"**Facility:** {humanize_value(case.get('facility'))}")
    st.write(f"**Pseudonymous patient code:** {humanize_value(case.get('patient_code'))}")
    st.subheader("Background metadata")
    for row in rows:
        st.write(f"**{row.label}:** {row.value}")
    if not rows:
        st.caption("No background metadata recorded.")
    st.subheader("Acute symptoms")
    for row in symptom_rows:
        st.write(f"**{row.label}:** {row.value}")
    if not symptom_rows:
        st.caption("No acute symptoms recorded.")
    with st.expander("Developer data", expanded=False):
        st.json(payload)


def _render_media(client: ApiClient, case: dict[str, object]) -> None:
    st.subheader("Submitted image and audio")
    for attachment in case.get("attachments", []):
        try:
            data = client.read_attachment(str(attachment["id"]))
            if attachment["kind"] == "face":
                st.markdown("**Facial image evidence**")
                st.image(data, caption="Submitted facial image")
            else:
                st.markdown("**Speech recording evidence**")
                st.audio(data, format=attachment["media_type"])
        except Exception:
            st.error("Submitted media is unavailable or could not be loaded.")


def main() -> None:
    client = get_client()
    st.title("RuralStroke-Assist · Clinician review")
    st.caption("Review submitted research evidence and record a triage-urgency opinion")
    try:
        cases = client.list_cases().get("items", [])
    except ApiClientError as exc:
        st.info("No submitted cases are available for review.")
        _error(exc)
        return
    cases = [case for case in cases if case["status"] in {"SUBMITTED", "IN_REVIEW", "REVIEWED_AGREED", "REVIEWED_OVERRIDDEN"}]
    if not cases:
        st.info("No submitted cases are available for review.")
        return
    selected_case = st.selectbox("Case queue", cases, format_func=lambda item: short_case_reference(item["id"]))
    try:
        case = client.get_case(selected_case["id"])
    except ApiClientError as exc:
        _error(exc)
        return
    st.write(f"{short_case_reference(case['id'])} · Status: `{case['status']}`")
    with st.expander("Technical case details", expanded=False):
        st.write("Full case ID", case["id"])
        st.write("Created", case["created_at"])
    _render_context(case)
    _render_media(client, case)
    st.subheader("Stored assessment summary")
    if case.get("assessment_result"):
        render_assessment_result(st, case["assessment_result"], include_limitations=False)
    if case["status"] == "SUBMITTED" and st.button("Begin review", type="primary"):
        try:
            client.claim_review(case["id"])
            st.rerun()
        except ApiClientError as exc:
            _error(exc)
    if case["status"] == "IN_REVIEW":
        with st.form("review_form"):
            decision = st.radio("Decision", ["Agree with proposed triage urgency", "Override proposed triage urgency"])
            notes = st.text_area("Review notes")
            alternative = st.text_input("Alternative disposition or urgency (required for override)")
            save_review = st.form_submit_button("Record clinician review")
        if save_review:
            try:
                client.create_review(case["id"], {"agree": decision.startswith("Agree"), "notes": notes, "alternative_disposition": alternative or None})
                st.success("Review recorded.")
                st.rerun()
            except ApiClientError as exc:
                _error(exc)
    if case["status"] in {"REVIEWED_AGREED", "REVIEWED_OVERRIDDEN"}:
        st.success("A separate clinician review is recorded for this submitted snapshot.")
    st.caption("Clinician review reads the stored assessment snapshot and never reruns inference.")


if __name__ == "__main__":
    main()
