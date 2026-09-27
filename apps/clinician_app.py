"""Streamlit clinician review client for submitted assessment snapshots."""

from __future__ import annotations

import streamlit as st

from rural_stroke_assist.client import ApiClient, ApiClientError
from rural_stroke_assist.ui.common import apply_theme, render_assessment_result, render_product_header
from rural_stroke_assist.ui.oidc import streamlit_api_client
from rural_stroke_assist.ui.presentation import build_context_rows, humanize_value, short_case_reference


def review_notes_error(notes: str) -> str | None:
    if not notes.strip():
        return "Add a brief review rationale before recording this decision."
    return None


def preferred_case_id(case_ids: list[str], current_case_id: str | None) -> str | None:
    """Keep the clinician's current queue selection across status-changing reruns."""
    if current_case_id in case_ids:
        return current_case_id
    return case_ids[0] if case_ids else None


st.set_page_config(page_title="RuralStroke-Triage Clinician", page_icon="🩺", layout="wide")
apply_theme(st)


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def _error(st: object, exc: ApiClientError, action: str) -> None:
    if exc.status_code == 422:
        st.error(f"{action} could not be validated. Check the review fields and try again.")
    elif exc.status_code >= 500:
        st.error(f"{action} is temporarily unavailable. The submitted assessment remains saved.")
    else:
        st.error(exc.payload.get("message", f"{action} could not be completed."))


def _render_context(case: dict[str, object]) -> None:
    payload = case.get("assessment_input") or {}
    context_rows, symptom_rows = build_context_rows(payload)
    positive = [row.label for row in symptom_rows if row.value == "Yes"]
    st.markdown("#### Acute symptoms")
    if positive:
        st.warning("Reported symptom evidence: " + ", ".join(positive))
    else:
        st.info("No acute symptoms were reported in the submitted inputs.")
    onset = next((row.value for row in symptom_rows if row.label == "Symptom onset minutes"), "Unknown")
    st.caption(f"Onset time: {onset} minutes" if onset != "Unknown" else "Onset time: Unknown")
    with st.expander("Case context and background risk", expanded=False):
        st.write(f"**Facility:** {humanize_value(case.get('facility'))}")
        st.write(f"**Pseudonymous case code:** {humanize_value(case.get('patient_code'))}")
        if context_rows:
            columns = st.columns(3)
            for index, row in enumerate(context_rows):
                with columns[index % len(columns)]:
                    st.markdown(f"**{row.label}**  \n{row.value}")
        else:
            st.caption("No background metadata was recorded.")


def _render_media(client: ApiClient, case: dict[str, object]) -> None:
    attachments = case.get("attachments", [])
    if not attachments:
        st.caption("No media attachments were submitted.")
        return
    with st.expander("Submitted media", expanded=False):
        face_col, audio_col = st.columns(2)
        for attachment in attachments:
            try:
                data = client.read_attachment(str(attachment["id"]))
                if attachment["kind"] == "face":
                    with face_col:
                        st.image(data, caption="Submitted visual input", use_container_width=True)
                else:
                    with audio_col:
                        st.audio(data, format=attachment["media_type"])
                        st.caption("Submitted speech input")
            except Exception:
                st.warning("A submitted media file could not be loaded. The assessment snapshot remains available.")


def _render_decision_panel(client: ApiClient, case: dict[str, object]) -> None:
    status = str(case.get("status", ""))
    st.markdown("### Clinical review")
    if status == "SUBMITTED":
        st.caption("Claim the submitted snapshot to record an independent clinician review.")
        if st.button("Begin clinical review", type="primary", use_container_width=True):
            try:
                client.claim_review(str(case["id"]))
                st.rerun()
            except ApiClientError as exc:
                _error(st, exc, "Review claim")
        return

    if status == "IN_REVIEW":
        with st.form("review_form"):
            decision = st.radio(
                "Proposed urgency",
                ["Agree with proposed urgency", "Override proposed urgency"],
            )
            notes = st.text_area("Review notes · required", height=115)
            alternative = st.text_input("Alternative disposition · required for override")
            save_review = st.form_submit_button("Record clinician review", type="primary")
        if save_review:
            message = review_notes_error(notes)
            if message:
                st.error(message)
            elif decision.startswith("Override") and not alternative.strip():
                st.error("Add the alternative disposition or urgency before recording an override.")
            else:
                try:
                    client.create_review(
                        str(case["id"]),
                        {
                            "agree": decision.startswith("Agree"),
                            "notes": notes.strip(),
                            "alternative_disposition": alternative.strip() or None,
                        },
                    )
                    st.rerun()
                except ApiClientError as exc:
                    _error(st, exc, "Clinician review")
        return

    if status in {"REVIEWED_AGREED", "REVIEWED_OVERRIDDEN"}:
        st.success("Clinician review recorded and persisted.")
        st.markdown("**Persisted state**")
        st.write("Agreed with proposed urgency" if status == "REVIEWED_AGREED" else "Overrode proposed urgency")
        try:
            reviews = client.list_reviews(str(case["id"]))
            if reviews:
                latest = reviews[-1]
                if latest.get("notes"):
                    st.caption(f"Review rationale: {latest['notes']}")
                if latest.get("alternative_disposition"):
                    st.caption(f"Alternative disposition: {latest['alternative_disposition']}")
        except ApiClientError:
            st.caption("Review state is saved; rationale could not be reloaded at this time.")


def main() -> None:
    client = streamlit_api_client(st)
    if client is None:
        render_product_header(st, role="clinician", active_step=3)
        st.info("Sign in with the assigned demonstration account to review submitted cases.")
        return
    try:
        cases = client.list_cases().get("items", [])
    except ApiClientError as exc:
        render_product_header(st, role="clinician", active_step=3)
        _error(st, exc, "Clinician case queue")
        return
    cases = [case for case in cases if case["status"] in {"SUBMITTED", "IN_REVIEW", "REVIEWED_AGREED", "REVIEWED_OVERRIDDEN"}]
    render_product_header(st, role="clinician", active_step=3)
    if not cases:
        st.info("No submitted cases are available for review.")
        return

    cases_by_id = {str(case["id"]): case for case in cases}
    case_ids = list(cases_by_id)
    selected_case_id = preferred_case_id(
        case_ids,
        st.session_state.get("selected_clinician_case_id"),
    )
    selected_index = case_ids.index(selected_case_id) if selected_case_id in case_ids else 0
    queue_col, result_col, decision_col = st.columns([2.5, 6.5, 3.2], gap="medium")
    with queue_col:
        st.markdown("### Case queue")
        selected_case_id = st.selectbox(
            "Submitted cases",
            case_ids,
            index=selected_index,
            key="selected_clinician_case_id",
            format_func=lambda case_id: (
                f"{short_case_reference(case_id)} {chr(183)} "
                f"{cases_by_id[case_id]['status'].replace('_', ' ').title()}"
            ),
            label_visibility="collapsed",
        )
        selected_case = cases_by_id[selected_case_id]
        try:
            case = client.get_case(str(selected_case["id"]))
        except ApiClientError as exc:
            _error(st, exc, "Case retrieval")
            return
        st.markdown(f"**{short_case_reference(str(case['id']))}**")
        st.caption(f"Status · {str(case['status']).replace('_', ' ').title()}")
        _render_context(case)
        _render_media(client, case)

    with result_col:
        st.markdown("### Stored assessment")
        if case.get("assessment_result"):
            render_assessment_result(st, case["assessment_result"], include_limitations=False, compact=True)
        else:
            st.warning("No completed assessment snapshot is attached to this case.")
        st.caption("Clinician review reads the submitted immutable snapshot and never reruns inference.")

    with decision_col:
        _render_decision_panel(client, case)


if __name__ == "__main__":
    main()
