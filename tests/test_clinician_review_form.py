from __future__ import annotations

from apps.clinician_app import preferred_case_id, review_notes_error


def test_clinician_review_form_requires_notes_before_api_submission() -> None:
    assert review_notes_error("") == "Add a brief review rationale before recording this decision."
    assert review_notes_error("   ") == "Add a brief review rationale before recording this decision."
    assert review_notes_error("Reviewed the submitted evidence.") is None


def test_clinician_selection_survives_queue_order_and_status_changes() -> None:
    assert preferred_case_id(["first", "reviewed-case", "third"], "reviewed-case") == "reviewed-case"
    assert preferred_case_id(["first", "third"], "removed-case") == "first"
    assert preferred_case_id([], None) is None
