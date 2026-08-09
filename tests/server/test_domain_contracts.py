from __future__ import annotations

import pytest

from rural_stroke_assist.server.domain.authorization import require_case_access
from rural_stroke_assist.server.domain.cursors import decode_cursor, encode_cursor
from rural_stroke_assist.server.domain.hashing import canonical_sha256
from rural_stroke_assist.server.domain.identity import Principal
from rural_stroke_assist.server.domain.states import CaseState, can_transition
from rural_stroke_assist.server.api.schemas.assessments import AssessmentCreateRequest


def test_canonical_hash_is_order_independent() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_cursor_round_trip_preserves_created_at_and_id() -> None:
    value = encode_cursor("2026-08-07T00:00:00Z", "case-1")
    assert decode_cursor(value) == ("2026-08-07T00:00:00Z", "case-1")


def test_lifecycle_allows_only_declared_transitions() -> None:
    assert can_transition(CaseState.DRAFT, CaseState.ASSESSED)
    assert can_transition(CaseState.ASSESSED, CaseState.SUBMITTED)
    assert not can_transition(CaseState.SUBMITTED, CaseState.DRAFT)


def test_case_access_requires_role_and_facility() -> None:
    principal = Principal("collector-1", {"collector"}, {"facility-a"})
    assert require_case_access(principal, "collector", "facility-a") is None
    with pytest.raises(PermissionError):
        require_case_access(principal, "clinician", "facility-a")
    with pytest.raises(PermissionError):
        require_case_access(principal, "collector", "facility-b")


def test_assessment_request_contains_ids_and_structured_input_without_paths() -> None:
    request = AssessmentCreateRequest.model_validate(
            {
                "id": "00000000-0000-0000-0000-000000000002",
                "case_id": "00000000-0000-0000-0000-000000000001",
            "face_attachment_id": None,
            "audio_attachment_id": None,
            "metadata": None,
            "acute_symptoms": None,
        }
    )
    assert request.case_id.int == 1
    assert "path" not in request.model_dump_json().lower()
