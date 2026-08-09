"""Central case lifecycle rules."""

from __future__ import annotations

from enum import StrEnum


class CaseState(StrEnum):
    DRAFT = "DRAFT"
    ASSESSED = "ASSESSED"
    SUBMITTED = "SUBMITTED"
    IN_REVIEW = "IN_REVIEW"
    REVIEWED_AGREED = "REVIEWED_AGREED"
    REVIEWED_OVERRIDDEN = "REVIEWED_OVERRIDDEN"


TRANSITIONS = {
    (CaseState.DRAFT, CaseState.ASSESSED),
    (CaseState.ASSESSED, CaseState.SUBMITTED),
    (CaseState.SUBMITTED, CaseState.IN_REVIEW),
    (CaseState.IN_REVIEW, CaseState.REVIEWED_AGREED),
    (CaseState.IN_REVIEW, CaseState.REVIEWED_OVERRIDDEN),
}


def can_transition(current: CaseState, target: CaseState) -> bool:
    return (current, target) in TRANSITIONS
