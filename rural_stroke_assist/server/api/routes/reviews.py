from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from rural_stroke_assist.server.api.dependencies import get_principal, get_session
from rural_stroke_assist.server.api.schemas.reviews import ReviewClaimResponse, ReviewCreateRequest, ReviewResponse
from rural_stroke_assist.server.application.cases import get_case, require_state
from rural_stroke_assist.server.application.idempotency import begin_idempotency, complete_idempotency, require_idempotency_key
from rural_stroke_assist.server.application.reviews import validate_review_decision
from rural_stroke_assist.server.domain.states import CaseState
from rural_stroke_assist.server.errors import ApiError
from rural_stroke_assist.server.infrastructure.db.models import ReviewModel
from rural_stroke_assist.server.principal import Principal

router = APIRouter(prefix="/api/v1/cases", tags=["reviews"])


def _review_json(review: ReviewModel) -> dict[str, Any]:
    return {"id": review.id, "case_id": review.case_id, "reviewer_subject": review.reviewer_subject, "agree": review.agree, "notes": review.notes, "alternative_disposition": review.alternative_disposition, "created_at": review.created_at.isoformat()}


@router.post("/{case_id}/review-claim", operation_id="review_claim", response_model=ReviewClaimResponse)
def claim_review(case_id: UUID, idempotency_key: str = Header(..., alias="Idempotency-Key"), session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> ReviewClaimResponse | dict[str, Any]:
    if not principal.has_role("clinician"):
        raise ApiError("forbidden", "Clinician role is required.", status_code=403)
    case = get_case(session, case_id, principal)
    require_state(case, CaseState.SUBMITTED)
    record = begin_idempotency(session, principal, "review_claim", require_idempotency_key(idempotency_key), {"case_id": str(case_id)})
    if isinstance(record, dict):
        return record
    case.status = CaseState.IN_REVIEW.value
    case.review_claimed_by = principal.subject
    case.review_claimed_at = datetime.now(timezone.utc)
    session.flush()
    output = {"case_id": case.id, "status": case.status, "claimed_by": principal.subject}
    complete_idempotency(record, output, case.id)
    session.commit()
    return ReviewClaimResponse.model_validate(output)


@router.post("/{case_id}/reviews", operation_id="review_create", response_model=ReviewResponse, status_code=201)
def create_review(case_id: UUID, request: ReviewCreateRequest, idempotency_key: str = Header(..., alias="Idempotency-Key"), session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> ReviewResponse | dict[str, Any]:
    if not principal.has_role("clinician"):
        raise ApiError("forbidden", "Clinician role is required.", status_code=403)
    case = get_case(session, case_id, principal)
    require_state(case, CaseState.IN_REVIEW)
    if case.review_claimed_by != principal.subject:
        raise ApiError("forbidden", "The review must be claimed by this clinician.", status_code=403)
    validate_review_decision(request.agree, request.alternative_disposition)
    record = begin_idempotency(session, principal, "review_create", require_idempotency_key(idempotency_key), {"case_id": str(case_id), **request.model_dump()})
    if isinstance(record, dict):
        return record
    if case.submission is None:
        raise ApiError("conflict", "A submitted snapshot is required.", status_code=409)
    review = ReviewModel(case_id=case.id, submission_id=case.submission.id, reviewer_subject=principal.subject, agree=request.agree, notes=request.notes, alternative_disposition=request.alternative_disposition)
    session.add(review)
    case.status = CaseState.REVIEWED_AGREED.value if request.agree else CaseState.REVIEWED_OVERRIDDEN.value
    session.flush()
    output = _review_json(review)
    complete_idempotency(record, output, review.id)
    session.commit()
    return ReviewResponse.model_validate(output)


@router.get("/{case_id}/reviews", operation_id="review_list", response_model=list[ReviewResponse])
def list_reviews(case_id: UUID, session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> list[ReviewResponse]:
    case = get_case(session, case_id, principal)
    return [ReviewResponse.model_validate(_review_json(review)) for review in case.reviews]
