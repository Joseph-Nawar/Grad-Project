from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from rural_stroke_assist.server.api.dependencies import get_principal, get_session
from rural_stroke_assist.server.api.schemas.cases import CaseCreateRequest, CasePage, CaseResponse, CaseUpdateRequest, SubmissionRequest
from rural_stroke_assist.server.api.schemas.attachments import AttachmentResponse
from rural_stroke_assist.server.api.schemas.common import SnapshotResponse
from rural_stroke_assist.server.application.cases import ensure_if_match, get_case, require_state
from rural_stroke_assist.server.application.idempotency import begin_idempotency, complete_idempotency, require_idempotency_key
from rural_stroke_assist.server.domain.cursors import decode_cursor, encode_cursor
from rural_stroke_assist.server.domain.states import CaseState
from rural_stroke_assist.server.errors import ApiError
from rural_stroke_assist.server.infrastructure.db.models import CaseModel, IdempotencyModel, SubmissionModel
from rural_stroke_assist.server.principal import Principal

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])


def _json_case(case: CaseModel) -> dict[str, Any]:
    latest_assessment = case.assessments[-1] if case.assessments else None
    return {"id": case.id, "facility": case.facility, "patient_code": case.patient_code, "status": case.status, "version": case.version, "created_at": case.created_at, "updated_at": case.updated_at, "submitted_at": case.submitted_at, "assessment_id": latest_assessment.id if latest_assessment else None, "assessment_result": latest_assessment.result_snapshot if latest_assessment else None, "assessment_input": case.assessment_input, "attachments": [{"id": item.id, "kind": item.kind, "media_type": item.media_type, "size_bytes": item.size_bytes, "checksum_sha256": item.checksum_sha256} for item in case.attachments]}


def _response(case: CaseModel, response: Response) -> CaseResponse:
    response.headers["ETag"] = f'"{case.version}"'
    return CaseResponse.model_validate(_json_case(case))


@router.post("", operation_id="case_create", response_model=CaseResponse, status_code=201)
def create_case(request: CaseCreateRequest, response: Response, idempotency_key: str = Header(..., alias="Idempotency-Key"), session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> CaseResponse | dict[str, Any]:
    if not principal.has_role("collector") and not principal.has_role("demo-admin"):
        raise ApiError("forbidden", "Collector role is required.", status_code=403)
    if not principal.can_access_facility(request.facility):
        raise ApiError("forbidden", "Facility scope is not authorized.", status_code=403)
    payload = request.model_dump(mode="json")
    record = begin_idempotency(session, principal, "case_create", require_idempotency_key(idempotency_key), payload)
    if isinstance(record, dict):
        return record
    if session.get(CaseModel, request.id) is not None:
        raise ApiError("conflict", "Case ID already exists.", status_code=409)
    case = CaseModel(id=request.id, facility=request.facility, patient_code=request.patient_code, collector_subject=principal.subject, status="DRAFT", assessment_input=request.assessment_input.model_dump(mode="json", by_alias=True))
    session.add(case)
    session.flush()
    output = _json_case(case)
    complete_idempotency(record, output, case.id)
    session.commit()
    return _response(case, response)


@router.get("", operation_id="case_list", response_model=CasePage)
def list_cases(response: Response, limit: int = Query(default=25, ge=1, le=100), cursor: str | None = None, session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> CasePage:
    query = select(CaseModel).options(selectinload(CaseModel.attachments)).order_by(CaseModel.created_at.desc(), CaseModel.id.desc()).limit(limit + 1)
    if principal.facilities and "*" not in principal.facilities:
        query = query.where(CaseModel.facility.in_(principal.facilities))
    if principal.has_role("collector") and not principal.has_role("clinician"):
        query = query.where(CaseModel.collector_subject == principal.subject)
    if cursor:
        created_at, resource_id = decode_cursor(cursor)
        query = query.where((CaseModel.created_at < datetime.fromisoformat(created_at.replace("Z", "+00:00"))) | ((CaseModel.created_at == datetime.fromisoformat(created_at.replace("Z", "+00:00"))) & (CaseModel.id < UUID(resource_id))))
    items = list(session.scalars(query))
    next_cursor = None
    if len(items) > limit:
        last = items.pop()
        next_cursor = encode_cursor(last.created_at.isoformat().replace("+00:00", "Z"), str(last.id))
    return CasePage(items=[CaseResponse.model_validate(_json_case(item)) for item in items], page={"next_cursor": next_cursor})


@router.get("/{case_id}", operation_id="case_get", response_model=CaseResponse)
def get_case_route(case_id: UUID, response: Response, session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> CaseResponse:
    case = get_case(session, case_id, principal)
    return _response(case, response)


@router.patch("/{case_id}", operation_id="case_update", response_model=CaseResponse)
def update_case(case_id: UUID, request: CaseUpdateRequest, response: Response, if_match: str | None = Header(default=None, alias="If-Match"), session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> CaseResponse:
    case = get_case(session, case_id, principal)
    if not principal.has_role("collector") or case.collector_subject != principal.subject:
        raise ApiError("forbidden", "Collector ownership is required.", status_code=403)
    require_state(case, CaseState.DRAFT)
    ensure_if_match(case, if_match)
    if request.patient_code is not None:
        case.patient_code = request.patient_code
    if request.assessment_input is not None:
        case.assessment_input = request.assessment_input.model_dump(mode="json", by_alias=True)
    case.updated_at = datetime.now(timezone.utc)
    session.commit()
    return _response(case, response)


@router.post("/{case_id}/submit", operation_id="case_submit", response_model=CaseResponse)
def submit_case(case_id: UUID, request: SubmissionRequest, response: Response, if_match: str | None = Header(default=None, alias="If-Match"), idempotency_key: str = Header(..., alias="Idempotency-Key"), session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> CaseResponse | dict[str, Any]:
    case = get_case(session, case_id, principal)
    if not principal.has_role("collector") or case.collector_subject != principal.subject:
        raise ApiError("forbidden", "Collector ownership is required.", status_code=403)
    require_state(case, CaseState.ASSESSED)
    ensure_if_match(case, if_match)
    record = begin_idempotency(session, principal, "case_submit", require_idempotency_key(idempotency_key), {"case_id": str(case_id), **request.model_dump()})
    if isinstance(record, dict):
        return record
    assessment = case.assessments[-1] if case.assessments else None
    if assessment is None or not request.confirmed:
        raise ApiError("validation_error", "Assessment and confirmation are required before submission.", status_code=400)
    snapshot = {"case_id": str(case.id), "assessment_id": str(assessment.id), "input": case.assessment_input, "assessment": assessment.result_snapshot, "attachments": [str(item.id) for item in case.attachments]}
    from rural_stroke_assist.server.domain.hashing import canonical_sha256
    submission = SubmissionModel(case_id=case.id, assessment_id=assessment.id, snapshot=snapshot, snapshot_sha256=canonical_sha256(snapshot), submitted_by=principal.subject)
    session.add(submission)
    case.status = CaseState.SUBMITTED.value
    case.submitted_at = datetime.now(timezone.utc)
    case.updated_at = case.submitted_at
    session.flush()
    output = _json_case(case)
    complete_idempotency(record, output, case.id)
    session.commit()
    return _response(case, response)


@router.get("/{case_id}/snapshot", operation_id="case_snapshot", response_model=SnapshotResponse)
def get_snapshot(case_id: UUID, session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> SnapshotResponse:
    case = get_case(session, case_id, principal)
    if case.submission is None:
        raise ApiError("not_found", "Submitted snapshot was not found.", status_code=404)
    return SnapshotResponse(snapshot=case.submission.snapshot, snapshot_sha256=case.submission.snapshot_sha256, submitted_at=case.submitted_at)


@router.get("/{case_id}/attachments", operation_id="case_attachments_list", response_model=list[AttachmentResponse])
def list_case_attachments(case_id: UUID, session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> list[AttachmentResponse]:
    case = get_case(session, case_id, principal)
    return [AttachmentResponse.model_validate(item) for item in case.attachments]
