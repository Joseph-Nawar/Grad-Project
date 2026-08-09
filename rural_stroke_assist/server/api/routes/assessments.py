from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.orm import Session

from rural_stroke_assist.assessment.exceptions import AssessmentError
from rural_stroke_assist.assessment.service import AssessmentService
from rural_stroke_assist.cases.serialization import assessment_result_to_dict
from rural_stroke_assist.server.api.dependencies import get_principal, get_session
from rural_stroke_assist.server.api.schemas.assessments import AssessmentCreateRequest, AssessmentResponse
from rural_stroke_assist.server.application.cases import get_case, require_state
from rural_stroke_assist.server.application.idempotency import begin_idempotency, complete_idempotency, require_idempotency_key
from rural_stroke_assist.server.application.assessments import build_service_input
from rural_stroke_assist.server.domain.states import CaseState
from rural_stroke_assist.server.errors import ApiError
from rural_stroke_assist.server.infrastructure.db.models import AssessmentModel, AttachmentModel
from rural_stroke_assist.server.principal import Principal

router = APIRouter(prefix="/api/v1/assessments", tags=["assessments"])


def _json_assessment(row: AssessmentModel) -> dict[str, Any]:
    return {"id": row.id, "case_id": row.case_id, "status": row.status, "request_snapshot": row.request_snapshot, "result_snapshot": row.result_snapshot, "created_at": row.created_at.isoformat()}


@router.post("", operation_id="assessment_create", response_model=AssessmentResponse, status_code=201)
def create_assessment(request: AssessmentCreateRequest, app_request: Request, idempotency_key: str = Header(..., alias="Idempotency-Key"), session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> AssessmentResponse | dict[str, Any]:
    case = get_case(session, request.case_id, principal)
    if not principal.has_role("collector") or case.collector_subject != principal.subject:
        raise ApiError("forbidden", "Collector ownership is required.", status_code=403)
    require_state(case, CaseState.DRAFT)
    payload = request.model_dump(mode="json", by_alias=True)
    record = begin_idempotency(session, principal, "assessment_create", require_idempotency_key(idempotency_key), payload)
    if isinstance(record, dict):
        return record
    attachments = {item.id: item for item in case.attachments}
    face_path = None
    audio_path = None
    store = getattr(app_request.app.state, "attachment_store", None)
    for attachment_id, target in ((request.face_attachment_id, "face"), (request.audio_attachment_id, "audio")):
        if attachment_id is None:
            continue
        attachment = attachments.get(attachment_id)
        if attachment is None or attachment.kind != target:
            raise ApiError("not_found", "Attachment was not found.", status_code=404)
        from rural_stroke_assist.server.infrastructure.storage.protocol import StoredAttachment
        path = store.resolve(StoredAttachment(attachment.id, attachment.case_id, attachment.kind, attachment.media_type, attachment.storage_key, attachment.size_bytes, attachment.checksum_sha256))
        if target == "face":
            face_path = path
        else:
            audio_path = path
    try:
        service_input = build_service_input(
            session_id=request.session_id,
            face_path=face_path,
            audio_path=audio_path,
            metadata=request.metadata.model_dump(by_alias=True) if request.metadata is not None else None,
            acute_symptoms=request.acute_symptoms.model_dump() if request.acute_symptoms is not None else None,
        )
        service: AssessmentService = app_request.app.state.assessment_service
        result = service.assess(service_input)
    except AssessmentError as exc:
        raise ApiError("assessment_unavailable", "The assessment operation could not be performed.", status_code=503) from exc
    except ValueError as exc:
        raise ApiError("validation_error", "Assessment input is invalid.", status_code=422) from exc
    result_snapshot = assessment_result_to_dict(result)
    request_snapshot = request.model_dump(mode="json", by_alias=True)
    row = AssessmentModel(id=request.id, case_id=case.id, client_id=request.id, actor_subject=principal.subject, status=result.status.value, request_snapshot=request_snapshot, result_snapshot=result_snapshot)
    session.add(row)
    case.status = CaseState.ASSESSED.value
    session.flush()
    output = _json_assessment(row)
    complete_idempotency(record, output, row.id)
    session.commit()
    return AssessmentResponse.model_validate(output)


@router.get("/{assessment_id}", operation_id="assessment_get", response_model=AssessmentResponse)
def get_assessment(assessment_id: UUID, session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> AssessmentResponse:
    row = session.get(AssessmentModel, assessment_id)
    if row is None:
        raise ApiError("not_found", "Assessment was not found.", status_code=404)
    get_case(session, row.case_id, principal)
    return AssessmentResponse.model_validate(_json_assessment(row))


@router.get("/{assessment_id}/result", operation_id="assessment_result_get", response_model=AssessmentResponse)
def get_assessment_result(assessment_id: UUID, session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> AssessmentResponse:
    return get_assessment(assessment_id, session, principal)
