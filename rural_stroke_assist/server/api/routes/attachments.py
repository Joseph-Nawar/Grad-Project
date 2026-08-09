from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from rural_stroke_assist.server.api.dependencies import get_principal, get_session
from rural_stroke_assist.server.api.schemas.attachments import AttachmentResponse
from rural_stroke_assist.server.application.cases import get_case
from rural_stroke_assist.server.errors import ApiError
from rural_stroke_assist.server.infrastructure.db.models import AttachmentModel
from rural_stroke_assist.server.principal import Principal

router = APIRouter(prefix="/api/v1/attachments", tags=["attachments"])


@router.post("", operation_id="attachment_upload", response_model=AttachmentResponse, status_code=201)
def upload_attachment(request: Request, case_id: UUID = Form(...), kind: str = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> AttachmentResponse:
    case = get_case(session, case_id, principal)
    if not principal.has_role("collector") or case.collector_subject != principal.subject:
        raise ApiError("forbidden", "Collector ownership is required.", status_code=403)
    store = getattr(request.app.state, "attachment_store", None)
    if store is None:
        raise ApiError("dependency_unavailable", "Attachment storage is unavailable.", status_code=503)
    try:
        stored = store.save(case_id, kind, file.file, filename=file.filename, media_type=file.content_type or "")
    except ValueError as exc:
        message = str(exc)
        if "exceeds" in message.lower():
            raise ApiError("attachment_too_large", "Attachment exceeds the configured size limit.", status_code=413) from exc
        if "media type" in message.lower() or "unsupported" in message.lower():
            raise ApiError("unsupported_media_type", "The attachment media type is not supported.", status_code=415) from exc
        raise ApiError("validation_error", "The attachment is invalid.", status_code=400) from exc
    try:
        row = AttachmentModel(id=stored.id, case_id=case_id, kind=kind, media_type=stored.media_type, storage_key=stored.storage_key, original_filename=None, size_bytes=stored.size_bytes, checksum_sha256=stored.checksum_sha256)
        session.add(row)
        session.commit()
    except Exception:
        store.delete(stored)
        session.rollback()
        raise ApiError("dependency_unavailable", "Attachment could not be persisted.", status_code=503)
    return AttachmentResponse.model_validate(row)


@router.get(
    "/{attachment_id}",
    operation_id="attachment_read",
    response_class=FileResponse,
    responses={
        200: {"description": "Managed attachment bytes.", "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}, "image/png": {"schema": {"type": "string", "format": "binary"}}, "audio/wav": {"schema": {"type": "string", "format": "binary"}}, "audio/mpeg": {"schema": {"type": "string", "format": "binary"}}, "audio/ogg": {"schema": {"type": "string", "format": "binary"}}}},
    },
)
def read_attachment(attachment_id: UUID, request: Request, session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> FileResponse:
    row = session.get(AttachmentModel, attachment_id)
    if row is None:
        raise ApiError("not_found", "Attachment was not found.", status_code=404)
    get_case(session, row.case_id, principal)
    store = getattr(request.app.state, "attachment_store", None)
    if store is None:
        raise ApiError("dependency_unavailable", "Attachment storage is unavailable.", status_code=503)
    from rural_stroke_assist.server.infrastructure.storage.protocol import StoredAttachment
    path = store.resolve(StoredAttachment(row.id, row.case_id, row.kind, row.media_type, row.storage_key, row.size_bytes, row.checksum_sha256))
    return FileResponse(path, media_type=row.media_type)


@router.delete("/{attachment_id}", operation_id="attachment_delete", status_code=204)
def delete_attachment(attachment_id: UUID, request: Request, session: Session = Depends(get_session), principal: Principal = Depends(get_principal)) -> None:
    row = session.get(AttachmentModel, attachment_id)
    if row is None:
        raise ApiError("not_found", "Attachment was not found.", status_code=404)
    case = get_case(session, row.case_id, principal)
    if not principal.has_role("collector") or case.collector_subject != principal.subject:
        raise ApiError("forbidden", "Collector ownership is required.", status_code=403)
    store = getattr(request.app.state, "attachment_store", None)
    from rural_stroke_assist.server.infrastructure.storage.protocol import StoredAttachment
    store.delete(StoredAttachment(row.id, row.case_id, row.kind, row.media_type, row.storage_key, row.size_bytes, row.checksum_sha256))
    session.delete(row)
    session.commit()
