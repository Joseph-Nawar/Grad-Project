"""Database-backed idempotency decisions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from fastapi.encoders import jsonable_encoder

from rural_stroke_assist.server.domain.hashing import canonical_sha256
from rural_stroke_assist.server.errors import ApiError
from rural_stroke_assist.server.infrastructure.db.models import IdempotencyModel
from rural_stroke_assist.server.principal import Principal


def require_idempotency_key(value: str | None) -> str:
    if not value or not value.strip() or len(value) > 255:
        raise ApiError("validation_error", "Idempotency-Key is required for this operation.", status_code=400)
    return value.strip()


def begin_idempotency(session: Session, principal: Principal, operation: str, key: str, payload: Any, *, expiry_seconds: int = 3600) -> IdempotencyModel | dict[str, Any]:
    key = require_idempotency_key(key)
    request_hash = canonical_sha256(payload)
    existing = session.scalar(select(IdempotencyModel).where(IdempotencyModel.actor_subject == principal.subject, IdempotencyModel.operation == operation, IdempotencyModel.key == key).with_for_update())
    now = datetime.now(timezone.utc)
    if existing is not None:
        if existing.request_hash != request_hash:
            raise ApiError("idempotency_conflict", "The idempotency key was reused with a different request.", status_code=409)
        if existing.state == "COMPLETED" and existing.response_snapshot is not None:
            return existing.response_snapshot
        if existing.expires_at > now:
            raise ApiError("idempotency_in_progress", "An identical operation is already in progress.", status_code=409)
        existing.state = "IN_PROGRESS"
        existing.expires_at = now + timedelta(seconds=expiry_seconds)
        existing.updated_at = now
        return existing
    record = IdempotencyModel(actor_subject=principal.subject, operation=operation, key=key, request_hash=request_hash, state="IN_PROGRESS", expires_at=now + timedelta(seconds=expiry_seconds))
    session.add(record)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        existing = session.scalar(select(IdempotencyModel).where(IdempotencyModel.actor_subject == principal.subject, IdempotencyModel.operation == operation, IdempotencyModel.key == key))
        if existing is not None and existing.request_hash == request_hash and existing.state == "COMPLETED" and existing.response_snapshot is not None:
            return existing.response_snapshot
        raise ApiError("idempotency_in_progress", "An identical operation is already in progress.", status_code=409)
    return record


def complete_idempotency(record: IdempotencyModel, response: dict[str, Any], resource_id: UUID | None = None) -> None:
    record.state = "COMPLETED"
    record.response_snapshot = jsonable_encoder(response)
    record.resource_id = resource_id
    record.updated_at = datetime.now(timezone.utc)
