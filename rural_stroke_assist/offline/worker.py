"""Single-owner durable synchronization worker."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Protocol

from rural_stroke_assist.client import ApiClientError
from rural_stroke_assist.offline.clock import ExponentialBackoff
from rural_stroke_assist.offline.contracts import (
    OutboxEventType,
    OutboxState,
    SyncState,
    WorkflowState,
)
from rural_stroke_assist.offline.http import FailureAction, classify_http_failure
from rural_stroke_assist.offline.store import LocalAttachment, OutboxEvent, SQLiteOfflineStore
from rural_stroke_assist.offline.token import TokenProvider, TokenRefreshError


class SyncTransport(Protocol):
    def create_case(
        self, payload: dict[str, Any], *, idempotency_key: str, token: str | None
    ) -> dict[str, Any]: ...
    def upload_attachment(
        self, attachment: LocalAttachment, data: bytes, *, idempotency_key: str, token: str | None
    ) -> dict[str, Any]: ...
    def import_assessment(
        self, payload: dict[str, Any], *, idempotency_key: str, token: str | None
    ) -> dict[str, Any]: ...
    def get_case_with_etag(
        self, case_id: str, *, token: str | None
    ) -> tuple[dict[str, Any], str | None]: ...
    def submit_case(
        self,
        case_id: str,
        payload: dict[str, Any],
        etag: str,
        *,
        idempotency_key: str,
        token: str | None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SyncOutcome:
    event_id: str | None
    state: OutboxState | None
    message: str = ""


class SyncWorker:
    def __init__(
        self,
        *,
        store: SQLiteOfflineStore,
        transport: SyncTransport,
        token_provider: TokenProvider,
        worker_id: str,
        backoff: ExponentialBackoff | None = None,
        random_value: callable | None = None,
    ) -> None:
        self.store = store
        self.transport = transport
        self.token_provider = token_provider
        self.worker_id = worker_id
        self.backoff = backoff or ExponentialBackoff()
        self.random_value = random_value or random.random

    def run_once(self, *, now: str) -> SyncOutcome:
        event = self.store.lease_next(worker_id=self.worker_id, now=now, lease_seconds=60)
        if event is None:
            return SyncOutcome(None, None, "idle")
        try:
            self._execute(event, now=now, allow_refresh=True)
            return SyncOutcome(event.event_id, OutboxState.ACKED)
        except ApiClientError as exc:
            outcome = self._handle_failure(event, exc, now=now)
            return outcome
        except TokenRefreshError as exc:
            safe = str(exc)[:500]
            self.store.transition_event(
                event.event_id, OutboxState.BLOCKED_AUTH, now=now, last_error=safe
            )
            self.store.set_sync_state(event.case_id, SyncState.BLOCKED_AUTH)
            return SyncOutcome(event.event_id, OutboxState.BLOCKED_AUTH, safe)

    def _execute(self, event: OutboxEvent, *, now: str, allow_refresh: bool) -> None:
        token = self.token_provider.get_token()
        try:
            if event.event_type is OutboxEventType.CREATE_CASE:
                response = self.transport.create_case(
                    event.payload, idempotency_key=event.event_id, token=token
                )
                self.store.set_remote_case(event.case_id, version=response.get("version"))
            elif event.event_type is OutboxEventType.UPLOAD_ATTACHMENT:
                attachment = self.store.get_attachment(event.payload["attachment_id"])
                path = (self.store.media_root / attachment.relative_path).resolve()
                if self.store.media_root.resolve() not in path.parents or not path.is_file():
                    raise ApiClientError(
                        422,
                        {
                            "code": "validation_error",
                            "message": "Managed attachment is unavailable.",
                        },
                    )
                response = self.transport.upload_attachment(
                    attachment, path.read_bytes(), idempotency_key=event.event_id, token=token
                )
                self.store.set_remote_attachment(attachment.attachment_id, str(response["id"]))
            elif event.event_type is OutboxEventType.IMPORT_ASSESSMENT:
                self.transport.import_assessment(
                    event.payload, idempotency_key=event.event_id, token=token
                )
            elif event.event_type is OutboxEventType.SUBMIT_CASE:
                _, etag = self.transport.get_case_with_etag(event.case_id, token=token)
                if not etag:
                    raise ApiClientError(
                        409,
                        {"code": "conflict", "message": "Central case did not provide an ETag."},
                    )
                response = self.transport.submit_case(
                    event.case_id, event.payload, etag, idempotency_key=event.event_id, token=token
                )
                self.store.set_remote_case(
                    event.case_id,
                    version=response.get("version"),
                    submitted_hash=event.payload.get("assessment_hash"),
                )
                self.store.set_sync_state(
                    event.case_id, SyncState.SYNCED, workflow_state=WorkflowState.SYNCED
                )
            self.store.ack_event(event.event_id, now=now)
        except ApiClientError as exc:
            if exc.status_code == 401 and allow_refresh:
                self.token_provider.refresh()
                self._execute(event, now=now, allow_refresh=False)
            else:
                raise

    def _handle_failure(self, event: OutboxEvent, exc: ApiClientError, *, now: str) -> SyncOutcome:
        classification = classify_http_failure(exc.status_code, error_code=exc.code)
        safe = str(exc)[:500]
        if classification.action is FailureAction.RETRY:
            delay = self.backoff.delay_seconds(
                attempt=max(1, event.attempt_count), random_value=float(self.random_value())
            )
            retry_at = _add_seconds(now, delay)
            retry_after = exc.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                retry_at = _add_seconds(now, min(float(retry_after), self.backoff.maximum_seconds))
            self.store.transition_event(
                event.event_id,
                OutboxState.RETRY_WAIT,
                now=now,
                last_error=safe,
                next_attempt_at=retry_at,
            )
            self.store.set_sync_state(event.case_id, SyncState.RETRY_WAIT)
            return SyncOutcome(event.event_id, OutboxState.RETRY_WAIT, safe)
        if exc.status_code == 401:
            self.store.transition_event(
                event.event_id, OutboxState.BLOCKED_AUTH, now=now, last_error=safe
            )
            self.store.set_sync_state(event.case_id, SyncState.BLOCKED_AUTH)
            return SyncOutcome(event.event_id, OutboxState.BLOCKED_AUTH, safe)
        target = (
            OutboxState.CONFLICT
            if classification.action is FailureAction.CONFLICT
            else OutboxState.DEAD_LETTER
        )
        sync_state = (
            SyncState.CONFLICT if target is OutboxState.CONFLICT else SyncState.DEAD_LETTER
        )
        self.store.transition_event(event.event_id, target, now=now, last_error=safe)
        self.store.set_sync_state(
            event.case_id,
            sync_state,
            workflow_state=WorkflowState.CONFLICT
            if target is OutboxState.CONFLICT
            else WorkflowState.DEAD_LETTER,
        )
        return SyncOutcome(event.event_id, target, safe)


def _add_seconds(value: str, seconds: float) -> str:
    from datetime import datetime, timedelta

    return (
        (datetime.fromisoformat(value.replace("Z", "+00:00")) + timedelta(seconds=seconds))
        .isoformat()
        .replace("+00:00", "Z")
    )
