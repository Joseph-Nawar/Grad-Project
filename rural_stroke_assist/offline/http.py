"""Transport error classification shared by sync worker tests and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from rural_stroke_assist.client import ApiClient
from rural_stroke_assist.offline.store import LocalAttachment


class FailureAction(StrEnum):
    RETRY = "RETRY"
    REFRESH_AUTH = "REFRESH_AUTH"
    DEAD_LETTER = "DEAD_LETTER"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class FailureClassification:
    action: FailureAction
    indefinite: bool = False
    reason: str = ""


def classify_http_failure(status_code: int | None, *, error_code: str | None = None) -> FailureClassification:
    if status_code is None:
        return FailureClassification(FailureAction.RETRY, indefinite=True, reason="connectivity")
    if status_code == 401:
        return FailureClassification(FailureAction.REFRESH_AUTH, reason="authentication")
    if status_code == 429:
        return FailureClassification(FailureAction.RETRY, indefinite=True, reason="rate_limited")
    if 500 <= status_code <= 599:
        return FailureClassification(FailureAction.RETRY, indefinite=True, reason="server_error")
    if status_code == 409 and error_code == "idempotency_conflict":
        return FailureClassification(FailureAction.DEAD_LETTER, reason="idempotency_conflict")
    if status_code == 409:
        return FailureClassification(FailureAction.CONFLICT, reason="state_conflict")
    if status_code in {400, 403, 404, 413, 415, 422}:
        return FailureClassification(FailureAction.DEAD_LETTER, reason="permanent_validation_or_authorization")
    return FailureClassification(FailureAction.RETRY, indefinite=True, reason="unexpected_transport_failure")


class ApiClientTransport:
    """Adapter that keeps the worker independent of HTTP details."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client

    def _token(self, token: str | None) -> None:
        self.client.token = token or ""

    def create_case(self, payload: dict[str, Any], *, idempotency_key: str, token: str | None) -> dict[str, Any]:
        self._token(token)
        return self.client.create_case(case_id=payload["id"], facility=payload["facility"], patient_code=payload.get("patient_code"), assessment_input=payload.get("assessment_input", {}), idempotency_key=idempotency_key)

    def upload_attachment(self, attachment: LocalAttachment, data: bytes, *, idempotency_key: str, token: str | None) -> dict[str, Any]:
        self._token(token)
        filename = PurePosixPath(attachment.relative_path).name
        return self.client.upload_attachment(attachment.case_id, attachment.kind, data, filename, attachment.media_type, attachment_id=attachment.attachment_id, idempotency_key=idempotency_key)

    def import_assessment(self, payload: dict[str, Any], *, idempotency_key: str, token: str | None) -> dict[str, Any]:
        self._token(token)
        return self.client.import_assessment(payload, idempotency_key=idempotency_key)

    def get_case_with_etag(self, case_id: str, *, token: str | None) -> tuple[dict[str, Any], str | None]:
        self._token(token)
        return self.client.get_case_with_etag(case_id)

    def submit_case(self, case_id: str, payload: dict[str, Any], etag: str, *, idempotency_key: str, token: str | None) -> dict[str, Any]:
        self._token(token)
        return self.client.submit_case(case_id, payload, etag, idempotency_key=idempotency_key)
