"""Typed synchronous HTTP client used by both Streamlit applications."""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID, uuid4

import httpx


class ApiClientError(RuntimeError):
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.payload = payload or {}
        super().__init__(self.payload.get("message", "API request failed."))


class ApiClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, *, client: httpx.Client | None = None) -> None:
        self.base_url = (base_url or os.getenv("RURALSTROKE_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.token = token or os.getenv("RURALSTROKE_API_TOKEN", "")
        self.client = client or httpx.Client(base_url=self.base_url, timeout=60.0)

    def _request(self, method: str, path: str, *, idempotency_key: str | None = None, etag: str | None = None, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", {}))
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if etag:
            headers["If-Match"] = etag
        try:
            response = self.client.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiClientError(503, {"message": "The API is unavailable."}) from exc
        if response.status_code >= 400:
            try:
                payload = response.json()
            except ValueError:
                payload = {"message": "The API returned an invalid error response."}
            raise ApiClientError(response.status_code, payload)
        if response.status_code == 204:
            return None
        return response.json()

    def create_case(self, *, case_id: str | None, facility: str, patient_code: str | None, assessment_input: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/cases", idempotency_key=str(uuid4()), json={"id": case_id or str(uuid4()), "facility": facility, "patient_code": patient_code, "assessment_input": assessment_input})

    def list_cases(self, *, cursor: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/api/v1/cases", params={"cursor": cursor} if cursor else None)

    def get_case(self, case_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/cases/{case_id}")

    def update_case(self, case_id: str, payload: dict[str, Any], etag: str) -> dict[str, Any]:
        return self._request("PATCH", f"/api/v1/cases/{case_id}", etag=etag, json=payload)

    def upload_attachment(self, case_id: str, kind: str, data: bytes, filename: str, media_type: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/attachments", idempotency_key=str(uuid4()), data={"case_id": case_id, "kind": kind}, files={"file": (filename, data, media_type)})

    def create_assessment(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = {**payload, "id": payload.get("id", str(uuid4()))}
        return self._request("POST", "/api/v1/assessments", idempotency_key=str(uuid4()), json=payload)

    def submit_case(self, case_id: str, payload: dict[str, Any], etag: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/cases/{case_id}/submit", idempotency_key=str(uuid4()), etag=etag, json=payload)

    def get_snapshot(self, case_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/cases/{case_id}/snapshot")

    def read_attachment_url(self, attachment_id: str) -> str:
        return f"{self.base_url}/api/v1/attachments/{attachment_id}"

    def read_attachment(self, attachment_id: str) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        response = self.client.get(f"/api/v1/attachments/{attachment_id}", headers=headers)
        if response.status_code >= 400:
            raise ApiClientError(response.status_code, response.json())
        return response.content

    def claim_review(self, case_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/cases/{case_id}/review-claim", idempotency_key=str(uuid4()))

    def create_review(self, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/cases/{case_id}/reviews", idempotency_key=str(uuid4()), json=payload)

    def list_reviews(self, case_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/api/v1/cases/{case_id}/reviews")
