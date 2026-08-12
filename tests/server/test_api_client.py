from __future__ import annotations

import httpx

from rural_stroke_assist.client import ApiClient


def test_typed_client_uses_versioned_routes_and_bearer_token() -> None:
    seen: list[tuple[str, str, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.headers)))
        return httpx.Response(200, json={"items": [], "page": {"next_cursor": None}})

    client = ApiClient("http://api", "token", client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api"))
    assert client.list_cases() == {"items": [], "page": {"next_cursor": None}}
    assert seen[0][1] == "/api/v1/cases"
    assert seen[0][2]["authorization"] == "Bearer token"


def test_sync_client_sends_stable_resource_ids_and_event_idempotency_keys() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/cases/00000000-0000-0000-0000-000000000001"):
            return httpx.Response(200, headers={"ETag": '"3"'}, json={"id": request.url.path.rsplit("/", 1)[-1], "version": 3})
        if request.url.path.endswith("/assessments/import"):
            return httpx.Response(201, json={"id": "assessment", "case_id": "case", "status": "complete", "request_snapshot": {}, "result_snapshot": {}, "created_at": "now"})
        if request.url.path.endswith("/attachments"):
            return httpx.Response(201, json={"id": "attachment", "case_id": "case", "kind": "face", "media_type": "image/jpeg", "size_bytes": 3, "checksum_sha256": "a" * 64})
        return httpx.Response(201, json={"id": "case", "version": 1})

    client = ApiClient("http://api", "token", client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api"))
    client.create_case(case_id="00000000-0000-0000-0000-000000000001", facility="post", patient_code=None, assessment_input={}, idempotency_key="event-case")
    client.upload_attachment("case", "face", b"abc", "capture.jpg", "image/jpeg", attachment_id="attachment", idempotency_key="event-attachment")
    client.import_assessment({"envelope": {"case_id": "case"}, "assessment_hash": "a" * 64}, idempotency_key="event-import")
    _, etag = client.get_case_with_etag("00000000-0000-0000-0000-000000000001")

    assert seen[0].headers["idempotency-key"] == "event-case"
    assert b'name="attachment_id"' in seen[1].content
    assert seen[1].headers["idempotency-key"] == "event-attachment"
    assert seen[2].headers["idempotency-key"] == "event-import"
    assert etag == '"3"'
