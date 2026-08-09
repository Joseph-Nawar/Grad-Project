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
