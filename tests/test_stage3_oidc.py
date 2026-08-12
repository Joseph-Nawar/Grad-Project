from __future__ import annotations

import httpx

from rural_stroke_assist.ui.oidc import OIDCSettings, build_authorization_url, exchange_code


def test_oidc_authorization_url_uses_pkce_and_callback_state() -> None:
    settings = OIDCSettings(
        issuer="https://ruralstroke.auth.eu-central-1.amazoncognito.com",
        client_id="client-1",
        client_secret="client-secret",
        redirect_uri="https://demo.example.org/collector",
    )

    url = build_authorization_url(settings, state="state-1", code_challenge="challenge-1")

    assert "response_type=code" in url
    assert "client_id=client-1" in url
    assert "redirect_uri=https%3A%2F%2Fdemo.example.org%2Fcollector" in url
    assert "code_challenge=challenge-1" in url
    assert "code_challenge_method=S256" in url
    assert "state=state-1" in url


def test_exchange_code_returns_access_token_and_keeps_secret_out_of_url() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})

    settings = OIDCSettings(
        issuer="https://ruralstroke.auth.eu-central-1.amazoncognito.com",
        client_id="client-1",
        client_secret="client-secret",
        redirect_uri="https://demo.example.org/collector",
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))

    tokens = exchange_code(settings, code="code-1", code_verifier="verifier-1", client=client)

    assert tokens["access_token"] == "access"
    assert "client-secret" not in str(requests[0].url)
    assert "client_secret=client-secret" in str(requests[0].content)
    assert requests[0].url.path == "/oauth2/token"
