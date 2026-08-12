"""Small OIDC authorization-code/PKCE bridge for cloud Streamlit clients."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from rural_stroke_assist.client import ApiClient


@dataclass(frozen=True)
class OIDCSettings:
    issuer: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: str = "openid email profile"

    @classmethod
    def from_environment(cls) -> "OIDCSettings":
        secret = os.getenv("RURALSTROKE_OIDC_CLIENT_SECRET", "")
        secret_file = os.getenv("RURALSTROKE_OIDC_CLIENT_SECRET_FILE")
        if secret_file:
            secret = Path(secret_file).read_text(encoding="utf-8").strip()
        values = {
            "issuer": os.getenv("RURALSTROKE_OIDC_ISSUER", "").rstrip("/"),
            "client_id": os.getenv("RURALSTROKE_OIDC_CLIENT_ID", ""),
            "client_secret": secret,
            "redirect_uri": os.getenv("RURALSTROKE_OIDC_REDIRECT_URI", ""),
            "scope": os.getenv("RURALSTROKE_OIDC_SCOPE", cls.scope),
        }
        if not all(values[key] for key in ("issuer", "client_id", "client_secret", "redirect_uri")):
            raise RuntimeError("Cognito OIDC issuer, client ID, client secret, and redirect URI are required.")
        return cls(**values)


def build_authorization_url(settings: OIDCSettings, *, state: str, code_challenge: str) -> str:
    parameters = {
        'response_type': 'code',
        'client_id': settings.client_id,
        'redirect_uri': settings.redirect_uri,
        'scope': settings.scope,
        'state': state,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256',
    }
    return f"{settings.issuer}/oauth2/authorize?{urlencode(parameters)}"


def exchange_code(settings: OIDCSettings, *, code: str, code_verifier: str, client: httpx.Client | None = None) -> dict[str, Any]:
    owned_client = client is None
    http_client = client or httpx.Client(timeout=15.0)
    try:
        response = http_client.post(
            f"{settings.issuer}/oauth2/token",
            data={
                "grant_type": "authorization_code",
                "client_id": settings.client_id,
                "client_secret": settings.client_secret,
                "code": code,
                "redirect_uri": settings.redirect_uri,
                "code_verifier": code_verifier,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
            raise RuntimeError("OIDC token response did not contain an access token.")
        return payload
    finally:
        if owned_client:
            http_client.close()


def _refresh_token(settings: OIDCSettings, refresh_token: str) -> dict[str, Any]:
    response = httpx.post(
        f"{settings.issuer}/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "client_id": settings.client_id,
            "client_secret": settings.client_secret,
            "refresh_token": refresh_token,
        },
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
        raise RuntimeError("OIDC refresh response did not contain an access token.")
    return payload


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _expired(access_token: str, skew_seconds: int = 30) -> bool:
    try:
        payload = access_token.split(".")[1]
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        exp = int(__import__("json").loads(decoded)["exp"])
        return exp <= int(time.time()) + skew_seconds
    except (IndexError, KeyError, TypeError, ValueError, UnicodeDecodeError):
        return True


def streamlit_api_client(st: Any) -> ApiClient | None:
    """Return a token-bearing API client, or ``None`` while login is needed."""

    if os.getenv("RURALSTROKE_AUTH_MODE", "local").lower() != "cognito":
        return ApiClient()
    settings = OIDCSettings.from_environment()
    session = st.session_state
    verifier, challenge = session.setdefault("oidc_pkce", _pkce())
    state = session.setdefault("oidc_state", secrets.token_urlsafe(32))
    tokens = session.get("oidc_tokens")
    code = st.query_params.get("code")
    returned_state = st.query_params.get("state")
    if not tokens and code:
        if returned_state != state:
            raise RuntimeError("OIDC state validation failed.")
        tokens = exchange_code(settings, code=str(code), code_verifier=verifier)
        session["oidc_tokens"] = tokens
        st.query_params.clear()
    if tokens and _expired(str(tokens.get("access_token", ""))):
        refresh = tokens.get("refresh_token")
        if refresh:
            tokens = {**tokens, **_refresh_token(settings, str(refresh))}
            session["oidc_tokens"] = tokens
        else:
            session.pop("oidc_tokens", None)
            tokens = None
    if not tokens:
        url = build_authorization_url(settings, state=state, code_challenge=challenge)
        st.markdown(f"[Sign in with Cognito]({url})")
        return None
    return ApiClient(token=str(tokens["access_token"]))
