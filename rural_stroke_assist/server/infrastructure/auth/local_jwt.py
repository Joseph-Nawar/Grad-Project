"""Signed local/test JWT verifier and token issuer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import jwt

from rural_stroke_assist.server.principal import Principal


class TokenVerifier(Protocol):
    def verify(self, token: str) -> Principal: ...


class LocalJwtVerifier:
    def __init__(self, *, secret: str, issuer: str, audience: str) -> None:
        self.secret = secret
        self.issuer = issuer
        self.audience = audience

    def verify(self, token: str) -> Principal:
        try:
            payload: dict[str, Any] = jwt.decode(token, self.secret, algorithms=["HS256"], issuer=self.issuer, audience=self.audience)
            subject = str(payload["sub"])
            roles = payload.get("roles", [])
            facilities = payload.get("facilities", [])
            if not isinstance(roles, list) or not isinstance(facilities, list):
                raise ValueError
            return Principal(subject, roles, facilities)
        except Exception as exc:
            raise ValueError("Invalid access token.") from exc


def create_demo_token(*, secret: str, issuer: str, audience: str, subject: str, roles: list[str], facilities: list[str], ttl_seconds: int = 3600) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": subject, "roles": roles, "facilities": facilities, "iss": issuer, "aud": audience, "iat": now, "exp": now + timedelta(seconds=ttl_seconds)}
    return jwt.encode(payload, secret, algorithm="HS256")
