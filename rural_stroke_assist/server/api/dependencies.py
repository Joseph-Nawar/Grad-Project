"""FastAPI dependencies for sessions and server-side authorization."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from rural_stroke_assist.server.errors import ApiError
from rural_stroke_assist.server.app import create_token_verifier
from rural_stroke_assist.server.principal import Principal

bearer = HTTPBearer(auto_error=False)


def get_session(request: Request) -> Generator[Session, None, None]:
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise ApiError("dependency_unavailable", "Database session is unavailable.", status_code=503)
    with factory() as session:
        yield session


def get_principal(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
    if credentials is None:
        raise ApiError("authentication_required", "A bearer token is required.", status_code=401)
    verifier = getattr(request.app.state, "token_verifier", None)
    if verifier is None:
        settings = getattr(request.app.state, "settings", None)
        if settings is None:
            raise ApiError("dependency_unavailable", "Authentication is unavailable.", status_code=503)
        verifier = create_token_verifier(settings)
    try:
        return verifier.verify(credentials.credentials)
    except ValueError as exc:
        raise ApiError("authentication_required", "The access token is invalid.", status_code=401) from exc


def require_role(role: str):
    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has_role(role):
            raise ApiError("forbidden", "The principal is not authorized for this operation.", status_code=403)
        return principal

    return dependency
