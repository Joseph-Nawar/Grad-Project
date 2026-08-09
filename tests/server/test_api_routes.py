from __future__ import annotations

import httpx
import pytest

from rural_stroke_assist.server.app import create_app
from rural_stroke_assist.server.infrastructure.auth.local_jwt import LocalJwtVerifier, create_demo_token


@pytest.mark.asyncio
async def test_asgi_operational_and_identity_routes_use_correlation_and_safe_auth_errors() -> None:
    app = create_app(initialize_resources=False)
    app.state.token_verifier = LocalJwtVerifier(secret="test-secret", issuer="test", audience="api")
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            live = await client.get("/health/live", headers={"X-Correlation-ID": "00000000-0000-0000-0000-000000000001"})
            assert live.status_code == 200
            assert live.headers["X-Correlation-ID"] == "00000000-0000-0000-0000-000000000001"
            unauthenticated = await client.get("/api/v1/identity/me")
            assert unauthenticated.status_code == 401
            assert set(unauthenticated.json()) == {"code", "message", "details", "correlation_id"}
            class NullSession:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

            app.state.session_factory = NullSession
            token = create_demo_token(secret="test-secret", issuer="test", audience="api", subject="collector-1", roles=["collector"], facilities=["facility-a"])
            invalid = await client.post("/api/v1/cases", json={}, headers={"Authorization": f"Bearer {token}"})
            assert invalid.status_code == 422
            assert set(invalid.json()) == {"code", "message", "details", "correlation_id"}


@pytest.mark.asyncio
async def test_asgi_identity_uses_signed_local_jwt_not_role_headers() -> None:
    app = create_app(initialize_resources=False)
    app.state.token_verifier = LocalJwtVerifier(secret="test-secret", issuer="test", audience="api")
    token = create_demo_token(secret="test-secret", issuer="test", audience="api", subject="collector-1", roles=["collector"], facilities=["facility-a"])
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/identity/me", headers={"Authorization": f"Bearer {token}", "X-Role": "demo-admin"})
            assert response.status_code == 200
            assert response.json()["roles"] == ["collector"]
