from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from rural_stroke_assist.server.app import create_app
from rural_stroke_assist.server.config import ServerSettings
from rural_stroke_assist.server.infrastructure.auth.local_jwt import create_demo_token


pytestmark = pytest.mark.postgres


def test_concurrent_case_creation_has_one_effect_and_safe_replay() -> None:
    import os

    database_url = os.getenv("RURALSTROKE_DATABASE_URL")
    if not database_url:
        pytest.skip("BLOCKED: RURALSTROKE_DATABASE_URL is not configured; no SQLite substitution is permitted.")
    secret = "postgres-concurrency-test-secret-0123456789"

    with TemporaryDirectory() as temp:
        settings = ServerSettings(database_url=database_url, jwt_secret=secret, storage_root=Path(temp))
        app = create_app(settings=settings, assessment_service=object())
        case_id = str(uuid4())
        subject = f"concurrent-{case_id}"
        key = f"concurrent-case-{case_id}"
        token = create_demo_token(secret=secret, issuer=settings.jwt_issuer, audience=settings.jwt_audience, subject=subject, roles=["collector"], facilities=["concurrency-facility"])
        body = {"id": case_id, "facility": "concurrency-facility", "patient_code": None, "assessment_input": {}}
        with TestClient(app) as client:
            def submit(_: int) -> int:
                response = client.post("/api/v1/cases", json=body, headers={"Authorization": f"Bearer {token}", "Idempotency-Key": key})
                return response.status_code

            with ThreadPoolExecutor(max_workers=5) as pool:
                statuses = list(pool.map(submit, range(5)))
            conflict = client.post("/api/v1/cases", json={**body, "patient_code": "different"}, headers={"Authorization": f"Bearer {token}", "Idempotency-Key": key})
        engine = create_engine(database_url)
        try:
            with engine.connect() as connection:
                effects = connection.execute(text("SELECT count(*) FROM cases WHERE id = :id"), {"id": case_id}).scalar_one()
                records = connection.execute(text("SELECT count(*) FROM idempotency_records WHERE actor_subject = :subject AND operation = 'case_create' AND key = :key"), {"subject": subject, "key": key}).scalar_one()
        finally:
            engine.dispose()
        assert statuses == [201] * 5
        assert conflict.status_code == 409
        assert effects == 1
        assert records == 1
