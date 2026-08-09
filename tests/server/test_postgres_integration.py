from __future__ import annotations

import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from uuid import uuid4

from rural_stroke_assist.server.infrastructure.db.models import CaseModel


pytestmark = pytest.mark.postgres


def test_alembic_applies_stage2_schema_to_real_postgresql() -> None:
    database_url = os.getenv("RURALSTROKE_DATABASE_URL")
    if not database_url:
        pytest.skip("BLOCKED: RURALSTROKE_DATABASE_URL is not configured; no SQLite substitution is permitted.")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert {"cases", "attachments", "assessments", "submissions", "reviews", "idempotency_records", "alembic_version"}.issubset(tables)
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar_one() == 1
            assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "0002_integrity"
            assert connection.execute(text("SELECT data_type FROM information_schema.columns WHERE table_name='cases' AND column_name='id'")).scalar_one() == "uuid"
        with Session(engine) as session:
            session.add(CaseModel(id=uuid4(), facility="pg-test", collector_subject="test", status="INVALID", assessment_input={}))
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()
        assert "uq_idempotency_actor_operation_key" in {item["name"] for item in inspect(engine).get_unique_constraints("idempotency_records")}
    finally:
        engine.dispose()
