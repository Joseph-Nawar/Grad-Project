"""Synchronous SQLAlchemy engine/session helpers."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_db_engine(database_url: str | None) -> Engine:
    if not database_url:
        raise RuntimeError("RURALSTROKE_DATABASE_URL is required; SQLite fallback is not supported.")
    return create_engine(database_url, future=True, pool_pre_ping=True)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False, autoflush=False)


def operation_session(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as session:
        yield session
