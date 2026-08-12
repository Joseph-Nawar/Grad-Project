from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rural_stroke_assist.offline.contracts import AssessmentEnvelope, AttachmentRef, OutboxEventType, OutboxState, WorkflowState
from rural_stroke_assist.offline.store import ImmutableLocalSnapshotError, SQLiteOfflineStore


def envelope(case_id: str, assessment_id: str) -> AssessmentEnvelope:
    return AssessmentEnvelope(
        case_id=case_id,
        assessment_id=assessment_id,
        request={
            "session_id": "s1",
            "metadata": {"age": 60},
            "acute_symptoms": {"face_drooping": True},
            "attachments": [
                AttachmentRef(
                    attachment_id="00000000-0000-0000-0000-000000000003",
                    kind="face",
                    media_type="image/png",
                    size_bytes=4,
                    sha256="a" * 64,
                ).model_dump(mode="json")
            ],
        },
        result={"status": "partial", "value": 0.8},
        provenance={"schema_version": 1, "model_ids": {"acute_symptoms": "rules-v1"}, "fusion_id": "fusion-v1"},
    )


def prepared_store(tmp_path: Path) -> SQLiteOfflineStore:
    store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    store.initialize()
    store.create_case("00000000-0000-0000-0000-000000000001", "post", None, {"session_id": "s1"})
    store.save_attachment(
        case_id="00000000-0000-0000-0000-000000000001",
        attachment_id="00000000-0000-0000-0000-000000000003",
        kind="face",
        media_type="image/png",
        size_bytes=4,
        sha256="a" * 64,
        relative_path="cases/00000000-0000-0000-0000-000000000001/face.png",
    )
    store.save_assessment(
        "00000000-0000-0000-0000-000000000001",
        envelope("00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"),
    )
    return store


def test_sqlite_store_migrates_with_wal_foreign_keys_and_restart_persistence(tmp_path: Path) -> None:
    store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    store.initialize()
    with store._connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
    store.create_case("case-1", "post", "p1", {"session_id": "s1"})
    restarted = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    restarted.initialize()
    assert restarted.get_case("case-1").facility == "post"


def test_queue_case_atomically_freezes_snapshot_and_creates_ordered_intents(tmp_path: Path) -> None:
    store = prepared_store(tmp_path)

    events = store.queue_case("00000000-0000-0000-0000-000000000001")

    assert [event.event_type for event in events] == [
        OutboxEventType.CREATE_CASE,
        OutboxEventType.UPLOAD_ATTACHMENT,
        OutboxEventType.IMPORT_ASSESSMENT,
        OutboxEventType.SUBMIT_CASE,
    ]
    assert [event.sequence for event in events] == [1, 2, 3, 4]
    assert all(event.state is OutboxState.PENDING for event in events)
    case = store.get_case("00000000-0000-0000-0000-000000000001")
    assert case.workflow_state is WorkflowState.QUEUED
    assert case.assessment_hash
    assert all(event.event_id and event.idempotency_key for event in events)

    with pytest.raises(ImmutableLocalSnapshotError):
        store.save_assessment(
            case.case_id,
            envelope(case.case_id, "00000000-0000-0000-0000-000000000099"),
        )


def test_queue_case_rolls_back_when_case_is_not_assessed(tmp_path: Path) -> None:
    store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    store.initialize()
    store.create_case("case-1", "post", None, {"session_id": "s1"})

    with pytest.raises(ValueError, match="assessed"):
        store.queue_case("case-1")

    assert store.list_outbox("case-1") == []
    assert store.get_case("case-1").workflow_state is WorkflowState.DRAFT
