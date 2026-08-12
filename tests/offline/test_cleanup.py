from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

from rural_stroke_assist.offline.contracts import (
    AssessmentEnvelope,
    SyncState,
    WorkflowState,
)
from rural_stroke_assist.offline.store import SQLiteOfflineStore


def test_cleanup_removes_only_graceful_synchronized_media(tmp_path: Path) -> None:
    store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    store.initialize()
    store.create_case("case-1", "post", None, {"session_id": "s1"})
    store.save_attachment(
        case_id="case-1",
        attachment_id="attachment-1",
        kind="face",
        media_type="image/jpeg",
        size_bytes=3,
        sha256="a" * 64,
        relative_path="case-1/face.jpg",
    )
    media = tmp_path / "media" / "case-1"
    media.mkdir(parents=True)
    path = media / "face.jpg"
    path.write_bytes(b"abc")
    os.utime(path, (time.time() - 100, time.time() - 100))
    store.save_assessment(
        "case-1",
        AssessmentEnvelope(
            case_id="case-1",
            assessment_id="assessment-1",
            request={},
            result={},
            provenance={"schema_version": 1, "model_ids": {}, "fusion_id": "canonical-fusion-v1"},
        ),
    )
    store.queue_case("case-1")
    store.set_remote_attachment("attachment-1", "remote-1")
    store.set_remote_case("case-1", version=2, submitted_hash="a" * 64)
    for index, event in enumerate(store.list_outbox("case-1")):
        leased = store.lease_next(
            worker_id="cleanup-test", now=f"2027-01-01T00:00:0{index}Z", lease_seconds=30
        )
        assert leased is not None and leased.event_id == event.event_id
        store.ack_event(event.event_id, now=f"2027-01-01T00:00:0{index}Z")
    store.set_sync_state("case-1", SyncState.SYNCED, workflow_state=WorkflowState.SYNCED)

    removed = store.cleanup_synchronized_media(grace_seconds=60, now=time.time())

    assert removed == 1
    assert not path.exists()
    assert store.get_attachment("attachment-1").remote_id == "remote-1"


@pytest.mark.parametrize(
    "state",
    [
        SyncState.PENDING,
        SyncState.RETRY_WAIT,
        SyncState.BLOCKED_AUTH,
        SyncState.CONFLICT,
        SyncState.DEAD_LETTER,
    ],
)
def test_cleanup_retains_media_for_every_unresolved_sync_state(
    tmp_path: Path, state: SyncState
) -> None:
    store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    store.initialize()
    store.create_case("case-1", "post", None, {"session_id": "s1"})
    store.save_attachment(
        case_id="case-1",
        attachment_id="attachment-1",
        kind="face",
        media_type="image/jpeg",
        size_bytes=3,
        sha256="a" * 64,
        relative_path="case-1/face.jpg",
    )
    media = tmp_path / "media" / "case-1"
    media.mkdir(parents=True)
    path = media / "face.jpg"
    path.write_bytes(b"abc")
    store.set_sync_state("case-1", state)

    assert store.cleanup_synchronized_media(grace_seconds=0, now=2_000_000_000) == 0
    assert path.exists()
