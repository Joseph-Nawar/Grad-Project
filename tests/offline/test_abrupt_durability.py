from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from rural_stroke_assist.offline.contracts import AssessmentEnvelope, WorkflowState
from rural_stroke_assist.offline.store import SQLiteOfflineStore


def test_abrupt_process_termination_leaves_valid_recoverable_store(tmp_path: Path) -> None:
    database = tmp_path / "collector.sqlite3"
    media = tmp_path / "media"
    store = SQLiteOfflineStore(database, media)
    store.initialize()
    store.create_case("case-1", "post", None, {"session_id": "s1"})
    store.save_assessment(
        "case-1",
        AssessmentEnvelope(
            case_id="case-1",
            assessment_id="assessment-1",
            request={"session_id": "s1"},
            result={"status": "complete"},
            provenance={"schema_version": 1, "model_ids": {}, "fusion_id": "canonical-fusion-v1"},
        ),
    )

    child_code = """
import os
import sqlite3
connection = sqlite3.connect(r'{database}', isolation_level=None)
connection.execute('PRAGMA journal_mode=WAL')
connection.execute('BEGIN IMMEDIATE')
connection.execute(
    \"INSERT INTO outbox_events(event_id, case_id, event_type, payload_schema_version, payload_json, payload_hash, sequence, state, next_attempt_at, created_at, updated_at) VALUES ('partial', 'case-1', 'CREATE_CASE', 1, '{{}}', 'x', 1, 'PENDING', '2027-01-01T00:00:00Z', '2027-01-01T00:00:00Z', '2027-01-01T00:00:00Z')\"
)
os._exit(137)
""".format(database=str(database).replace("\\", "\\\\"))
    completed = subprocess.run([sys.executable, "-c", child_code], check=False, timeout=30)
    assert completed.returncode == 137

    restarted = SQLiteOfflineStore(database, media)
    restarted.initialize()
    with restarted._connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            connection.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()[0]
            == "2"
        )
        assert connection.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 0
    assert restarted.get_case("case-1").workflow_state is WorkflowState.ASSESSED

    restarted.queue_case("case-1")
    leased = restarted.lease_next(
        worker_id="worker-a", now="2027-01-01T00:00:00Z", lease_seconds=30
    )
    assert leased is not None
    recovered = restarted.lease_next(
        worker_id="worker-b", now="2027-01-01T00:01:00Z", lease_seconds=30
    )
    assert recovered is not None
    assert recovered.event_id == leased.event_id
