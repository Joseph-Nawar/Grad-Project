"""Managed SQLite persistence for one collector edge."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from rural_stroke_assist.offline.contracts import (
    AssessmentEnvelope,
    OutboxEventType,
    OutboxState,
    SyncState,
    WorkflowState,
)
from rural_stroke_assist.offline.envelope import canonical_assessment_hash, canonical_json

SCHEMA_VERSION = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ImmutableLocalSnapshotError(ValueError):
    """Raised when a queued or assessed immutable snapshot would be changed."""


@dataclass(frozen=True)
class LocalCase:
    case_id: str
    collector_identity: str
    facility: str
    patient_code: str | None
    workflow_state: WorkflowState
    sync_state: SyncState
    assessment_input: dict[str, Any]
    assessment_id: str | None
    assessment_envelope: AssessmentEnvelope | None
    assessment_hash: str | None
    remote_version: int | None
    remote_submitted_hash: str | None


@dataclass(frozen=True)
class LocalAttachment:
    attachment_id: str
    case_id: str
    kind: str
    media_type: str
    size_bytes: int
    sha256: str
    relative_path: str
    remote_id: str | None = None


@dataclass(frozen=True)
class OutboxEvent:
    event_id: str
    case_id: str
    event_type: OutboxEventType
    payload_schema_version: int
    payload: dict[str, Any]
    payload_hash: str
    sequence: int
    state: OutboxState
    attempt_count: int
    next_attempt_at: str
    last_error: str | None
    lease_owner: str | None
    lease_expires_at: str | None

    @property
    def idempotency_key(self) -> str:
        """The stable event identity used for every transport retry."""

        return self.event_id


class SQLiteOfflineStore:
    def __init__(self, database_path: str | Path, media_root: str | Path) -> None:
        self.database_path = Path(database_path)
        self.media_root = Path(media_root)

    @contextmanager
    def _connect(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.media_root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            current = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'version'"
            ).fetchone()
            if current is None:
                schema_statements = (
                    """
                    CREATE TABLE local_cases (
                        case_id TEXT PRIMARY KEY,
                        facility TEXT NOT NULL,
                        collector_identity TEXT NOT NULL,
                        patient_code TEXT,
                        workflow_state TEXT NOT NULL,
                        sync_state TEXT NOT NULL,
                        assessment_input_json TEXT NOT NULL,
                        assessment_id TEXT,
                        assessment_envelope_json TEXT,
                        assessment_hash TEXT,
                        remote_version INTEGER,
                        remote_submitted_hash TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE local_attachments (
                        attachment_id TEXT PRIMARY KEY,
                        case_id TEXT NOT NULL REFERENCES local_cases(case_id) ON DELETE CASCADE,
                        kind TEXT NOT NULL,
                        media_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
                        sha256 TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        remote_id TEXT
                    );
                    CREATE TABLE outbox_events (
                        event_id TEXT PRIMARY KEY,
                        case_id TEXT NOT NULL REFERENCES local_cases(case_id) ON DELETE CASCADE,
                        event_type TEXT NOT NULL,
                        payload_schema_version INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        state TEXT NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        next_attempt_at TEXT NOT NULL,
                        last_error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        lease_owner TEXT,
                        lease_expires_at TEXT,
                        UNIQUE(case_id, sequence)
                    );
                    CREATE INDEX ix_outbox_ready ON outbox_events(state, next_attempt_at, case_id, sequence)
                    """,
                )
                for statement in schema_statements:
                    for command in statement.split(";"):
                        if command.strip():
                            connection.execute(command)
                connection.execute(
                    "INSERT INTO schema_meta(key, value) VALUES ('version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(current["value"]) < 2:
                connection.execute(
                    "ALTER TABLE local_cases ADD COLUMN collector_identity TEXT NOT NULL DEFAULT 'unknown'"
                )
                connection.execute(
                    "UPDATE schema_meta SET value = ? WHERE key = 'version'",
                    (str(SCHEMA_VERSION),),
                )
            elif int(current["value"]) > SCHEMA_VERSION:
                raise RuntimeError("Offline database schema is newer than this runtime.")
            connection.execute("COMMIT")

    def create_case(
        self,
        case_id: str,
        facility: str,
        patient_code: str | None,
        assessment_input: dict[str, Any],
        collector_identity: str = "collector",
    ) -> LocalCase:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO local_cases(case_id, facility, collector_identity, patient_code, workflow_state, sync_state, assessment_input_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    facility,
                    collector_identity,
                    patient_code,
                    WorkflowState.DRAFT.value,
                    SyncState.IDLE.value,
                    canonical_json(assessment_input),
                    now,
                    now,
                ),
            )
            connection.execute("COMMIT")
        return self.get_case(case_id)

    def get_case(self, case_id: str) -> LocalCase:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM local_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise KeyError(case_id)
        return self._case_from_row(row)

    def save_attachment(
        self,
        *,
        case_id: str,
        attachment_id: str,
        kind: str,
        media_type: str,
        size_bytes: int,
        sha256: str,
        relative_path: str,
    ) -> LocalAttachment:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO local_attachments(attachment_id, case_id, kind, media_type, size_bytes, sha256, relative_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    attachment_id,
                    case_id,
                    kind,
                    media_type,
                    size_bytes,
                    sha256.lower(),
                    relative_path.replace("\\", "/"),
                ),
            )
            connection.execute("COMMIT")
        return self.get_attachment(attachment_id)

    def get_attachment(self, attachment_id: str) -> LocalAttachment:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM local_attachments WHERE attachment_id = ?", (attachment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(attachment_id)
        return LocalAttachment(**dict(row))

    def list_attachments(self, case_id: str) -> list[LocalAttachment]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM local_attachments WHERE case_id = ? ORDER BY attachment_id",
                (case_id,),
            ).fetchall()
        return [LocalAttachment(**dict(row)) for row in rows]

    def save_assessment(self, case_id: str, envelope: AssessmentEnvelope) -> LocalCase:
        if envelope.case_id != case_id:
            raise ValueError("Assessment envelope case ID does not match the local case.")
        serialized = canonical_json(envelope)
        digest = canonical_assessment_hash(envelope)
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT workflow_state, assessment_envelope_json, assessment_hash FROM local_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise KeyError(case_id)
            if row["assessment_envelope_json"] is not None and (
                row["assessment_hash"] != digest or row["assessment_envelope_json"] != serialized
            ):
                connection.execute("ROLLBACK")
                raise ImmutableLocalSnapshotError("The local assessment snapshot is immutable.")
            if row["workflow_state"] not in {
                WorkflowState.DRAFT.value,
                WorkflowState.ASSESSED.value,
            }:
                connection.execute("ROLLBACK")
                raise ImmutableLocalSnapshotError("Queued assessment snapshots are immutable.")
            connection.execute(
                "UPDATE local_cases SET workflow_state = ?, assessment_id = ?, assessment_envelope_json = ?, assessment_hash = ?, updated_at = ? WHERE case_id = ?",
                (
                    WorkflowState.ASSESSED.value,
                    envelope.assessment_id,
                    serialized,
                    digest,
                    now,
                    case_id,
                ),
            )
            connection.execute("COMMIT")
        return self.get_case(case_id)

    def queue_case(self, case_id: str) -> list[OutboxEvent]:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            case = connection.execute(
                "SELECT * FROM local_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            if case is None:
                connection.execute("ROLLBACK")
                raise KeyError(case_id)
            if (
                case["workflow_state"] != WorkflowState.ASSESSED.value
                or case["assessment_envelope_json"] is None
            ):
                connection.execute("ROLLBACK")
                raise ValueError("Only an assessed case can be queued.")
            existing = connection.execute(
                "SELECT COUNT(*) FROM outbox_events WHERE case_id = ?", (case_id,)
            ).fetchone()[0]
            if existing:
                connection.execute("ROLLBACK")
                raise ValueError("The case is already queued.")
            envelope = AssessmentEnvelope.model_validate(
                json.loads(case["assessment_envelope_json"])
            )
            attachments = connection.execute(
                "SELECT * FROM local_attachments WHERE case_id = ? ORDER BY attachment_id",
                (case_id,),
            ).fetchall()
            payloads: list[tuple[OutboxEventType, dict[str, Any]]] = [
                (
                    OutboxEventType.CREATE_CASE,
                    {
                        "id": case_id,
                        "facility": case["facility"],
                        "patient_code": case["patient_code"],
                        "assessment_input": json.loads(case["assessment_input_json"]),
                    },
                ),
            ]
            for attachment in attachments:
                payloads.append(
                    (
                        OutboxEventType.UPLOAD_ATTACHMENT,
                        {
                            "case_id": case_id,
                            "attachment_id": attachment["attachment_id"],
                            "kind": attachment["kind"],
                            "media_type": attachment["media_type"],
                            "size_bytes": attachment["size_bytes"],
                            "sha256": attachment["sha256"],
                        },
                    )
                )
            payloads.extend(
                [
                    (
                        OutboxEventType.IMPORT_ASSESSMENT,
                        {
                            "envelope": envelope.model_dump(mode="json"),
                            "assessment_hash": case["assessment_hash"],
                        },
                    ),
                    (
                        OutboxEventType.SUBMIT_CASE,
                        {"confirmed": True, "assessment_hash": case["assessment_hash"]},
                    ),
                ]
            )
            for sequence, (event_type, payload) in enumerate(payloads, start=1):
                event_id = str(uuid4())
                payload_hash = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
                connection.execute(
                    "INSERT INTO outbox_events(event_id, case_id, event_type, payload_schema_version, payload_json, payload_hash, sequence, state, next_attempt_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event_id,
                        case_id,
                        event_type.value,
                        1,
                        canonical_json(payload),
                        payload_hash,
                        sequence,
                        OutboxState.PENDING.value,
                        now,
                        now,
                        now,
                    ),
                )
            connection.execute(
                "UPDATE local_cases SET workflow_state = ?, sync_state = ?, updated_at = ? WHERE case_id = ?",
                (WorkflowState.QUEUED.value, SyncState.PENDING.value, now, case_id),
            )
            connection.execute("COMMIT")
        return self.list_outbox(case_id)

    def list_outbox(self, case_id: str) -> list[OutboxEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM outbox_events WHERE case_id = ? ORDER BY sequence", (case_id,)
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def lease_next(self, *, worker_id: str, now: str, lease_seconds: int) -> OutboxEvent | None:
        expires = datetime.fromisoformat(now.replace("Z", "+00:00"))
        lease_expires = expires.timestamp() + lease_seconds
        lease_at = (
            datetime.fromtimestamp(lease_expires, timezone.utc).isoformat().replace("+00:00", "Z")
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT event.*
                FROM outbox_events AS event
                WHERE (
                    (event.state IN ('PENDING', 'RETRY_WAIT') AND event.next_attempt_at <= ?)
                    OR (event.state = 'IN_FLIGHT' AND event.lease_expires_at <= ?)
                )
                AND NOT EXISTS (
                    SELECT 1 FROM outbox_events AS previous
                    WHERE previous.case_id = event.case_id
                      AND previous.sequence < event.sequence
                      AND previous.state <> 'ACKED'
                )
                ORDER BY event.created_at, event.case_id, event.sequence
                LIMIT 1
                """,
                (now, now),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute(
                "UPDATE outbox_events SET state = ?, attempt_count = attempt_count + 1, lease_owner = ?, lease_expires_at = ?, updated_at = ? WHERE event_id = ?",
                (OutboxState.IN_FLIGHT.value, worker_id, lease_at, now, row["event_id"]),
            )
            connection.execute("COMMIT")
        return self._event_from_row(
            {
                **dict(row),
                "state": OutboxState.IN_FLIGHT.value,
                "attempt_count": row["attempt_count"] + 1,
                "lease_owner": worker_id,
                "lease_expires_at": lease_at,
            }
        )

    def ack_event(self, event_id: str, *, now: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE outbox_events SET state = ?, lease_owner = NULL, lease_expires_at = NULL, updated_at = ? WHERE event_id = ? AND state = ?",
                (OutboxState.ACKED.value, now, event_id, OutboxState.IN_FLIGHT.value),
            ).rowcount
            if updated != 1:
                connection.execute("ROLLBACK")
                raise ValueError("Only an in-flight event can be acknowledged.")
            connection.execute("COMMIT")

    def transition_event(
        self,
        event_id: str,
        state: OutboxState,
        *,
        now: str,
        last_error: str | None = None,
        next_attempt_at: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE outbox_events SET state = ?, last_error = ?, next_attempt_at = COALESCE(?, next_attempt_at), lease_owner = NULL, lease_expires_at = NULL, updated_at = ? WHERE event_id = ?",
                (state.value, last_error, next_attempt_at, now, event_id),
            )
            connection.execute("COMMIT")

    def set_remote_case(
        self, case_id: str, *, version: int | None, submitted_hash: str | None = None
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE local_cases SET remote_version = ?, remote_submitted_hash = COALESCE(?, remote_submitted_hash), updated_at = ? WHERE case_id = ?",
                (version, submitted_hash, _now(), case_id),
            )
            connection.execute("COMMIT")

    def set_remote_attachment(self, attachment_id: str, remote_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE local_attachments SET remote_id = ? WHERE attachment_id = ?",
                (remote_id, attachment_id),
            )
            connection.execute("COMMIT")

    def set_sync_state(
        self, case_id: str, state: SyncState, *, workflow_state: WorkflowState | None = None
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if workflow_state is None:
                connection.execute(
                    "UPDATE local_cases SET sync_state = ?, updated_at = ? WHERE case_id = ?",
                    (state.value, _now(), case_id),
                )
            else:
                connection.execute(
                    "UPDATE local_cases SET sync_state = ?, workflow_state = ?, updated_at = ? WHERE case_id = ?",
                    (state.value, workflow_state.value, _now(), case_id),
                )
            connection.execute("COMMIT")

    def sync_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS total FROM outbox_events GROUP BY state"
            ).fetchall()
        return {str(row["state"]): int(row["total"]) for row in rows}

    def cleanup_synchronized_media(self, *, grace_seconds: float, now: float) -> int:
        removed = 0
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT local_cases.case_id
                FROM local_cases
                WHERE local_cases.workflow_state = ?
                  AND local_cases.sync_state = ?
                  AND local_cases.remote_submitted_hash IS NOT NULL
                  AND EXISTS (SELECT 1 FROM outbox_events WHERE case_id = local_cases.case_id)
                  AND NOT EXISTS (
                      SELECT 1 FROM outbox_events
                      WHERE case_id = local_cases.case_id AND state <> ?
                  )
                """,
                (WorkflowState.SYNCED.value, SyncState.SYNCED.value, OutboxState.ACKED.value),
            ).fetchall()
        for row in rows:
            for attachment in self.list_attachments(row["case_id"]):
                if attachment.remote_id is None:
                    continue
                path = (self.media_root / attachment.relative_path).resolve()
                if self.media_root.resolve() not in path.parents or not path.is_file():
                    continue
                if now - path.stat().st_mtime >= grace_seconds:
                    path.unlink()
                    removed += 1
        return removed

    @staticmethod
    def _case_from_row(row: sqlite3.Row) -> LocalCase:
        return LocalCase(
            case_id=row["case_id"],
            collector_identity=row["collector_identity"],
            facility=row["facility"],
            patient_code=row["patient_code"],
            workflow_state=WorkflowState(row["workflow_state"]),
            sync_state=SyncState(row["sync_state"]),
            assessment_input=json.loads(row["assessment_input_json"]),
            assessment_id=row["assessment_id"],
            assessment_envelope=None
            if row["assessment_envelope_json"] is None
            else AssessmentEnvelope.model_validate(json.loads(row["assessment_envelope_json"])),
            assessment_hash=row["assessment_hash"],
            remote_version=row["remote_version"],
            remote_submitted_hash=row["remote_submitted_hash"],
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> OutboxEvent:
        return OutboxEvent(
            event_id=row["event_id"],
            case_id=row["case_id"],
            event_type=OutboxEventType(row["event_type"]),
            payload_schema_version=row["payload_schema_version"],
            payload=json.loads(row["payload_json"]),
            payload_hash=row["payload_hash"],
            sequence=row["sequence"],
            state=OutboxState(row["state"]),
            attempt_count=row["attempt_count"],
            next_attempt_at=row["next_attempt_at"],
            last_error=row["last_error"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
        )
