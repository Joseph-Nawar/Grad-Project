"""SQLite case repository with transactional, concurrent-safe access."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from rural_stroke_assist.cases.contracts import AttachmentKind, AttachmentReference, AuditEvent, Case, CaseStatus, ClinicianReview
from rural_stroke_assist.cases.exceptions import CaseNotFoundError, ImmutableSnapshotError
from rural_stroke_assist.cases.serialization import case_to_dict, dumps_json, loads_json


SCHEMA_VERSION = 1


class SQLiteCaseRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, case_json TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS audit_events (case_id TEXT NOT NULL, event_index INTEGER NOT NULL, event_json TEXT NOT NULL, PRIMARY KEY(case_id, event_index), FOREIGN KEY(case_id) REFERENCES cases(case_id))")
            connection.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)", (str(SCHEMA_VERSION),))
            connection.execute("COMMIT")

    def save(self, case: Case) -> None:
        payload = dumps_json(case_to_dict(case))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT case_json FROM cases WHERE case_id = ?", (case.case_id,)).fetchone()
            if existing is not None:
                previous = loads_json(existing["case_json"])
                if previous["status"] in {CaseStatus.SUBMITTED.value, CaseStatus.IN_REVIEW.value, CaseStatus.REVIEWED_AGREED.value, CaseStatus.REVIEWED_OVERRIDDEN.value}:
                    if previous["assessment_result"] != case_to_dict(case)["assessment_result"] or previous["assessment_input"] != case_to_dict(case)["assessment_input"]:
                        connection.execute("ROLLBACK")
                        raise ImmutableSnapshotError("Submitted assessment snapshots are immutable.")
            connection.execute("INSERT OR REPLACE INTO cases(case_id, status, created_at, updated_at, case_json) VALUES (?, ?, ?, ?, ?)", (case.case_id, case.status.value, case.created_at, case.updated_at, payload))
            connection.execute("DELETE FROM audit_events WHERE case_id = ?", (case.case_id,))
            for index, event in enumerate(case.audit_events):
                connection.execute("INSERT INTO audit_events(case_id, event_index, event_json) VALUES (?, ?, ?)", (case.case_id, index, dumps_json({"event_type": event.event_type, "at": event.at, "actor": event.actor, "detail": event.detail})))
            connection.execute("COMMIT")

    def get(self, case_id: str) -> Case:
        with self._connect() as connection:
            row = connection.execute("SELECT case_json FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise CaseNotFoundError(case_id)
        return self._from_dict(loads_json(row["case_json"]))

    def list(self, *, statuses: Sequence[CaseStatus] | None = None) -> list[Case]:
        with self._connect() as connection:
            if statuses:
                marks = ",".join("?" for _ in statuses)
                rows = connection.execute(f"SELECT case_json FROM cases WHERE status IN ({marks}) ORDER BY created_at ASC, case_id ASC", tuple(item.value for item in statuses)).fetchall()
            else:
                rows = connection.execute("SELECT case_json FROM cases ORDER BY created_at ASC, case_id ASC").fetchall()
        return [self._from_dict(loads_json(row["case_json"])) for row in rows]

    @staticmethod
    def _from_dict(value: dict) -> Case:
        return Case(
            case_id=value["case_id"], patient_code=value.get("patient_code"), collector_identity=value["collector_identity"], facility=value["facility"],
            assessment_input=value["assessment_input"], attachments=tuple(AttachmentReference(AttachmentKind(item["kind"]), item["relative_path"], item["media_type"], item["size_bytes"]) for item in value["attachments"]),
            assessment_result=value.get("assessment_result"), created_at=value["created_at"], updated_at=value["updated_at"], submitted_at=value.get("submitted_at"), status=CaseStatus(value["status"]),
            clinician_review=None if value.get("clinician_review") is None else ClinicianReview(**value["clinician_review"]),
            audit_events=tuple(AuditEvent(**item) for item in value.get("audit_events", [])),
        )
