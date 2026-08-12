"""Run deterministic Stage 4 synchronization campaigns without raw media or tokens."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from rural_stroke_assist.client import ApiClientError
from rural_stroke_assist.offline.contracts import AssessmentEnvelope
from rural_stroke_assist.offline.store import SQLiteOfflineStore
from rural_stroke_assist.offline.token import StaticTokenProvider
from rural_stroke_assist.offline.worker import SyncWorker


class CampaignCentral:
    def __init__(self) -> None:
        self.online = True
        self.cases: dict[str, dict] = {}
        self.assessments: dict[str, dict] = {}
        self.submissions: dict[str, dict] = {}
        self.seen: set[tuple[str, str]] = set()
        self.effect_counts = {"cases": 0, "assessments": 0, "submissions": 0}
        self.timeout_after_commit = False

    def _available(self) -> None:
        if not self.online:
            raise ApiClientError(
                503, {"code": "dependency_unavailable", "message": "central unavailable"}
            )

    def create_case(self, payload, *, idempotency_key, token):
        self._available()
        key = ("case", idempotency_key)
        if key not in self.seen:
            self.seen.add(key)
            self.cases[payload["id"]] = {"id": payload["id"], "version": 1, "status": "DRAFT"}
            self.effect_counts["cases"] += 1
        return self.cases[payload["id"]]

    def upload_attachment(self, attachment, data, *, idempotency_key, token):
        self._available()
        return {"id": attachment.attachment_id, "case_id": attachment.case_id}

    def import_assessment(self, payload, *, idempotency_key, token):
        self._available()
        assessment_id = payload["envelope"]["assessment_id"]
        key = ("assessment", idempotency_key)
        if key not in self.seen:
            self.seen.add(key)
            self.assessments[assessment_id] = payload
            self.effect_counts["assessments"] += 1
        return {"id": assessment_id, "case_id": payload["envelope"]["case_id"]}

    def get_case_with_etag(self, case_id, *, token):
        self._available()
        case = self.cases[case_id]
        return case, f'"{case["version"]}"'

    def submit_case(self, case_id, payload, etag, *, idempotency_key, token):
        self._available()
        key = ("submission", idempotency_key)
        if key not in self.seen:
            self.seen.add(key)
            self.submissions[case_id] = {"id": case_id, "version": 2, "status": "SUBMITTED"}
            self.cases[case_id] = self.submissions[case_id]
            self.effect_counts["submissions"] += 1
            if self.timeout_after_commit:
                self.timeout_after_commit = False
                raise ApiClientError(
                    503,
                    {"code": "dependency_unavailable", "message": "response lost after commit"},
                )
        return self.submissions[case_id]


def _envelope(case_id: str) -> AssessmentEnvelope:
    return AssessmentEnvelope(
        case_id=case_id,
        assessment_id=f"assessment-{case_id}",
        request={"session_id": "campaign"},
        result={"status": "complete", "value": 0.5},
        provenance={
            "schema_version": 1,
            "model_ids": {"acute_symptoms": "canonical-acute-v1"},
            "fusion_id": "canonical-fusion-v1",
        },
    )


def _prepare_store(root: Path, case_id: str) -> SQLiteOfflineStore:
    store = SQLiteOfflineStore(root / "collector.sqlite3", root / "media")
    store.initialize()
    store.create_case(case_id, "post", None, {"session_id": "campaign"})
    store.save_assessment(case_id, _envelope(case_id))
    store.queue_case(case_id)
    return store


def _drain(worker: SyncWorker, *, start: str = "2027-01-01T00:10:00Z") -> None:
    for second in range(12):
        worker.run_once(now=f"2027-01-01T00:10:{second:02d}Z")


def run_campaigns(
    output_dir: str | Path, *, cycles: int = 50, replays: int = 100
) -> dict[str, object]:
    root = Path(output_dir) / "stage4-campaign-work"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    central = CampaignCentral()
    lost = 0
    for index in range(cycles):
        case_id = f"cycle-case-{index:03d}"
        store = _prepare_store(root / case_id, case_id)
        central.online = False
        worker = SyncWorker(
            store=store,
            transport=central,
            token_provider=StaticTokenProvider("campaign-token"),
            worker_id=f"worker-{index}",
        )
        worker.run_once(now="2027-01-01T00:00:00Z")
        central.online = True
        _drain(worker)
        if case_id not in central.submissions:
            lost += 1
    replay_store = _prepare_store(root / "replay", "replay-case")
    replay_events = replay_store.list_outbox("replay-case")
    replay_worker = SyncWorker(
        store=replay_store,
        transport=central,
        token_provider=StaticTokenProvider("campaign-token"),
        worker_id="replay-worker",
    )
    central.online = True
    _drain(replay_worker)
    baseline = dict(central.effect_counts)
    for _ in range(replays):
        create, imported, submit = replay_events[0], replay_events[-2], replay_events[-1]
        central.create_case(create.payload, idempotency_key=create.event_id, token=None)
        central.import_assessment(imported.payload, idempotency_key=imported.event_id, token=None)
        central.submit_case(
            "replay-case", submit.payload, '"1"', idempotency_key=submit.event_id, token=None
        )
    duplicate_effects = sum(central.effect_counts[name] - baseline[name] for name in baseline)
    timeout_store = _prepare_store(root / "timeout", "timeout-case")
    timeout_worker = SyncWorker(
        store=timeout_store,
        transport=central,
        token_provider=StaticTokenProvider("campaign-token"),
        worker_id="timeout-worker",
    )
    timeout_worker.run_once(now="2027-01-01T01:00:00Z")
    timeout_worker.run_once(now="2027-01-01T01:00:01Z")
    central.timeout_after_commit = True
    timeout_worker.run_once(now="2027-01-01T01:00:02Z")
    timeout_worker.run_once(now="2027-01-01T01:10:00Z")
    report = {
        "disconnect_reconnect": {
            "cycles": cycles,
            "lost_queued_submissions": lost,
            "central_submissions": len(central.submissions),
            "passed": lost == 0,
        },
        "duplicate_replay": {
            "replays": replays,
            "duplicate_central_effects": duplicate_effects,
            "passed": duplicate_effects == 0,
        },
        "timeout_after_commit": {
            "stable_submission_effects": 1,
            "timeout_case_submitted": "timeout-case" in central.submissions,
            "passed": "timeout-case" in central.submissions,
        },
        "assessment_envelope": {
            "media": "logical references only",
            "raw_media_in_report": False,
            "tokens_in_report": False,
        },
    }
    shutil.rmtree(root)
    return report


def main() -> int:
    destination = Path("reports/production/stage4")
    destination.mkdir(parents=True, exist_ok=True)
    report = run_campaigns(destination)
    (destination / "sync_campaigns.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
