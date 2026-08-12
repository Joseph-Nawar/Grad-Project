from __future__ import annotations

from pathlib import Path

from rural_stroke_assist.offline.contracts import AssessmentEnvelope, OutboxState
from rural_stroke_assist.offline.store import SQLiteOfflineStore


def test_expired_lease_is_recovered_and_order_is_preserved(tmp_path: Path) -> None:
    store = SQLiteOfflineStore(tmp_path / "collector.sqlite3", tmp_path / "media")
    store.initialize()
    store.create_case("case-1", "post", None, {"session_id": "s1"})
    store.save_assessment(
        "case-1",
        AssessmentEnvelope(case_id="case-1", assessment_id="assessment-1", request={"session_id": "s1"}, result={"status": "complete"}, provenance={"schema_version": 1, "model_ids": {}, "fusion_id": "fusion-v1"}),
    )
    store.queue_case("case-1")

    first = store.lease_next(worker_id="worker-a", now="2027-01-01T00:00:00Z", lease_seconds=30)
    assert first is not None
    assert first.state is OutboxState.IN_FLIGHT
    assert store.lease_next(worker_id="worker-b", now="2027-01-01T00:00:01Z", lease_seconds=30) is None

    recovered = store.lease_next(worker_id="worker-b", now="2027-01-01T00:01:00Z", lease_seconds=30)
    assert recovered is not None
    assert recovered.event_id == first.event_id
    assert recovered.lease_owner == "worker-b"
    assert recovered.attempt_count == first.attempt_count + 1

    store.ack_event(recovered.event_id, now="2027-01-01T00:01:01Z")
    next_event = store.lease_next(worker_id="worker-b", now="2027-01-01T00:01:02Z", lease_seconds=30)
    assert next_event is not None
    assert next_event.sequence == 2
