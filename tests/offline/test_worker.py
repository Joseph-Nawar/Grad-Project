from __future__ import annotations

from pathlib import Path

from rural_stroke_assist.client import ApiClientError
from rural_stroke_assist.offline.contracts import AssessmentEnvelope, OutboxState, SyncState
from rural_stroke_assist.offline.store import SQLiteOfflineStore
from rural_stroke_assist.offline.token import FileTokenProvider, StaticTokenProvider
from rural_stroke_assist.offline.worker import SyncWorker


def prepared_store(tmp_path: Path) -> SQLiteOfflineStore:
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
    (tmp_path / "media" / "case-1").mkdir(parents=True)
    (tmp_path / "media" / "case-1" / "face.jpg").write_bytes(b"abc")
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
    store.queue_case("case-1")
    return store


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def create_case(self, payload, *, idempotency_key, token):
        self.calls.append(("create", idempotency_key))
        return {"id": payload["id"], "version": 1}

    def upload_attachment(self, attachment, data, *, idempotency_key, token):
        self.calls.append(("upload", idempotency_key))
        return {"id": attachment.attachment_id, "case_id": attachment.case_id}

    def import_assessment(self, payload, *, idempotency_key, token):
        self.calls.append(("import", idempotency_key))
        return {
            "id": payload["envelope"]["assessment_id"],
            "case_id": payload["envelope"]["case_id"],
        }

    def get_case_with_etag(self, case_id, *, token):
        self.calls.append(("get", case_id))
        return {"id": case_id, "status": "ASSESSED", "version": 1}, '"1"'

    def submit_case(self, case_id, payload, etag, *, idempotency_key, token):
        self.calls.append(("submit", idempotency_key))
        return {"id": case_id, "version": 2, "status": "SUBMITTED"}


def test_worker_preserves_order_and_uses_event_ids_as_idempotency_keys(tmp_path: Path) -> None:
    store = prepared_store(tmp_path)
    transport = RecordingTransport()
    worker = SyncWorker(
        store=store,
        transport=transport,
        token_provider=StaticTokenProvider("token"),
        worker_id="worker-1",
    )

    for index in range(5):
        worker.run_once(now=f"2027-01-01T00:00:0{index}Z")

    assert [name for name, _ in transport.calls] == ["create", "upload", "import", "get", "submit"]
    assert all(key for _, key in transport.calls if _ != "get")
    assert store.get_case("case-1").sync_state is SyncState.SYNCED
    assert all(event.state is OutboxState.ACKED for event in store.list_outbox("case-1"))


def test_retryable_failure_remains_durable_and_auth_refresh_retries_same_event(
    tmp_path: Path,
) -> None:
    store = prepared_store(tmp_path)

    class RetryThenSuccess(RecordingTransport):
        def __init__(self) -> None:
            super().__init__()
            self.first = True

        def create_case(self, payload, *, idempotency_key, token):
            self.calls.append(("create", idempotency_key))
            if self.first:
                self.first = False
                raise ApiClientError(
                    503, {"code": "dependency_unavailable", "message": "temporary"}
                )
            return {"id": payload["id"], "version": 1}

    transport = RetryThenSuccess()
    worker = SyncWorker(
        store=store,
        transport=transport,
        token_provider=StaticTokenProvider("token"),
        worker_id="worker-1",
    )
    worker.run_once(now="2027-01-01T00:00:00Z")
    event = store.list_outbox("case-1")[0]
    assert event.state is OutboxState.RETRY_WAIT
    assert event.last_error == "temporary"

    class Refreshing(RecordingTransport):
        def __init__(self) -> None:
            super().__init__()
            self.failed = True

        def create_case(self, payload, *, idempotency_key, token):
            self.calls.append((token, idempotency_key))
            if self.failed:
                self.failed = False
                raise ApiClientError(
                    401, {"code": "authentication_required", "message": "expired"}
                )
            return {"id": payload["id"], "version": 1}

    provider = StaticTokenProvider("old", refresh_token="new")
    refreshing = Refreshing()
    worker = SyncWorker(
        store=store, transport=refreshing, token_provider=provider, worker_id="worker-1"
    )
    worker.run_once(now="2027-01-01T00:10:00Z")
    assert refreshing.calls[0][1] == refreshing.calls[1][1]
    assert refreshing.calls[1][0] == "new"


def test_permanent_failure_is_dead_lettered_without_deleting_event(tmp_path: Path) -> None:
    store = prepared_store(tmp_path)

    class Invalid(RecordingTransport):
        def create_case(self, payload, *, idempotency_key, token):
            raise ApiClientError(422, {"code": "validation_error", "message": "invalid"})

    worker = SyncWorker(
        store=store,
        transport=Invalid(),
        token_provider=StaticTokenProvider("token"),
        worker_id="worker-1",
    )
    worker.run_once(now="2027-01-01T00:00:00Z")
    event = store.list_outbox("case-1")[0]
    assert event.state is OutboxState.DEAD_LETTER
    assert event.last_error == "invalid"


def test_rate_limit_honors_bounded_retry_after_and_state_conflict_is_conservative(
    tmp_path: Path,
) -> None:
    store = prepared_store(tmp_path)

    class RateLimited(RecordingTransport):
        def create_case(self, payload, *, idempotency_key, token):
            raise ApiClientError(
                429, {"code": "rate_limited", "message": "slow down"}, headers={"retry-after": "7"}
            )

    worker = SyncWorker(
        store=store,
        transport=RateLimited(),
        token_provider=StaticTokenProvider("token"),
        worker_id="worker-1",
    )
    worker.run_once(now="2027-01-01T00:00:00Z")
    event = store.list_outbox("case-1")[0]
    assert event.state is OutboxState.RETRY_WAIT
    assert event.next_attempt_at == "2027-01-01T00:00:07Z"

    class Conflict(RecordingTransport):
        def create_case(self, payload, *, idempotency_key, token):
            raise ApiClientError(409, {"code": "conflict", "message": "stale"})

    store2 = prepared_store(tmp_path / "conflict")
    conflict_worker = SyncWorker(
        store=store2,
        transport=Conflict(),
        token_provider=StaticTokenProvider("token"),
        worker_id="worker-1",
    )
    conflict_worker.run_once(now="2027-01-01T00:00:00Z")
    assert store2.list_outbox("case-1")[0].state is OutboxState.CONFLICT


def test_failed_auth_refresh_blocks_event_without_deleting_it(tmp_path: Path) -> None:
    store = prepared_store(tmp_path)

    class Unauthorized(RecordingTransport):
        def create_case(self, payload, *, idempotency_key, token):
            raise ApiClientError(401, {"code": "authentication_required", "message": "expired"})

    worker = SyncWorker(
        store=store,
        transport=Unauthorized(),
        token_provider=StaticTokenProvider("expired"),
        worker_id="worker-1",
    )
    worker.run_once(now="2027-01-01T00:00:00Z")
    assert store.list_outbox("case-1")[0].state is OutboxState.BLOCKED_AUTH


def test_file_token_provider_reloads_rotated_demo_token(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("old", encoding="utf-8")
    provider = FileTokenProvider(token_file)
    assert provider.get_token() == "old"
    token_file.write_text("new", encoding="utf-8")
    assert provider.refresh() == "new"
