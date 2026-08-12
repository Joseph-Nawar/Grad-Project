from __future__ import annotations

from datetime import datetime, timezone
import io
from types import SimpleNamespace
from uuid import UUID

from fastapi import Response, UploadFile
import pytest
from starlette.datastructures import Headers

from rural_stroke_assist.server.api.routes import assessments as assessment_routes
from rural_stroke_assist.server.api.schemas.assessments import AssessmentImportRequest
from rural_stroke_assist.server.application.attachments import attachment_request_fingerprint
from rural_stroke_assist.server.domain.hashing import canonical_sha256
from rural_stroke_assist.server.errors import ApiError
from rural_stroke_assist.server.infrastructure.db.models import AssessmentModel, CaseModel
from rural_stroke_assist.server.infrastructure.storage.filesystem import FilesystemAttachmentStore
from rural_stroke_assist.server.principal import Principal


def valid_envelope() -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "00000000-0000-0000-0000-000000000001",
        "assessment_id": "00000000-0000-0000-0000-000000000002",
        "request": {"session_id": "s1", "attachments": []},
        "result": {"status": "complete"},
        "provenance": {
            "schema_version": 1,
            "model_ids": {"face": "canonical-face-v1", "acute_symptoms": "canonical-acute-v1"},
            "fusion_id": "canonical-fusion-v1",
        },
    }


def test_import_contract_accepts_path_free_envelope_and_rejects_unapproved_provenance() -> None:
    request = AssessmentImportRequest(envelope=valid_envelope(), assessment_hash="a" * 64)
    assert request.envelope["case_id"] == "00000000-0000-0000-0000-000000000001"

    invalid = valid_envelope()
    invalid["provenance"] = {
        "schema_version": 1,
        "model_ids": {"face": "unapproved"},
        "fusion_id": "canonical-fusion-v1",
    }
    with pytest.raises(ValueError, match="provenance"):
        AssessmentImportRequest(envelope=invalid, assessment_hash="a" * 64)


def test_attachment_store_accepts_optional_stable_attachment_uuid(tmp_path) -> None:
    store = FilesystemAttachmentStore(tmp_path)
    attachment_id = UUID(int=3)

    stored = store.save(
        "case",
        "face",
        io.BytesIO(b"image"),
        filename="capture.jpg",
        media_type="image/jpeg",
        attachment_id=attachment_id,
    )

    assert stored.id == attachment_id
    assert stored.storage_key.endswith(f"/{attachment_id}.jpg")


def test_import_route_never_invokes_server_inference(monkeypatch) -> None:
    envelope = valid_envelope()
    request = AssessmentImportRequest(
        envelope=envelope, assessment_hash=canonical_sha256(envelope)
    )

    class FakeSession:
        def __init__(self) -> None:
            self.added = None

        def get(self, model, identifier):
            assert model is AssessmentModel
            return None

        def add(self, row):
            self.added = row

        def flush(self):
            assert self.added is not None

        def commit(self):
            return None

    class Case:
        facility = "post"
        collector_subject = "collector"
        status = "DRAFT"

    session = FakeSession()
    monkeypatch.setattr(assessment_routes, "get_case", lambda *_args, **_kwargs: Case())
    monkeypatch.setattr(assessment_routes, "begin_idempotency", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(assessment_routes, "complete_idempotency", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        assessment_routes,
        "_json_assessment",
        lambda _row: {
            "id": "assessment",
            "case_id": envelope["case_id"],
            "status": "complete",
            "request_snapshot": {},
            "result_snapshot": {},
            "created_at": "now",
        },
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("server inference must not run during assessment import")

    monkeypatch.setattr(
        "rural_stroke_assist.assessment.service.AssessmentService.assess", fail_if_called
    )
    result = assessment_routes.import_assessment(
        request, "event-1", session, Principal("collector", ["collector"], ["post"])
    )

    assert result["case_id"] == envelope["case_id"]


def test_attachment_fingerprint_uses_logical_values_not_multipart_boundaries() -> None:
    first = attachment_request_fingerprint(
        case_id="case",
        attachment_id="attachment",
        kind="face",
        media_type="image/jpeg",
        size_bytes=3,
        checksum_sha256="a" * 64,
    )
    second = attachment_request_fingerprint(
        case_id="case",
        attachment_id="attachment",
        kind="face",
        media_type="image/jpeg",
        size_bytes=3,
        checksum_sha256="a" * 64,
    )

    assert first == second
    assert "multipart" not in first


class ImportSession:
    def __init__(self, existing: object | None = None) -> None:
        self.existing = existing
        self.added = None

    def get(self, model, identifier):
        if model is AssessmentModel:
            return self.existing
        return None

    def add(self, row):
        self.added = row

    def flush(self):
        return None

    def commit(self):
        return None


def import_case() -> SimpleNamespace:
    return SimpleNamespace(facility="post", collector_subject="collector", status="DRAFT")


def prepare_import(monkeypatch, *, session: ImportSession | None = None):
    envelope = valid_envelope()
    request = AssessmentImportRequest(
        envelope=envelope, assessment_hash=canonical_sha256(envelope)
    )
    fake_session = session or ImportSession()
    monkeypatch.setattr(assessment_routes, "get_case", lambda *_args, **_kwargs: import_case())
    monkeypatch.setattr(assessment_routes, "begin_idempotency", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(assessment_routes, "complete_idempotency", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        assessment_routes,
        "_json_assessment",
        lambda _row: {
            "id": request.envelope["assessment_id"],
            "case_id": request.envelope["case_id"],
            "status": "complete",
            "request_snapshot": {},
            "result_snapshot": {},
            "created_at": "now",
        },
    )
    return request, fake_session


def test_import_rejects_clinician_and_out_of_scope_collector(monkeypatch) -> None:
    request, session = prepare_import(monkeypatch)
    with pytest.raises(ApiError, match="Collector ownership") as clinician_error:
        assessment_routes.import_assessment(
            request, "event-clinician", session, Principal("clinician", ["clinician"], ["post"])
        )
    assert clinician_error.value.status_code == 403

    def scoped_get_case(fake_session, case_id, principal):
        case = SimpleNamespace(facility="post", collector_subject="collector")
        if not principal.can_access_facility(case.facility):
            raise ApiError("not_found", "Case was not found.", status_code=404)
        return case

    monkeypatch.setattr(assessment_routes, "get_case", scoped_get_case)
    with pytest.raises(ApiError) as scope_error:
        assessment_routes.import_assessment(
            request,
            "event-scope",
            session,
            Principal("collector", ["collector"], ["other-facility"]),
        )
    assert scope_error.value.code == "not_found"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("fusion_id", "unknown-fusion"),
        ("model_ids", {"face": "unknown-model"}),
    ],
)
def test_import_rejects_unknown_schema_model_and_fusion_identifiers(
    field: str, value: object
) -> None:
    envelope = valid_envelope()
    if field == "schema_version":
        envelope[field] = value
    else:
        provenance = dict(envelope["provenance"])
        provenance[field] = value
        envelope["provenance"] = provenance
    with pytest.raises(ValueError, match="approved"):
        AssessmentImportRequest(envelope=envelope, assessment_hash="a" * 64)


def test_import_rejects_tampered_hash_and_case_identity(monkeypatch) -> None:
    request, session = prepare_import(monkeypatch)
    tampered = request.model_copy(update={"assessment_hash": "b" * 64})
    with pytest.raises(ApiError, match="hash") as tampered_error:
        assessment_routes.import_assessment(
            tampered, "event-tampered", session, Principal("collector", ["collector"], ["post"])
        )
    assert tampered_error.value.status_code == 422

    other = valid_envelope()
    other["case_id"] = "00000000-0000-0000-0000-000000000099"
    other_request = AssessmentImportRequest(
        envelope=other, assessment_hash=canonical_sha256(other)
    )
    monkeypatch.setattr(
        assessment_routes,
        "get_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ApiError("not_found", "Case was not found.", status_code=404)
        ),
    )
    with pytest.raises(ApiError) as identity_error:
        assessment_routes.import_assessment(
            other_request,
            "event-identity",
            session,
            Principal("collector", ["collector"], ["post"]),
        )
    assert identity_error.value.code == "not_found"


def test_import_rejects_existing_assessment_uuid_with_different_envelope(monkeypatch) -> None:
    envelope = valid_envelope()
    existing = SimpleNamespace(
        id=UUID(envelope["assessment_id"]),
        case_id=UUID(envelope["case_id"]),
        request_snapshot={"envelope": envelope, "assessment_hash": canonical_sha256(envelope)},
        result_snapshot=envelope["result"],
        status="complete",
        created_at=datetime.now(timezone.utc),
    )
    session = ImportSession(existing=existing)
    request, _ = prepare_import(monkeypatch, session=session)
    changed = dict(envelope)
    changed["result"] = {"status": "different"}
    changed_request = AssessmentImportRequest(
        envelope=changed, assessment_hash=canonical_sha256(changed)
    )
    with pytest.raises(ApiError, match="different snapshot") as conflict_error:
        assessment_routes.import_assessment(
            changed_request,
            "event-existing",
            session,
            Principal("collector", ["collector"], ["post"]),
        )
    assert conflict_error.value.status_code == 409


def test_stable_case_replay_mismatch_is_conflict(monkeypatch) -> None:
    from rural_stroke_assist.server.api.routes import cases as case_routes
    from rural_stroke_assist.server.api.schemas.cases import CaseCreateRequest

    case_id = UUID("00000000-0000-0000-0000-000000000001")
    existing = SimpleNamespace(
        id=case_id,
        collector_subject="collector",
        facility="other",
        patient_code=None,
        assessment_input={},
        version=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        submitted_at=None,
        assessments=[],
        attachments=[],
    )
    session = ImportSession()
    session.get = lambda model, identifier: existing if model is CaseModel else None
    monkeypatch.setattr(case_routes, "begin_idempotency", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(case_routes, "complete_idempotency", lambda *_args, **_kwargs: None)
    request = CaseCreateRequest(id=case_id, facility="post")
    with pytest.raises(ApiError, match="different case"):
        case_routes.create_case(
            request,
            Response(),
            "event-case",
            session,
            Principal("collector", ["collector"], ["post"]),
        )


def test_stable_attachment_replay_mismatch_is_conflict(monkeypatch) -> None:
    from rural_stroke_assist.server.api.routes import attachments as attachment_routes

    case_id = UUID("00000000-0000-0000-0000-000000000001")
    attachment_id = UUID("00000000-0000-0000-0000-000000000003")
    existing = SimpleNamespace(
        id=attachment_id,
        case_id=case_id,
        kind="face",
        media_type="image/jpeg",
        size_bytes=5,
        checksum_sha256="a" * 64,
    )
    session = ImportSession(existing=existing)
    session.get = lambda model, identifier: existing if identifier == attachment_id else None
    monkeypatch.setattr(attachment_routes, "get_case", lambda *_args, **_kwargs: import_case())
    monkeypatch.setattr(attachment_routes, "begin_idempotency", lambda *_args, **_kwargs: object())
    upload = UploadFile(
        file=io.BytesIO(b"other"),
        filename="capture.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(attachment_store=object()))
    )
    with pytest.raises(ApiError, match="different content"):
        attachment_routes.upload_attachment(
            request,
            case_id,
            "face",
            upload,
            attachment_id,
            "event-attachment",
            session,
            Principal("collector", ["collector"], ["post"]),
        )
