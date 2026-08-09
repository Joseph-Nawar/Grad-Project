from __future__ import annotations

from rural_stroke_assist.server.infrastructure.db.models import (
    AssessmentModel,
    AttachmentModel,
    CaseModel,
    IdempotencyModel,
    ReviewModel,
    SubmissionModel,
)


def test_stage2_models_have_required_tables_and_constraints() -> None:
    tables = {model.__tablename__ for model in (CaseModel, AttachmentModel, AssessmentModel, SubmissionModel, ReviewModel, IdempotencyModel)}
    assert tables == {"cases", "attachments", "assessments", "submissions", "reviews", "idempotency_records"}
    assert CaseModel.__table__.c.version is not None
    assert SubmissionModel.__table__.c.snapshot_sha256 is not None
    assert ReviewModel.__table__.c.submission_id is not None
    assert any(constraint.name == "uq_idempotency_actor_operation_key" for constraint in IdempotencyModel.__table__.constraints)
    assert any(constraint.name == "ck_cases_version_positive" for constraint in CaseModel.__table__.constraints)
