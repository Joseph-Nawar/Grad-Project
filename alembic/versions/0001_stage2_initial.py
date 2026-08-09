"""Initial Stage 2 PostgreSQL schema."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_stage2_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    uuid = postgresql.UUID(as_uuid=True)
    jsonb = postgresql.JSONB()
    timestamptz = sa.DateTime(timezone=True)
    op.create_table("cases", sa.Column("id", uuid, primary_key=True), sa.Column("facility", sa.String(128), nullable=False), sa.Column("patient_code", sa.String(128)), sa.Column("collector_subject", sa.String(255), nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("assessment_input", jsonb, nullable=False), sa.Column("version", sa.Integer, nullable=False, server_default="1"), sa.Column("created_at", timestamptz, nullable=False), sa.Column("updated_at", timestamptz, nullable=False), sa.Column("submitted_at", timestamptz), sa.Column("review_claimed_by", sa.String(255)), sa.Column("review_claimed_at", timestamptz), sa.CheckConstraint("version > 0", name="ck_cases_version_positive"))
    op.create_index("ix_cases_facility_created_at_id", "cases", ["facility", "created_at", "id"])
    op.create_index("ix_cases_status_created_at_id", "cases", ["status", "created_at", "id"])
    op.create_table("attachments", sa.Column("id", uuid, primary_key=True), sa.Column("case_id", uuid, sa.ForeignKey("cases.id", ondelete="CASCADE"), nullable=False), sa.Column("kind", sa.String(32), nullable=False), sa.Column("media_type", sa.String(128), nullable=False), sa.Column("storage_key", sa.String(512), nullable=False, unique=True), sa.Column("original_filename", sa.String(255)), sa.Column("size_bytes", sa.Integer, nullable=False), sa.Column("checksum_sha256", sa.String(64), nullable=False), sa.Column("created_at", timestamptz, nullable=False))
    op.create_index("ix_attachments_case_created_at", "attachments", ["case_id", "created_at"])
    op.create_table("assessments", sa.Column("id", uuid, primary_key=True), sa.Column("case_id", uuid, sa.ForeignKey("cases.id", ondelete="CASCADE"), nullable=False), sa.Column("client_id", uuid, nullable=False), sa.Column("actor_subject", sa.String(255), nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("request_snapshot", jsonb, nullable=False), sa.Column("result_snapshot", jsonb, nullable=False), sa.Column("created_at", timestamptz, nullable=False), sa.UniqueConstraint("case_id", "client_id", name="uq_assessments_case_client"))
    op.create_index("ix_assessments_case_created_at", "assessments", ["case_id", "created_at"])
    op.create_table("submissions", sa.Column("id", uuid, primary_key=True), sa.Column("case_id", uuid, sa.ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False), sa.Column("assessment_id", uuid, sa.ForeignKey("assessments.id", ondelete="RESTRICT"), nullable=False), sa.Column("snapshot", jsonb, nullable=False), sa.Column("snapshot_sha256", sa.String(64), nullable=False), sa.Column("submitted_by", sa.String(255), nullable=False), sa.Column("created_at", timestamptz, nullable=False), sa.UniqueConstraint("case_id", name="uq_submissions_case"))
    op.create_table("reviews", sa.Column("id", uuid, primary_key=True), sa.Column("case_id", uuid, sa.ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False), sa.Column("submission_id", uuid, sa.ForeignKey("submissions.id", ondelete="RESTRICT"), nullable=False), sa.Column("reviewer_subject", sa.String(255), nullable=False), sa.Column("agree", sa.Boolean, nullable=False), sa.Column("notes", sa.Text, nullable=False), sa.Column("alternative_disposition", sa.String(512)), sa.Column("created_at", timestamptz, nullable=False))
    op.create_index("ix_reviews_case_created_at", "reviews", ["case_id", "created_at"])
    op.create_table("idempotency_records", sa.Column("id", uuid, primary_key=True), sa.Column("actor_subject", sa.String(255), nullable=False), sa.Column("operation", sa.String(128), nullable=False), sa.Column("key", sa.String(255), nullable=False), sa.Column("request_hash", sa.String(64), nullable=False), sa.Column("state", sa.String(32), nullable=False), sa.Column("response_snapshot", jsonb), sa.Column("resource_id", uuid), sa.Column("created_at", timestamptz, nullable=False), sa.Column("updated_at", timestamptz, nullable=False), sa.Column("expires_at", timestamptz, nullable=False), sa.UniqueConstraint("actor_subject", "operation", "key", name="uq_idempotency_actor_operation_key"))
    op.create_index("ix_idempotency_expires_at", "idempotency_records", ["expires_at"])


def downgrade() -> None:
    op.drop_table("idempotency_records")
    op.drop_index("ix_reviews_case_created_at", table_name="reviews")
    op.drop_table("reviews")
    op.drop_table("submissions")
    op.drop_index("ix_assessments_case_created_at", table_name="assessments")
    op.drop_table("assessments")
    op.drop_index("ix_attachments_case_created_at", table_name="attachments")
    op.drop_table("attachments")
    op.drop_index("ix_cases_status_created_at_id", table_name="cases")
    op.drop_index("ix_cases_facility_created_at_id", table_name="cases")
    op.drop_table("cases")
