"""Add integrity checks required by the Stage 2 contract."""

from alembic import op

revision = "0002_integrity"
down_revision = "0001_stage2_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_cases_status",
        "cases",
        "status IN ('DRAFT', 'ASSESSED', 'SUBMITTED', 'IN_REVIEW', 'REVIEWED_AGREED', 'REVIEWED_OVERRIDDEN')",
    )
    op.create_check_constraint("ck_attachments_size_positive", "attachments", "size_bytes > 0")
    op.create_check_constraint("ck_idempotency_state", "idempotency_records", "state IN ('IN_PROGRESS', 'COMPLETED')")


def downgrade() -> None:
    op.drop_constraint("ck_idempotency_state", "idempotency_records", type_="check")
    op.drop_constraint("ck_attachments_size_positive", "attachments", type_="check")
    op.drop_constraint("ck_cases_status", "cases", type_="check")
