"""Typed central PostgreSQL mappings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from rural_stroke_assist.server.infrastructure.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CaseModel(Base):
    __tablename__ = "cases"
    __table_args__ = (CheckConstraint("version > 0", name="ck_cases_version_positive"), CheckConstraint("status IN ('DRAFT', 'ASSESSED', 'SUBMITTED', 'IN_REVIEW', 'REVIEWED_AGREED', 'REVIEWED_OVERRIDDEN')", name="ck_cases_status"), Index("ix_cases_facility_created_at_id", "facility", "created_at", "id"), Index("ix_cases_status_created_at_id", "status", "created_at", "id"))
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    facility: Mapped[str] = mapped_column(String(128), nullable=False)
    patient_code: Mapped[str | None] = mapped_column(String(128))
    collector_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="DRAFT")
    assessment_input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_claimed_by: Mapped[str | None] = mapped_column(String(255))
    review_claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attachments: Mapped[list["AttachmentModel"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    assessments: Mapped[list["AssessmentModel"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    submission: Mapped["SubmissionModel | None"] = relationship(back_populates="case", uselist=False)
    reviews: Mapped[list["ReviewModel"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    __mapper_args__ = {"version_id_col": version}


class AttachmentModel(Base):
    __tablename__ = "attachments"
    __table_args__ = (CheckConstraint("size_bytes > 0", name="ck_attachments_size_positive"), Index("ix_attachments_case_created_at", "case_id", "created_at"),)
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    media_type: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    case: Mapped[CaseModel] = relationship(back_populates="attachments")


class AssessmentModel(Base):
    __tablename__ = "assessments"
    __table_args__ = (UniqueConstraint("case_id", "client_id", name="uq_assessments_case_client"), Index("ix_assessments_case_created_at", "case_id", "created_at"))
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    case_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    client_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    actor_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    request_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    result_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    case: Mapped[CaseModel] = relationship(back_populates="assessments")


class SubmissionModel(Base):
    __tablename__ = "submissions"
    __table_args__ = (UniqueConstraint("case_id", name="uq_submissions_case"),)
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False)
    assessment_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("assessments.id", ondelete="RESTRICT"), nullable=False)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    snapshot_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    submitted_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    case: Mapped[CaseModel] = relationship(back_populates="submission")


class ReviewModel(Base):
    __tablename__ = "reviews"
    __table_args__ = (Index("ix_reviews_case_created_at", "case_id", "created_at"),)
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False)
    submission_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("submissions.id", ondelete="RESTRICT"), nullable=False)
    reviewer_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    agree: Mapped[bool] = mapped_column(Boolean, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False)
    alternative_disposition: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    case: Mapped[CaseModel] = relationship(back_populates="reviews")


class IdempotencyModel(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (CheckConstraint("state IN ('IN_PROGRESS', 'COMPLETED')", name="ck_idempotency_state"), UniqueConstraint("actor_subject", "operation", "key", name="uq_idempotency_actor_operation_key"), Index("ix_idempotency_expires_at", "expires_at"))
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    actor_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    operation: Mapped[str] = mapped_column(String(128), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="IN_PROGRESS")
    response_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    resource_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
