"""Application workflow independent of Streamlit and storage implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from pydantic import ValidationError

from rural_stroke_assist.assessment.contracts import AssessmentResult
from rural_stroke_assist.assessment.exceptions import AssessmentError
from rural_stroke_assist.assessment.service import AssessmentService
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.cases.attachment_store import AttachmentStore
from rural_stroke_assist.cases.contracts import AttachmentKind, AuditEvent, Case, CaseStatus, ClinicianReview, utc_now
from rural_stroke_assist.cases.exceptions import ImmutableSnapshotError, InvalidTransitionError
from rural_stroke_assist.cases.repository import CaseRepository
from rural_stroke_assist.cases.serialization import assessment_input_for_storage, assessment_result_to_dict


class CaseWorkflowService:
    def __init__(self, *, repository: CaseRepository, attachment_store: AttachmentStore, assessment_service: AssessmentService) -> None:
        self.repository = repository
        self.attachment_store = attachment_store
        self.assessment_service = assessment_service

    def initialize(self) -> None:
        self.repository.initialize()

    def create_draft(self, *, collector_identity: str, facility: str, assessment_input: AssessmentInput, patient_code: str | None = None, face_bytes: bytes | None = None, audio_bytes: bytes | None = None, face_filename: str | None = None, audio_filename: str | None = None, face_media_type: str | None = None, audio_media_type: str | None = None) -> Case:
        if not collector_identity.strip() or not facility.strip():
            raise InvalidTransitionError("Collector identity and facility are required.")
        case_id = str(uuid4())
        attachments = []
        if face_bytes:
            attachments.append(self.attachment_store.save(case_id, AttachmentKind.FACE, face_bytes, filename=face_filename or "capture.jpg", media_type=face_media_type))
        if audio_bytes:
            attachments.append(self.attachment_store.save(case_id, AttachmentKind.AUDIO, audio_bytes, filename=audio_filename or "recording.wav", media_type=audio_media_type))
        payload = assessment_input.model_dump(mode="json")
        payload = assessment_input_for_storage(payload)
        if any(item.kind is AttachmentKind.FACE for item in attachments):
            payload["face_image_path"] = next(item.relative_path for item in attachments if item.kind is AttachmentKind.FACE)
        if any(item.kind is AttachmentKind.AUDIO for item in attachments):
            payload["speech_audio_path"] = next(item.relative_path for item in attachments if item.kind is AttachmentKind.AUDIO)
        now = utc_now()
        case = Case(case_id, patient_code, collector_identity, facility, payload, tuple(attachments), None, now, now, None, CaseStatus.DRAFT, audit_events=(AuditEvent("DRAFT_CREATED", now, collector_identity),))
        self.repository.save(case)
        return case

    def get(self, case_id: str) -> Case:
        return self.repository.get(case_id)

    def list_cases(self, *, statuses: tuple[CaseStatus, ...] | None = None) -> list[Case]:
        return self.repository.list(statuses=statuses)

    def assess(self, case_id: str, *, actor: str) -> Case:
        case = self.get(case_id)
        self._require(case, {CaseStatus.DRAFT})
        service_input = self._to_service_input(case)
        result = self.assessment_service.assess(service_input)
        if not isinstance(result, AssessmentResult):
            raise AssessmentError("Assessment service returned an invalid result.")
        now = utc_now()
        updated = case.with_update(status=CaseStatus.ASSESSED, assessment_result=assessment_result_to_dict(result), audit_events=case.audit_events + (AuditEvent("ASSESSED", now, actor),))
        self.repository.save(updated)
        return updated

    def submit(self, case_id: str, *, actor: str, confirmed: bool) -> Case:
        case = self.get(case_id)
        self._require(case, {CaseStatus.ASSESSED})
        if not confirmed:
            raise InvalidTransitionError("Input review confirmation is required before submission.")
        now = utc_now()
        updated = case.with_update(status=CaseStatus.SUBMITTED, submitted_at=now, audit_events=case.audit_events + (AuditEvent("SUBMITTED", now, actor),))
        self.repository.save(updated)
        return updated

    def begin_review(self, case_id: str, *, actor: str) -> Case:
        case = self.get(case_id)
        self._require(case, {CaseStatus.SUBMITTED})
        now = utc_now()
        updated = case.with_update(status=CaseStatus.IN_REVIEW, audit_events=case.audit_events + (AuditEvent("IN_REVIEW", now, actor),))
        self.repository.save(updated)
        return updated

    def review(self, case_id: str, *, clinician_name: str, medical_centre: str, agree: bool, notes: str, alternative_disposition: str | None = None) -> Case:
        case = self.get(case_id)
        self._require(case, {CaseStatus.IN_REVIEW})
        if not clinician_name.strip() or not medical_centre.strip() or not notes.strip():
            raise InvalidTransitionError("Reviewer, medical centre, and notes are required.")
        if not agree and not alternative_disposition:
            raise InvalidTransitionError("Override notes and alternative disposition are required.")
        decision = "AGREE_WITH_PROPOSED_TRIAGE_URGENCY" if agree else "OVERRIDE_PROPOSED_TRIAGE_URGENCY"
        now = utc_now()
        review = ClinicianReview(clinician_name.strip(), medical_centre.strip(), decision, notes.strip(), alternative_disposition.strip() if alternative_disposition else None, now)
        status = CaseStatus.REVIEWED_AGREED if agree else CaseStatus.REVIEWED_OVERRIDDEN
        updated = case.with_update(status=status, clinician_review=review, audit_events=case.audit_events + (AuditEvent("REVIEWED", now, clinician_name),))
        self.repository.save(updated)
        return updated

    def _to_service_input(self, case: Case) -> AssessmentInput:
        payload = dict(case.assessment_input)
        for item in case.attachments:
            if item.kind is AttachmentKind.FACE:
                payload["face_image_path"] = self.attachment_store.resolve(item)
            elif item.kind is AttachmentKind.AUDIO:
                payload["speech_audio_path"] = self.attachment_store.resolve(item)
        try:
            return AssessmentInput.model_validate(payload)
        except ValidationError as exc:
            raise InvalidTransitionError("Stored assessment input is invalid.") from exc

    @staticmethod
    def _require(case: Case, allowed: set[CaseStatus]) -> None:
        if case.status not in allowed:
            raise InvalidTransitionError(f"Cannot operate on case in {case.status.value} status.")
