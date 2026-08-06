"""Explicit JSON-safe serialization for domain snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rural_stroke_assist.assessment.contracts import AssessmentResult
from rural_stroke_assist.cases.contracts import AuditEvent, Case, CaseStatus, ClinicianReview


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def assessment_result_to_dict(result: AssessmentResult) -> dict[str, Any]:
    modalities: dict[str, Any] = {}
    for name, execution in result.modality_executions.items():
        evidence = execution.evidence
        modalities[name] = {
            "modality": evidence.modality,
            "available": evidence.available,
            "score": evidence.score,
            "score_semantics": evidence.score_semantics,
            "label": evidence.label,
            "confidence": evidence.confidence,
            "quality_status": evidence.quality_status.value,
            "quality_findings": [
                {"code": finding.code, "message": finding.message, "status": finding.status.value}
                for finding in evidence.quality_findings
            ],
            "warnings": list(evidence.warnings),
            "details": _json_safe(dict(evidence.details)),
            "provenance": evidence.provenance,
            "duration_ns": execution.duration_ns,
            "failure": None if execution.failure is None else {
                "modality": execution.failure.modality,
                "error_type": execution.failure.error_type,
                "message": execution.failure.message,
            },
        }
    return {
        "status": result.status.value,
        "modality_executions": modalities,
        "fusion": None if result.fusion is None else {
            "evidence_score": result.fusion.evidence_score,
            "risk_band": result.fusion.risk_band,
            "modality_contributions": dict(result.fusion.modality_contributions),
            "normalized_weights_used": dict(result.fusion.normalized_weights_used),
            "warnings": list(result.fusion.warnings),
            "evidence_notes": list(result.fusion.evidence_notes),
            "provenance": result.fusion.provenance,
        },
        "explanations": [{"code": item.code, "message": item.message} for item in result.explanations],
        "warnings": list(result.warnings),
        "timings": {
            "modality_duration_ns": dict(result.timings.modality_duration_ns),
            "fusion_duration_ns": result.timings.fusion_duration_ns,
            "explanation_duration_ns": result.timings.explanation_duration_ns,
            "total_duration_ns": result.timings.total_duration_ns,
        },
        "provenance": list(result.provenance),
    }


def case_to_dict(case: Case) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "patient_code": case.patient_code,
        "collector_identity": case.collector_identity,
        "facility": case.facility,
        "assessment_input": dict(case.assessment_input),
        "attachments": [{"kind": item.kind.value, "relative_path": item.relative_path, "media_type": item.media_type, "size_bytes": item.size_bytes} for item in case.attachments],
        "assessment_result": case.assessment_result,
        "created_at": case.created_at,
        "updated_at": case.updated_at,
        "submitted_at": case.submitted_at,
        "status": case.status.value,
        "clinician_review": None if case.clinician_review is None else {
            "clinician_name": case.clinician_review.clinician_name,
            "medical_centre": case.clinician_review.medical_centre,
            "decision": case.clinician_review.decision,
            "notes": case.clinician_review.notes,
            "alternative_disposition": case.clinician_review.alternative_disposition,
            "reviewed_at": case.clinician_review.reviewed_at,
        },
        "audit_events": [{"event_type": event.event_type, "at": event.at, "actor": event.actor, "detail": event.detail} for event in case.audit_events],
    }


def dumps_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads_json(value: str) -> Any:
    return json.loads(value)


def assessment_input_for_storage(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result["face_image_path"] = None
    result["speech_audio_path"] = None
    return result
