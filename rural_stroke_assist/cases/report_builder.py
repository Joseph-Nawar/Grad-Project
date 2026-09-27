"""Deterministic report view model and download renderers."""

from __future__ import annotations

import html
import json
from typing import Any

from rural_stroke_assist.cases.contracts import Case


def build_report_view(case: Case) -> dict[str, Any]:
    return {
        "case": {"case_id": case.case_id, "patient_code": case.patient_code, "collector_identity": case.collector_identity, "facility": case.facility, "created_at": case.created_at, "status": case.status.value},
        "assessment_input": case.assessment_input,
        "attachments": [{"kind": item.kind.value, "media_type": item.media_type, "size_bytes": item.size_bytes} for item in case.attachments],
        "assessment_result": case.assessment_result,
        "clinician_review": None if case.clinician_review is None else {"clinician_name": case.clinician_review.clinician_name, "medical_centre": case.clinician_review.medical_centre, "decision": case.clinician_review.decision, "notes": case.clinician_review.notes, "alternative_disposition": case.clinician_review.alternative_disposition, "reviewed_at": case.clinician_review.reviewed_at},
        "non_diagnostic_statement": "This is a research screening and triage-support result, not a clinical diagnosis or calibrated probability.",
    }


def render_json(case: Case) -> str:
    return json.dumps(build_report_view(case), ensure_ascii=False, indent=2, sort_keys=True)


def render_markdown(case: Case) -> str:
    view = build_report_view(case)
    result = view["assessment_result"] or {}
    fusion = result.get("fusion") or {}
    lines = ["# RuralStroke-Triage Case Report", "", f"- Case ID: `{case.case_id}`", f"- Status: `{case.status.value}`", f"- Facility: {case.facility}", "", "## Assessment", f"- Evidence band: **{fusion.get('risk_band', 'INSUFFICIENT_EVIDENCE')}**", f"- Evidence score: `{fusion.get('evidence_score', 'n/a')}`", "", "## Technical result", "```json", json.dumps(result, indent=2, sort_keys=True), "```", "", "## Safety", view["non_diagnostic_statement"]]
    if case.clinician_review:
        lines.extend(["", "## Clinician review", f"- Decision: {case.clinician_review.decision}", f"- Reviewer: {case.clinician_review.clinician_name}", f"- Notes: {case.clinician_review.notes}"])
    return "\n".join(lines)


def render_html(case: Case) -> str:
    return "<html><body><pre>" + html.escape(render_markdown(case)) + "</pre></body></html>"
