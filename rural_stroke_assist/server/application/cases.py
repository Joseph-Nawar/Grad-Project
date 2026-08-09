"""Case application helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from rural_stroke_assist.server.api.schemas.cases import CaseResponse
from rural_stroke_assist.server.domain.states import CaseState
from rural_stroke_assist.server.errors import ApiError
from rural_stroke_assist.server.infrastructure.db.models import CaseModel
from rural_stroke_assist.server.principal import Principal


def get_case(session: Session, case_id: UUID, principal: Principal) -> CaseModel:
    case = session.get(CaseModel, case_id)
    known_role = bool(principal.roles.intersection({"collector", "clinician", "demo-admin"}))
    if case is None or not known_role or (not principal.can_access_facility(case.facility)) or (principal.has_role("collector") and case.collector_subject != principal.subject):
        raise ApiError("not_found", "Case was not found.", status_code=404)
    return case


def case_response(case: CaseModel) -> CaseResponse:
    return CaseResponse.model_validate(case)


def ensure_if_match(case: CaseModel, if_match: str | None) -> None:
    if if_match is None:
        raise ApiError("precondition_required", "If-Match is required for mutable operations.", status_code=428)
    expected = if_match.strip('"')
    if expected != str(case.version):
        raise ApiError("conflict", "The case version is stale.", details={"current_version": case.version}, status_code=409)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_state(case: CaseModel, state: CaseState) -> None:
    if case.status != state.value:
        raise ApiError("conflict", f"Case is not in {state.value} state.", status_code=409)
