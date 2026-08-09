"""Role and facility authorization decisions."""

from __future__ import annotations

from rural_stroke_assist.server.principal import Principal


def require_case_access(principal: Principal, required_role: str, facility: str) -> None:
    if not principal.has_role(required_role) or not principal.can_access_facility(facility):
        raise PermissionError("Resource is not accessible.")
