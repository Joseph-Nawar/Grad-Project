from __future__ import annotations

from pathlib import Path
import pytest

from rural_stroke_assist.server.app import create_app
from rural_stroke_assist.server.errors import ApiError, ErrorEnvelope
from rural_stroke_assist.server.principal import Principal


def test_principal_is_immutable_and_normalizes_claims() -> None:
    principal = Principal(subject="collector-1", roles=frozenset({"collector"}), facilities=frozenset({"facility-a"}))
    assert principal.subject == "collector-1"
    assert principal.has_role("collector")
    assert principal.can_access_facility("facility-a")
    with pytest.raises((AttributeError, TypeError)):
        principal.subject = "other"  # type: ignore[misc]


def test_api_error_has_stable_safe_envelope() -> None:
    error = ApiError(code="conflict", message="Conflict", details={"field": "version"}, correlation_id="corr-1")
    assert ErrorEnvelope.model_validate(error.to_dict()).model_dump() == {
        "code": "conflict",
        "message": "Conflict",
        "details": {"field": "version"},
        "correlation_id": "corr-1",
    }


def test_app_factory_exposes_unversioned_operational_routes() -> None:
    app = create_app(database_url="postgresql+psycopg://user:pass@localhost/db", initialize_resources=False)
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert {"/health/live", "/health/ready", "/version"}.issubset(paths)
    assert not any(path in {"/api/v1/health/live", "/api/v1/health/ready", "/api/v1/version"} for path in paths)


def test_application_startup_does_not_create_database_schema() -> None:
    source = (Path(__file__).resolve().parents[2] / "rural_stroke_assist" / "server" / "app.py").read_text(encoding="utf-8")
    assert "create_all(" not in source
