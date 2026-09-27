"""FastAPI application factory and Stage 2 lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.engine import Engine

from rural_stroke_assist.server.api.schemas.common import (
    ErrorResponse,
    HealthResponse,
    VersionResponse,
)
from rural_stroke_assist.server.config import ServerSettings
from rural_stroke_assist.server.correlation import CorrelationIdMiddleware
from rural_stroke_assist.server.errors import ApiError, ErrorEnvelope, error_response
from rural_stroke_assist.server.infrastructure.auth.cognito import CognitoTokenVerifier
from rural_stroke_assist.server.infrastructure.auth.local_jwt import LocalJwtVerifier
from rural_stroke_assist.server.infrastructure.db.session import create_db_engine, session_factory
from rural_stroke_assist.server.infrastructure.storage.filesystem import FilesystemAttachmentStore
from rural_stroke_assist.server.infrastructure.storage.s3 import S3AttachmentStore


def create_attachment_store(settings: ServerSettings, *, client: Any | None = None) -> Any:
    if settings.storage_backend == "filesystem":
        return FilesystemAttachmentStore(
            settings.storage_root, max_bytes=settings.max_attachment_bytes
        )
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("RURALSTROKE_S3_BUCKET is required when S3 storage is enabled.")
        return S3AttachmentStore(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            client=client,
            max_bytes=settings.max_attachment_bytes,
        )
    raise RuntimeError(f"Unsupported attachment storage backend: {settings.storage_backend}")


def create_token_verifier(settings: ServerSettings, *, fetch_json: Any | None = None) -> Any:
    if settings.auth_backend == "local":
        return LocalJwtVerifier(
            secret=settings.jwt_secret, issuer=settings.jwt_issuer, audience=settings.jwt_audience
        )
    if settings.auth_backend == "cognito":
        if not settings.cognito_issuer or not settings.cognito_client_id:
            raise RuntimeError(
                "Cognito issuer and client ID are required when Cognito auth is enabled."
            )
        kwargs: dict[str, Any] = {
            "issuer": settings.cognito_issuer,
            "client_ids": settings.cognito_client_ids or (settings.cognito_client_id,),
        }
        if settings.cognito_jwks_uri:
            kwargs["jwks_uri"] = settings.cognito_jwks_uri
        if fetch_json is not None:
            kwargs["fetch_json"] = fetch_json
        return CognitoTokenVerifier(**kwargs)
    raise RuntimeError(f"Unsupported authentication backend: {settings.auth_backend}")


def _validation_response(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = {
        "fields": [
            {"location": list(error.get("loc", [])), "type": error.get("type", "validation_error")}
            for error in exc.errors()
        ]
    }
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    content = ErrorEnvelope(
        code="validation_error",
        message="Request validation failed.",
        details=details,
        correlation_id=correlation_id,
    ).model_dump()
    return JSONResponse(status_code=422, content=content)


def _unexpected_response(request: Request, _exc: Exception) -> JSONResponse:
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    content = ErrorEnvelope(
        code="internal_error",
        message="An internal server error occurred.",
        details={},
        correlation_id=correlation_id,
    ).model_dump()
    return JSONResponse(status_code=500, content=content)


def create_app(
    *,
    settings: ServerSettings | None = None,
    database_url: str | None = None,
    storage_root: str | Path | None = None,
    assessment_service: Any | None = None,
    token_verifier: Any | None = None,
    initialize_resources: bool = True,
) -> FastAPI:
    app_settings = settings or ServerSettings.from_environment()
    if database_url is not None:
        app_settings = ServerSettings(**{**app_settings.__dict__, "database_url": database_url})
    if storage_root is not None:
        app_settings = ServerSettings(
            **{**app_settings.__dict__, "storage_root": Path(storage_root)}
        )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        engine: Engine | None = None
        if initialize_resources:
            engine = create_db_engine(app_settings.database_url)
            app_settings.storage_root.mkdir(parents=True, exist_ok=True)
            application.state.session_factory = session_factory(engine)
            application.state.attachment_store = create_attachment_store(app_settings)
            if assessment_service is None:
                from rural_stroke_assist.assessment.factory import (
                    create_default_assessment_service,
                )

                application.state.assessment_service = create_default_assessment_service()
            else:
                application.state.assessment_service = assessment_service
            application.state.engine = engine
            application.state.settings = app_settings
            application.state.token_verifier = token_verifier or create_token_verifier(
                app_settings
            )
        try:
            yield
        finally:
            if engine is not None:
                engine.dispose()

    app = FastAPI(title="RuralStroke-Triage API", version=app_settings.version, lifespan=lifespan)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_exception_handler(ApiError, error_response)
    app.add_exception_handler(RequestValidationError, _validation_response)
    app.add_exception_handler(Exception, _unexpected_response)

    from rural_stroke_assist.server.api.routes.assessments import router as assessment_router
    from rural_stroke_assist.server.api.routes.attachments import router as attachment_router
    from rural_stroke_assist.server.api.routes.cases import router as case_router
    from rural_stroke_assist.server.api.routes.identity import router as identity_router
    from rural_stroke_assist.server.api.routes.reviews import router as review_router

    app.include_router(identity_router)
    app.include_router(attachment_router)
    app.include_router(assessment_router)
    app.include_router(case_router)
    app.include_router(review_router)

    @app.get(
        "/health/live", operation_id="health_live", response_model=HealthResponse, tags=["health"]
    )
    def health_live() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready",
        operation_id="health_ready",
        response_model=HealthResponse,
        tags=["health"],
    )
    def health_ready(request: Request) -> HealthResponse:
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            return HealthResponse(status="ok")
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:
            raise ApiError(
                "dependency_unavailable", "Required service is not ready.", status_code=503
            ) from exc
        return HealthResponse(status="ok")

    @app.get("/version", operation_id="version", response_model=VersionResponse, tags=["health"])
    def version() -> VersionResponse:
        return VersionResponse(version=app_settings.version)

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        document = get_openapi(title=app.title, version=app.version, routes=app.routes)
        error_ref = {"$ref": "#/components/schemas/ErrorResponse"}
        common_errors = {
            "401": "Authentication required.",
            "403": "Forbidden.",
            "404": "Not found.",
            "409": "Conflict.",
            "422": "Request validation failed.",
            "500": "Internal server error.",
            "503": "Dependency unavailable.",
        }
        operation_errors = {
            "case_create": {"400": "Invalid case request."},
            "assessment_create": {
                "400": "Invalid assessment request.",
                "503": "Assessment unavailable.",
            },
            "assessment_import": {"422": "Imported assessment envelope or provenance is invalid."},
            "attachment_upload": {
                "409": "Stable attachment identity conflicts with existing content.",
                "413": "Attachment exceeds the configured size limit.",
                "415": "Unsupported attachment media type.",
            },
            "attachment_read": {"500": "Attachment storage failure."},
            "case_update": {"428": "If-Match is required."},
            "case_submit": {"400": "Submission is invalid.", "428": "If-Match is required."},
            "review_claim": {"400": "Review claim is invalid."},
            "review_create": {"400": "Review decision is invalid."},
            "health_ready": {"503": "Required services are not ready."},
        }
        required_idempotency = {
            "case_create",
            "assessment_create",
            "assessment_import",
            "case_submit",
            "review_claim",
            "review_create",
        }
        required_if_match = {"case_update", "case_submit"}
        for path, operations in document["paths"].items():
            for operation in operations.values():
                if not isinstance(operation, dict) or "operationId" not in operation:
                    continue
                operation_id = operation["operationId"]
                operation.setdefault("responses", {})
                statuses = dict(common_errors) if path.startswith("/api/v1/") else {}
                statuses.update(operation_errors.get(operation_id, {}))
                for status, description in statuses.items():
                    operation["responses"][status] = {
                        "description": description,
                        "content": {"application/json": {"schema": error_ref}},
                    }
                for parameter in operation.get("parameters", []):
                    if (
                        parameter.get("name") == "Idempotency-Key"
                        and operation_id in required_idempotency
                    ):
                        parameter["required"] = True
                    if parameter.get("name") == "If-Match" and operation_id in required_if_match:
                        parameter["required"] = True
                        parameter["schema"] = {"type": "string"}
        schemas = document.setdefault("components", {}).setdefault("schemas", {})
        schemas.pop("HTTPValidationError", None)
        schemas.pop("ValidationError", None)
        schemas["ErrorResponse"] = ErrorResponse.model_json_schema()
        app.openapi_schema = document
        return document

    app.openapi = custom_openapi
    return app
