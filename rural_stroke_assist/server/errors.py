"""Safe, stable API errors."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any]
    correlation_id: str


class ApiError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None, correlation_id: str | None = None, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.correlation_id = correlation_id
        self.status_code = status_code

    def to_dict(self, correlation_id: str | None = None) -> dict[str, Any]:
        return ErrorEnvelope(
            code=self.code,
            message=self.message,
            details=self.details,
            correlation_id=correlation_id or self.correlation_id or "unknown",
        ).model_dump()


def error_response(request: Request, error: ApiError) -> JSONResponse:
    correlation_id = getattr(request.state, "correlation_id", None)
    return JSONResponse(status_code=error.status_code, content=error.to_dict(correlation_id))
