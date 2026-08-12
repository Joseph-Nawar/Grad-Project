"""Authentication provider boundary for local demo and future Cognito tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class TokenRefreshError(RuntimeError):
    """Raised when a provider cannot refresh an access token."""


class TokenProvider(Protocol):
    def get_token(self) -> str | None: ...
    def refresh(self) -> str: ...


@dataclass
class StaticTokenProvider:
    token: str | None
    refresh_token: str | None = None

    def get_token(self) -> str | None:
        return self.token

    def refresh(self) -> str:
        if not self.refresh_token:
            raise TokenRefreshError("No refresh token is configured.")
        self.token = self.refresh_token
        return self.token


@dataclass
class FileTokenProvider:
    """Local-demo provider that reloads a rotated token without exposing it."""

    path: Path

    def get_token(self) -> str | None:
        try:
            value = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    def refresh(self) -> str:
        token = self.get_token()
        if not token:
            raise TokenRefreshError("The configured token file is unavailable.")
        return token
