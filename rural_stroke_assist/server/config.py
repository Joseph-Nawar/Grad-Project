"""Environment-backed server configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerSettings:
    database_url: str | None = None
    storage_root: Path = Path("runtime_data/server_attachments")
    jwt_secret: str = "stage2-local-development-secret-change-me"
    jwt_issuer: str = "ruralstroke-local"
    jwt_audience: str = "ruralstroke-api"
    token_ttl_seconds: int = 3600
    version: str = "0.0.1-stage2"
    max_attachment_bytes: int = 25 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> "ServerSettings":
        database_url = os.getenv("RURALSTROKE_DATABASE_URL")
        if database_url is None and os.getenv("RURALSTROKE_ENV", "").lower() == "test":
            database_url = os.getenv("RURALSTROKE_TEST_DATABASE_URL")
        return cls(
            database_url=database_url,
            storage_root=Path(os.getenv("RURALSTROKE_STORAGE_ROOT", str(cls.storage_root))),
            jwt_secret=os.getenv("RURALSTROKE_JWT_SECRET", cls.jwt_secret),
            jwt_issuer=os.getenv("RURALSTROKE_JWT_ISSUER", cls.jwt_issuer),
            jwt_audience=os.getenv("RURALSTROKE_JWT_AUDIENCE", cls.jwt_audience),
            token_ttl_seconds=int(os.getenv("RURALSTROKE_TOKEN_TTL_SECONDS", str(cls.token_ttl_seconds))),
            version=os.getenv("RURALSTROKE_API_VERSION", cls.version),
            max_attachment_bytes=int(os.getenv("RURALSTROKE_MAX_ATTACHMENT_BYTES", str(cls.max_attachment_bytes))),
        )
