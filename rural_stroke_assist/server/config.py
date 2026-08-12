"""Environment-backed server configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerSettings:
    database_url: str | None = None
    storage_root: Path = Path("runtime_data/server_attachments")
    storage_backend: str = "filesystem"
    s3_bucket: str | None = None
    s3_region: str = "eu-central-1"
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    auth_backend: str = "local"
    cognito_issuer: str | None = None
    cognito_client_id: str | None = None
    cognito_client_ids: tuple[str, ...] = ()
    cognito_jwks_uri: str | None = None
    jwt_secret: str = "stage2-local-development-secret-change-me"
    jwt_issuer: str = "ruralstroke-local"
    jwt_audience: str = "ruralstroke-api"
    token_ttl_seconds: int = 3600
    version: str = "0.0.1-stage2"
    max_attachment_bytes: int = 25 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> "ServerSettings":
        database_url = os.getenv("RURALSTROKE_DATABASE_URL")
        database_url_file = os.getenv("RURALSTROKE_DATABASE_URL_FILE")
        if database_url_file:
            database_url = Path(database_url_file).read_text(encoding="utf-8").strip()
        if database_url is None and os.getenv("RURALSTROKE_ENV", "").lower() == "test":
            database_url = os.getenv("RURALSTROKE_TEST_DATABASE_URL")
        jwt_secret = os.getenv("RURALSTROKE_JWT_SECRET", cls.jwt_secret)
        jwt_secret_file = os.getenv("RURALSTROKE_JWT_SECRET_FILE")
        if jwt_secret_file:
            jwt_secret = Path(jwt_secret_file).read_text(encoding="utf-8").strip()
        secret_access_key = os.getenv("RURALSTROKE_S3_SECRET_ACCESS_KEY")
        secret_file = os.getenv("RURALSTROKE_S3_SECRET_ACCESS_KEY_FILE")
        if secret_file:
            secret_access_key = Path(secret_file).read_text(encoding="utf-8").strip()
        access_key_id = os.getenv("RURALSTROKE_S3_ACCESS_KEY_ID")
        access_key_file = os.getenv("RURALSTROKE_S3_ACCESS_KEY_ID_FILE")
        if access_key_file:
            access_key_id = Path(access_key_file).read_text(encoding="utf-8").strip()
        client_ids = tuple(item.strip() for item in os.getenv("RURALSTROKE_COGNITO_CLIENT_IDS", "").split(",") if item.strip())
        client_id = os.getenv("RURALSTROKE_COGNITO_CLIENT_ID")
        if not client_ids and client_id:
            client_ids = (client_id,)
        return cls(
            database_url=database_url,
            storage_root=Path(os.getenv("RURALSTROKE_STORAGE_ROOT", str(cls.storage_root))),
            storage_backend=os.getenv("RURALSTROKE_STORAGE_BACKEND", cls.storage_backend).lower(),
            s3_bucket=os.getenv("RURALSTROKE_S3_BUCKET"),
            s3_region=os.getenv("RURALSTROKE_S3_REGION", cls.s3_region),
            s3_endpoint_url=os.getenv("RURALSTROKE_S3_ENDPOINT_URL"),
            s3_access_key_id=access_key_id,
            s3_secret_access_key=secret_access_key,
            auth_backend=os.getenv("RURALSTROKE_AUTH_BACKEND", cls.auth_backend).lower(),
            cognito_issuer=os.getenv("RURALSTROKE_COGNITO_ISSUER"),
            cognito_client_id=client_id,
            cognito_client_ids=client_ids,
            cognito_jwks_uri=os.getenv("RURALSTROKE_COGNITO_JWKS_URI"),
            jwt_secret=jwt_secret,
            jwt_issuer=os.getenv("RURALSTROKE_JWT_ISSUER", cls.jwt_issuer),
            jwt_audience=os.getenv("RURALSTROKE_JWT_AUDIENCE", cls.jwt_audience),
            token_ttl_seconds=int(os.getenv("RURALSTROKE_TOKEN_TTL_SECONDS", str(cls.token_ttl_seconds))),
            version=os.getenv("RURALSTROKE_API_VERSION", cls.version),
            max_attachment_bytes=int(os.getenv("RURALSTROKE_MAX_ATTACHMENT_BYTES", str(cls.max_attachment_bytes))),
        )
