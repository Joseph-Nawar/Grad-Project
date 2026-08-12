from __future__ import annotations

from pathlib import Path

from rural_stroke_assist.server.app import create_attachment_store
from rural_stroke_assist.server.config import ServerSettings
from rural_stroke_assist.server.infrastructure.storage.filesystem import FilesystemAttachmentStore
from rural_stroke_assist.server.infrastructure.storage.s3 import S3AttachmentStore
from rural_stroke_assist.server.infrastructure.auth.cognito import CognitoTokenVerifier


def test_settings_load_s3_backend_and_file_based_minio_secret(monkeypatch, tmp_path: Path) -> None:
    secret_file = tmp_path / "minio-secret"
    secret_file.write_text("minio-secret\n", encoding="utf-8")
    monkeypatch.setenv("RURALSTROKE_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("RURALSTROKE_S3_BUCKET", "ruralstroke")
    monkeypatch.setenv("RURALSTROKE_S3_ENDPOINT_URL", "http://object-store:9000")
    monkeypatch.setenv("RURALSTROKE_S3_ACCESS_KEY_ID", "minio-user")
    monkeypatch.setenv("RURALSTROKE_S3_SECRET_ACCESS_KEY_FILE", str(secret_file))

    settings = ServerSettings.from_environment()

    assert settings.storage_backend == "s3"
    assert settings.s3_bucket == "ruralstroke"
    assert settings.s3_secret_access_key == "minio-secret"
    assert settings.s3_region == "eu-central-1"


def test_create_attachment_store_preserves_filesystem_default(tmp_path: Path) -> None:
    settings = ServerSettings(storage_root=tmp_path)

    store = create_attachment_store(settings)

    assert isinstance(store, FilesystemAttachmentStore)


def test_create_attachment_store_selects_s3_backend() -> None:
    settings = ServerSettings(storage_backend="s3", s3_bucket="ruralstroke")

    store = create_attachment_store(settings, client=object())

    assert isinstance(store, S3AttachmentStore)


def test_create_token_verifier_selects_cognito_backend() -> None:
    from rural_stroke_assist.server.app import create_token_verifier

    settings = ServerSettings(
        auth_backend="cognito",
        cognito_issuer="https://cognito-idp.eu-central-1.amazonaws.com/eu-central-1_example",
        cognito_client_id="client-1",
    )

    verifier = create_token_verifier(settings, fetch_json=lambda _url: {"keys": []})

    assert isinstance(verifier, CognitoTokenVerifier)
