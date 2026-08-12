from __future__ import annotations

import base64
import json
from pathlib import Path

from scripts.stack import ensure_demo_secrets


def _decode_payload(token: str) -> dict[str, object]:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def test_ensure_demo_secrets_generates_gitignored_local_values(tmp_path: Path) -> None:
    secrets = ensure_demo_secrets(tmp_path)

    assert set(secrets) == {"postgres_password", "database_url", "jwt_secret", "minio_access_key", "minio_secret_key", "demo_token"}
    assert all(path.is_file() and path.read_text(encoding="utf-8").strip() for path in secrets.values())
    assert "postgresql+psycopg://ruralstroke:" in secrets["database_url"].read_text(encoding="utf-8")
    payload = _decode_payload(secrets["demo_token"].read_text(encoding="utf-8").strip())
    assert payload["iss"] == "ruralstroke-local"
    assert payload["aud"] == "ruralstroke-api"
    assert payload["roles"] == ["collector", "clinician"]
    assert payload["facilities"] == ["*"]


def test_ensure_demo_secrets_reuses_existing_values(tmp_path: Path) -> None:
    first = ensure_demo_secrets(tmp_path)
    values = {name: path.read_text(encoding="utf-8") for name, path in first.items()}

    second = ensure_demo_secrets(tmp_path)

    assert {name: path.read_text(encoding="utf-8") for name, path in second.items()} == values


def test_minio_secret_is_safe_as_cli_argument(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.stack.random_secrets.token_urlsafe", lambda length: "-leading-option")

    secrets = ensure_demo_secrets(tmp_path)

    assert not secrets["minio_secret_key"].read_text(encoding="utf-8").strip().startswith("-")
