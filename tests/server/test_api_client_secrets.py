from __future__ import annotations

from pathlib import Path

from rural_stroke_assist.client.sync_http import ApiClient


def test_api_client_loads_token_from_secret_file(monkeypatch, tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("demo-token\n", encoding="utf-8")
    monkeypatch.delenv("RURALSTROKE_API_TOKEN", raising=False)
    monkeypatch.setenv("RURALSTROKE_API_TOKEN_FILE", str(token_file))

    client = ApiClient()

    assert client.token == "demo-token"
    client.client.close()
