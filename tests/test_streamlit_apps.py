from __future__ import annotations

import os
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def run_app(path: Path, tmp_path: Path) -> AppTest:
    os.environ["RURALSTROKE_API_URL"] = "http://127.0.0.1:9"
    os.environ.pop("RURALSTROKE_API_TOKEN", None)
    os.environ.pop("RURALSTROKE_API_TOKEN_FILE", None)
    streamlit.cache_resource.clear()
    at = AppTest.from_file(str(path), default_timeout=30)
    at.run()
    return at


def test_collector_app_starts_with_shared_product_identity(tmp_path: Path) -> None:
    at = run_app(ROOT / "apps" / "collector_app.py", tmp_path)
    assert not at.exception
    assert any("RuralStroke-Triage" in item.value for item in at.markdown)
    assert any("Draft list is temporarily unavailable" in item.value for item in at.error)


def test_clinician_app_has_shared_identity_and_empty_queue_state(tmp_path: Path) -> None:
    at = run_app(ROOT / "apps" / "clinician_app.py", tmp_path)
    assert not at.exception
    assert any("RuralStroke-Triage" in item.value for item in at.markdown)
    assert any("Clinician case queue is temporarily unavailable" in item.value for item in at.error)
