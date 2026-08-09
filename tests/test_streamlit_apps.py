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
    streamlit.cache_resource.clear()
    at = AppTest.from_file(str(path), default_timeout=30)
    at.run()
    return at


def test_collector_app_starts_with_api_boundary_message(tmp_path: Path) -> None:
    at = run_app(ROOT / "apps" / "collector_app.py", tmp_path)
    assert not at.exception
    assert at.title[0].value == "RuralStroke-Assist · Collector"
    assert any(item.label == "Start or save draft" for item in at.button)


def test_clinician_app_has_empty_queue_state_when_api_unavailable(tmp_path: Path) -> None:
    at = run_app(ROOT / "apps" / "clinician_app.py", tmp_path)
    assert not at.exception
    assert at.title[0].value == "RuralStroke-Assist · Clinician review"
    assert any("No submitted cases" in item.value for item in at.info)
