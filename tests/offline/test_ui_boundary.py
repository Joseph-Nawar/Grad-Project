from __future__ import annotations

import os
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_edge_collector_starts_offline_and_exposes_safe_sync_controls(tmp_path: Path) -> None:
    os.environ["RURALSTROKE_OFFLINE_DB"] = str(tmp_path / "collector.sqlite3")
    os.environ["RURALSTROKE_OFFLINE_MEDIA"] = str(tmp_path / "media")
    os.environ["RURALSTROKE_API_URL"] = "http://127.0.0.1:9"
    os.environ.pop("RURALSTROKE_API_TOKEN", None)
    streamlit.cache_resource.clear()

    at = AppTest.from_file(
        str(Path(__file__).parents[2] / "apps" / "collector_edge_app.py"), default_timeout=60
    )
    at.run()

    assert not at.exception
    assert at.title[0].value == "RuralStroke-Assist - Collector Edge"
    assert any("Offline" in item.value for item in at.info)
    labels = {item.label for item in at.button}
    assert {
        "Save local draft",
        "Assess locally",
        "Queue for sync",
        "Retry synchronization",
        "Restore authentication and retry",
        "Clone as new local draft",
    }.issubset(labels)
    rendered = " ".join(item.value for item in at.markdown)
    assert "JWT" not in rendered
    assert "outbox" not in rendered.lower()
