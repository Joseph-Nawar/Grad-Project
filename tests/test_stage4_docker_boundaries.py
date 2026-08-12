from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_collector_edge_target_contains_ml_runtime_and_stays_non_root() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AS collector-edge" in dockerfile
    assert "requirements-ml.txt" in dockerfile
    assert "USER ruralstroke" in dockerfile


def test_compose_has_independent_edge_and_single_worker_named_volumes() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert {"collector-edge", "collector-sync", "clinician"}.issubset(services)
    assert "depends_on" not in services["collector-edge"]
    assert "depends_on" not in services["collector-sync"]
    assert services["collector-edge"]["volumes"] == services["collector-sync"]["volumes"]
    assert any("collector-sqlite" in item for item in services["collector-edge"]["volumes"])
    assert any("collector-media" in item for item in services["collector-edge"]["volumes"])
    assert services["clinician"]["image"] == "ruralstroke-ui:stage3-local"
    assert services["collector-edge"]["environment"]["RURALSTROKE_EDGE_RUNTIME_PROFILE"] == "optimized"
