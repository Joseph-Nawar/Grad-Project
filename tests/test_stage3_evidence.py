from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_stage3_evidence_schema_is_machine_readable_and_separate_from_stage2() -> None:
    schema = json.loads((ROOT / "reports/production/stage3/evidence.schema.json").read_text(encoding="utf-8"))

    assert schema["type"] == "object"
    required = set(schema["required"])
    assert {"image", "stack", "benchmark", "environment", "frozen_compatibility"} <= required
    assert "reports/production/stage2" not in str(schema)


def test_evidence_collector_does_not_capture_secret_values() -> None:
    script = (ROOT / "scripts/collect_stage3_evidence.py").read_text(encoding="utf-8")

    assert "JWT_SECRET" not in script
    assert "AWS_SECRET_ACCESS_KEY" not in script
    assert "container_rss_bytes" in script
    assert "p95_ms" in script
