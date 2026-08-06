from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_stage1_engineering_baseline import build_baseline
from scripts.validate_production_spec import validate_spec


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "evaluation" / "phase4" / "final_complete"
SPEC = ROOT / "docs" / "production" / "PORTFOLIO_PRODUCTION_SPEC.md"
TARGETS = ROOT / "config" / "production_targets.yaml"
ARCHITECTURE = ROOT / "docs" / "production" / "ARCHITECTURE_DECISIONS.md"


def test_baseline_builder_extracts_authoritative_phase4_measurements() -> None:
    baseline = build_baseline(EVIDENCE)

    assert baseline["baselines"]["assessment_service"]["warm_median_ms"] == 121.7612
    assert baseline["baselines"]["cold_start"]["successful_runs"] == 3
    assert baseline["baselines"]["face"]["coverage"] == pytest.approx(0.33647798742138363)
    assert baseline["baselines"]["corruption"]["passed"] == 10
    assert baseline["baselines"]["modality_combinations"]["combination_count"] == 16


def test_baseline_builder_is_json_deterministic() -> None:
    first = json.dumps(build_baseline(EVIDENCE), sort_keys=True, separators=(",", ":"))
    second = json.dumps(build_baseline(EVIDENCE), sort_keys=True, separators=(",", ":"))
    assert first == second


def test_production_spec_is_valid() -> None:
    errors = validate_spec(SPEC, TARGETS, ROOT / "reports" / "production" / "stage1_engineering_baseline.json")
    assert errors == []


def test_validator_rejects_missing_required_section(tmp_path: Path) -> None:
    spec = tmp_path / "spec.md"
    spec.write_text("# API boundary\n", encoding="utf-8")
    errors = validate_spec(spec, TARGETS, ROOT / "reports" / "production" / "stage1_engineering_baseline.json")
    assert any("required section" in error for error in errors)


def test_validator_rejects_invalid_status(tmp_path: Path) -> None:
    targets = tmp_path / "targets.yaml"
    targets.write_text("schema_version: 1\ntargets:\n  - id: x\n    status: achieved\n    value: 1\n    unit: ms\n    measurement_method: test\n", encoding="utf-8")
    errors = validate_spec(SPEC, targets, ROOT / "reports" / "production" / "stage1_engineering_baseline.json")
    assert any("invalid status" in error for error in errors)


@pytest.mark.parametrize(
    "target_fragment,expected",
    [
        ("value: 10\n    measurement_method: test", "unit"),
        ("value: 10\n    unit: ms", "measurement method"),
        ("status: measured_baseline\n    value: 10\n    unit: ms\n    measurement_method: test", "evidence"),
    ],
)
def test_validator_rejects_incomplete_numeric_targets(tmp_path: Path, target_fragment: str, expected: str) -> None:
    targets = tmp_path / "targets.yaml"
    targets.write_text(
        "schema_version: 1\ntargets:\n  - id: x\n    status: measured_baseline\n    " + target_fragment + "\n",
        encoding="utf-8",
    )
    errors = validate_spec(SPEC, targets, ROOT / "reports" / "production" / "stage1_engineering_baseline.json")
    assert any(expected in error.lower() for error in errors)


def test_validator_rejects_unsafe_claims_and_absolute_paths(tmp_path: Path) -> None:
    spec = tmp_path / "spec.md"
    spec.write_text(
        "# API boundary\n# Target roles and competencies\n# Deployment profiles\n"
        "# Authentication and authorization\n# Offline and synchronization contract\n"
        "# Immutable submissions and revision policy\n# Observability and drift\n# Retention and non-goals\n"
        "This is clinically accurate and regulatory compliant. C:\\secret\\file.\n",
        encoding="utf-8",
    )
    errors = validate_spec(spec, TARGETS, ROOT / "reports" / "production" / "stage1_engineering_baseline.json")
    assert any("absolute" in error for error in errors)
    assert any("unsupported" in error for error in errors)


def test_validator_rejects_contradictory_retention(tmp_path: Path) -> None:
    spec = tmp_path / "spec.md"
    spec.write_text(
        SPEC.read_text(encoding="utf-8")
        + "\nAll demo data is non-identifiable and the system retains identifiable patient data indefinitely.\n",
        encoding="utf-8",
    )
    errors = validate_spec(spec, TARGETS, ROOT / "reports" / "production" / "stage1_engineering_baseline.json")
    assert any("retention" in error.lower() or "contradict" in error.lower() for error in errors)


def test_validator_requires_authoritative_roadmap_order() -> None:
    errors = validate_spec(SPEC, TARGETS, ROOT / "reports" / "production" / "stage1_engineering_baseline.json")
    assert not any("roadmap" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "mutator,expected",
    [
        (lambda text: text.replace("Stage 8 — Recruiter-facing demos", "Stage 9 — Recruiter-facing demos"), "roadmap"),
        (lambda text: text.replace("Stage 5 — Edge model conversion", "Stage 5 — Future work"), "Stage 5"),
        (lambda text: text.replace("Collector durable outbox", "Central outbox"), "outbox"),
        (lambda text: text.replace("Cloud demo cases and attachments: 7 days", "Cloud demo cases and attachments: 90 days"), "retention"),
        (lambda text: text.replace("/api/v1/cases", "/cases"), "business endpoint"),
    ],
)
def test_validator_rejects_stage1_documentation_regressions(
    tmp_path: Path, mutator, expected: str
) -> None:
    spec = tmp_path / "spec.md"
    spec.write_text(mutator(SPEC.read_text(encoding="utf-8")), encoding="utf-8")
    errors = validate_spec(spec, TARGETS, ROOT / "reports" / "production" / "stage1_engineering_baseline.json")
    assert any(expected.lower() in error.lower() for error in errors)


def test_validator_rejects_operational_endpoint_under_business_version(tmp_path: Path) -> None:
    spec = tmp_path / "spec.md"
    spec.write_text(SPEC.read_text(encoding="utf-8").replace("`GET /health/live`", "`GET /api/v1/health/live`"), encoding="utf-8")
    errors = validate_spec(spec, TARGETS, ROOT / "reports" / "production" / "stage1_engineering_baseline.json")
    assert any("operational" in error.lower() for error in errors)


def test_validator_rejects_unresolved_stage2_technology_decision(tmp_path: Path) -> None:
    architecture = tmp_path / "architecture.md"
    architecture.write_text(ARCHITECTURE.read_text(encoding="utf-8").replace("FastAPI", "an undecided framework"), encoding="utf-8")
    errors = validate_spec(SPEC, TARGETS, ROOT / "reports" / "production" / "stage1_engineering_baseline.json", architecture)
    assert any("stage 2" in error.lower() or "fastapi" in error.lower() for error in errors)
