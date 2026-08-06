"""Validate the Stage 1 production specification and target metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "docs" / "production" / "PORTFOLIO_PRODUCTION_SPEC.md"
TARGETS_PATH = ROOT / "config" / "production_targets.yaml"
BASELINE_PATH = ROOT / "reports" / "production" / "stage1_engineering_baseline.json"
ARCHITECTURE_PATH = ROOT / "docs" / "production" / "ARCHITECTURE_DECISIONS.md"
TRACEABILITY_PATH = ROOT / "docs" / "production" / "ROADMAP_TRACEABILITY.md"
STATUSES = {"measured_baseline", "candidate_budget", "committed_invariant", "deferred"}
ROADMAP_SEQUENCE = (
    "Stage 0 — Frozen research MVP",
    "Stage 1 — Target engineering specification",
    "Stage 2 — Production API and central persistence",
    "Stage 3 — Docker Compose and deployable service stack",
    "Stage 4 — Offline outbox and resilient synchronization",
    "Stage 5 — Edge model conversion and parity benchmarking",
    "Stage 6 — Observability, drift simulation, and alerts",
    "Stage 7 — Load, security, reliability, and fault-injection evidence",
    "Stage 8 — Recruiter-facing demos, case study, and external validation",
)
RETENTION_DEFAULTS = (
    "Cloud demo cases and attachments: 7 days",
    "Operational logs:                 14 days",
    "Aggregated metrics:               30 days",
    "Local synchronized media:         deleted after a configurable grace period",
    "Pending or failed sync records:    retained until resolved or manually deleted",
)
STAGE_2_DECISIONS = (
    "FastAPI",
    "feature-oriented API/application/infrastructure boundaries",
    "Pydantic v2 schemas separate from domain contracts",
    "SQLAlchemy 2.x",
    "Alembic",
    "Psycopg 3",
    "real PostgreSQL container, not SQLite emulation",
    "httpx.AsyncClient",
    "cursor pagination",
    "(created_at, id)",
    "dedicated idempotency table",
    "opaque UUID",
    "SHA-256 checksum",
    "Filesystem implementation behind a storage protocol",
    "S3 implementation deferred",
    "verified claims or test JWT verifier",
    "Cognito-compatible OIDC/JWKS verifier",
    "generated schema snapshot",
    "operation-ID and contract tests",
)
REQUIRED_SECTIONS = (
    "target roles and competencies",
    "deployment profiles",
    "api boundary",
    "authentication and authorization",
    "offline and synchronization contract",
    "immutable submissions and revision policy",
    "observability and drift",
    "retention and privacy assumptions",
    "explicit non-goals",
)
ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:Users|home|tmp|var|etc)/)")
UNSUPPORTED_CLAIM = re.compile(
    r"\b(?:is|are|provides|achieves|guarantees)\s+(?:clinically accurate|regulatory compliant|clinical diagnostic accuracy)\b",
    re.IGNORECASE,
)
CURRENT_FUTURE_CLAIM = re.compile(
    r"\b(?:currently implemented|currently supports|implemented today)\b.{0,100}\b(?:fastapi|docker|postgres(?:ql)?|cognito|oidc|jwt|synchroni[sz]|kubernetes|opentelemetry)\b",
    re.IGNORECASE | re.DOTALL,
)


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return value if isinstance(value, dict) else {}


def _validate_sections(text: str) -> list[str]:
    lowered = text.lower()
    return [f"missing required section: {section}" for section in REQUIRED_SECTIONS if section not in lowered]


def _validate_targets(targets: dict[str, Any], baseline_path: Path) -> list[str]:
    errors: list[str] = []
    items = targets.get("targets")
    if not isinstance(items, list):
        return ["targets must be a list"]
    baseline_exists = baseline_path.is_file()
    for index, target in enumerate(items):
        prefix = f"target {index}"
        if not isinstance(target, dict):
            errors.append(f"{prefix} must be a mapping")
            continue
        status = target.get("status")
        if status not in STATUSES:
            errors.append(f"{prefix} has invalid status: {status}")
        value = target.get("value")
        stage = str(target.get("roadmap_stage", ""))
        if not re.fullmatch(r"Stage [1-8]", stage):
            errors.append(f"{prefix} has missing or invalid roadmap stage")
        if _numeric(value):
            if not target.get("unit"):
                errors.append(f"{prefix} numeric target requires a unit")
            if not target.get("measurement_method"):
                errors.append(f"{prefix} numeric target requires a measurement method")
        if status == "measured_baseline" and not target.get("evidence"):
            errors.append(f"{prefix} measured baseline requires evidence")
        if status == "measured_baseline" and not baseline_exists:
            errors.append(f"{prefix} measured baseline evidence file is missing")
        if status == "candidate_budget":
            verification = str(target.get("verification", "")).lower()
            if target.get("achieved") is True or re.search(r"\bachieved\b|\bverified\b", verification):
                errors.append(f"{prefix} candidate budget is presented as achieved")
            if not target.get("promotion_gate"):
                errors.append(f"{prefix} candidate budget requires a promotion gate")
        for field in ("environment", "rationale"):
            if not target.get(field):
                errors.append(f"{prefix} requires {field}")
    return errors


def _validate_roadmap(text: str) -> list[str]:
    positions = []
    for stage in ROADMAP_SEQUENCE:
        position = text.find(stage)
        if position < 0:
            return [f"roadmap missing required stage: {stage}"]
        positions.append(position)
    if positions != sorted(positions):
        return ["roadmap stages are out of order"]
    return []


def _validate_endpoints(text: str) -> list[str]:
    errors: list[str] = []
    operational = {"/health/live", "/health/ready", "/version"}
    for endpoint in re.findall(r"`(?:GET|POST|PATCH|DELETE|PUT)\s+([^`\s]+)", text):
        if endpoint in operational:
            continue
        if endpoint.startswith("/api/v1/health/") or endpoint == "/api/v1/version":
            errors.append("operational endpoint incorrectly placed under /api/v1")
        elif not endpoint.startswith("/api/v1/"):
            errors.append(f"business endpoint outside /api/v1: {endpoint}")
    return errors


def _validate_stage1_content(text: str, architecture_text: str, traceability_text: str) -> list[str]:
    errors = _validate_roadmap(text)
    errors.extend(_validate_roadmap(traceability_text))
    lowered = text.lower()
    for default in RETENTION_DEFAULTS:
        if default.lower() not in lowered:
            errors.append(f"missing retention default: {default}")
    if "collector durable outbox" not in lowered or "server transactional outbox" not in lowered:
        errors.append("collector and server outboxes are not explicitly distinguished")
    if re.search(r"creates the immutable submitted snapshot and outbox event", lowered):
        errors.append("case submission conflates collector and server outboxes")
    errors.extend(_validate_endpoints(text))
    architecture_lower = architecture_text.lower()
    for decision in STAGE_2_DECISIONS:
        if decision.lower() not in architecture_lower:
            errors.append(f"unresolved Stage 2 decision: {decision}")
    return errors


def validate_spec(
    spec_path: Path = SPEC_PATH,
    targets_path: Path = TARGETS_PATH,
    baseline_path: Path = BASELINE_PATH,
    architecture_path: Path = ARCHITECTURE_PATH,
    traceability_path: Path = TRACEABILITY_PATH,
) -> list[str]:
    """Return validation errors; an empty list means the Stage 1 spec is valid."""
    errors: list[str] = []
    text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    architecture_text = architecture_path.read_text(encoding="utf-8") if architecture_path.is_file() else ""
    traceability_text = traceability_path.read_text(encoding="utf-8") if traceability_path.is_file() else ""
    errors.extend(_validate_sections(text))
    errors.extend(_validate_stage1_content(text, architecture_text, traceability_text))
    if ABSOLUTE_PATH.search(text):
        errors.append("absolute local path found in specification")
    if UNSUPPORTED_CLAIM.search(text):
        errors.append("unsupported medical or regulatory claim found")
    if CURRENT_FUTURE_CLAIM.search(text):
        errors.append("future feature is described as currently implemented")
    if "non-identifiable" in text.lower() and re.search(r"identifiable .* indefinitely", text, re.IGNORECASE):
        errors.append("contradictory retention decisions found")
    try:
        targets = _load_yaml(targets_path)
    except (OSError, yaml.YAMLError) as exc:
        return errors + [f"could not load targets: {type(exc).__name__}: {exc}"]
    errors.extend(_validate_targets(targets, baseline_path))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--targets", type=Path, default=TARGETS_PATH)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--architecture", type=Path, default=ARCHITECTURE_PATH)
    parser.add_argument("--traceability", type=Path, default=TRACEABILITY_PATH)
    args = parser.parse_args()
    errors = validate_spec(args.spec, args.targets, args.baseline, args.architecture, args.traceability)
    if errors:
        print("Production specification validation: FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Production specification validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
