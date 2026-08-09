"""Measure warm API assessment latency without touching Phase 4 evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import statistics
import time
from uuid import uuid4

from sqlalchemy import create_engine, text

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.client import ApiClient, ApiClientError
from rural_stroke_assist.inference.metadata_adapter import MetadataInput

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "production" / "stage2_api_baseline.json"


def _metadata() -> dict[str, object]:
    return {
        "age": 60,
        "hypertension": 0,
        "heart_disease": 0,
        "avg_glucose_level": 100.0,
        "bmi": 25.0,
        "gender": "Male",
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Rural",
        "smoking_status": "Unknown",
    }


def run(iterations: int = 20) -> dict[str, object]:
    client = ApiClient()
    direct_service = create_default_assessment_service()
    direct_input = AssessmentInput(
        session_id="stage2-direct-baseline", metadata=MetadataInput.model_validate(_metadata())
    )
    direct_service.assess(direct_input)
    direct_timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        direct_service.assess(direct_input)
        direct_timings.append((time.perf_counter() - started) * 1000)
    # Exercise one request before timing; this warm-up is intentionally excluded.
    warmup_case = client.create_case(
        case_id=str(uuid4()),
        facility="benchmark-facility",
        patient_code=None,
        assessment_input={
            "session_id": "stage2-benchmark-warmup",
            "metadata": _metadata(),
            "acute_symptoms": None,
        },
    )
    client.create_assessment(
        {"case_id": warmup_case["id"], "metadata": _metadata(), "acute_symptoms": None}
    )
    timings: list[float] = []
    scores: list[float | None] = []
    statuses: list[int] = []
    for _ in range(iterations):
        case = client.create_case(
            case_id=str(uuid4()),
            facility="benchmark-facility",
            patient_code=None,
            assessment_input={
                "session_id": "stage2-benchmark",
                "metadata": _metadata(),
                "acute_symptoms": None,
            },
        )
        started = time.perf_counter()
        result = client.create_assessment(
            {"case_id": case["id"], "metadata": _metadata(), "acute_symptoms": None}
        )
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)
        statuses.append(201)
        scores.append(
            (result.get("result_snapshot") or {}).get("fusion", {}).get("evidence_score")
        )
    ordered = sorted(timings)
    api_version = client._request("GET", "/version")["version"]
    postgres_version = "unavailable"
    database_url = os.getenv("RURALSTROKE_DATABASE_URL")
    if database_url:
        engine = create_engine(database_url)
        try:
            postgres_version = str(engine.connect().execute(text("SELECT version()")).scalar_one())
        finally:
            engine.dispose()
    direct_ordered = sorted(direct_timings)
    direct_p50 = statistics.median(direct_timings)
    direct_p95 = direct_ordered[max(0, int(len(direct_ordered) * 0.95) - 1)]
    return {
        "schema_version": 1,
        "iterations": iterations,
        "warmup_excluded": True,
        "latency_ms": {
            "p50": statistics.median(timings),
            "p95": ordered[max(0, int(len(ordered) * 0.95) - 1)],
            "minimum": min(timings),
            "maximum": max(timings),
        },
        "success_count": sum(status == 201 for status in statuses),
        "statuses": statuses,
        "score_stability": {"all_equal": len(set(scores)) <= 1, "scores": scores},
        "environment": {
            "python": platform.python_version(),
            "api_url": client.base_url,
            "api_version": api_version,
            "postgresql_version": postgres_version,
        },
        "direct_service_baseline_ms": {
            "p50": direct_p50,
            "p95": direct_p95,
            "minimum": min(direct_timings),
            "maximum": max(direct_timings),
        },
        "api_overhead_ms": {
            "p50": statistics.median(timings) - direct_p50,
            "p95": ordered[max(0, int(len(ordered) * 0.95) - 1)] - direct_p95,
        },
        "candidate_p95_budget_ms": 500,
        "promotion_status": "measured",
    }


def main() -> int:
    if not os.getenv("RURALSTROKE_API_TOKEN"):
        print("RURALSTROKE_API_TOKEN is required for the Stage 2 benchmark.")
        return 2
    try:
        report = run()
    except ApiClientError as exc:
        print(f"Stage 2 benchmark unavailable: {exc}")
        return 2
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
