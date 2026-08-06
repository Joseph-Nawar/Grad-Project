"""Build a deterministic Stage 1 engineering baseline from Phase 4 evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "reports" / "evaluation" / "phase4" / "final_complete"
OUTPUT_PATH = ROOT / "reports" / "production" / "stage1_engineering_baseline.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_phase4_evidence(evidence_root: Path = EVIDENCE_ROOT) -> dict[str, Any]:
    """Load the JSON and CSV files used by the Stage 1 baseline."""
    json_names = (
        "profiling.json",
        "cold_start.json",
        "stability.json",
        "missing_modalities.json",
        "robustness.json",
        "system_corruption_summary.json",
        "face.json",
        "speech.json",
        "metadata_context.json",
        "symptoms.json",
        "run_manifest.json",
    )
    csv_names = ("cold_start_subprocess_runs.csv", "face_acceptance_by_quality.csv")
    return {
        "json": {name: _read_json(evidence_root / name) for name in json_names},
        "csv": {name: _read_csv(evidence_root / name) for name in csv_names},
    }


def build_baseline(evidence_root: Path = EVIDENCE_ROOT) -> dict[str, Any]:
    """Return stable measured values without rerunning evaluation."""
    evidence = load_phase4_evidence(evidence_root)
    data = evidence["json"]
    profiling = data["profiling.json"]
    cold = data["cold_start.json"]
    stability = data["stability.json"]
    face = data["face.json"]
    speech = data["speech.json"]
    metadata = data["metadata_context.json"]
    corruption = data["system_corruption_summary.json"]
    return {
        "schema_version": 1,
        "source": {
            "evaluation_root": "reports/evaluation/phase4/final_complete",
            "run_manifest": data["run_manifest.json"]["run_id"],
            "method": "Parsed existing Phase 4 JSON/CSV artifacts; no evaluation was rerun.",
        },
        "baselines": {
            "assessment_service": {
                "warm_runs": profiling["assessment_service"]["warm_runs"],
                "warm_median_ms": profiling["assessment_service"]["warm_median_ms"],
                "warm_p95_ms": profiling["assessment_service"]["warm_p95_ms"],
                "peak_rss_bytes": profiling["assessment_service"]["peak_rss_bytes"],
                "evidence": ["profiling.json"],
            },
            "cold_start": {
                "successful_runs": cold["successful_runs"],
                "process_wall_ms": [run["process_wall_ms"] for run in cold["runs"]],
                "assessment_ms": [run["assessment_ms"] for run in cold["runs"]],
                "peak_child_rss_bytes": [run["peak_child_rss_bytes"] for run in cold["runs"]],
                "evidence": ["cold_start.json", "cold_start_subprocess_runs.csv"],
            },
            "memory": {
                "assessment_peak_rss_bytes": profiling["assessment_service"]["peak_rss_bytes"],
                "assessment_memory_increase_bytes": profiling["assessment_service"]["memory_increase_bytes"],
                "adapter_peak_rss_bytes": {
                    name: profiling["adapters"][name]["peak_rss_bytes"]
                    for name in sorted(profiling["adapters"])
                },
                "evidence": ["profiling.json"],
            },
            "stability": {
                "assessment_runs": stability["assessment_service"]["runs"],
                "assessment_score_max_absolute_difference": stability["assessment_service"]["max_absolute_difference"],
                "all_checked_components_consistent": all(
                    item["band_consistent"] and item["label_consistent"] and item["quality_consistent"]
                    for item in stability.values()
                ),
                "evidence": ["stability.json", "stability_results.json"],
            },
            "face": {
                "held_out": face["direct"]["n"],
                "coverage": face["adapter_coverage"]["coverage"],
                "accepted": face["adapter_coverage"]["accepted"],
                "rejected": face["adapter_coverage"]["rejected"],
                "direct_roc_auc": face["direct"]["roc_auc"],
                "accepted_subset_roc_auc": face["adapter_aware"]["roc_auc"],
                "evidence": ["face.json", "face_adapter_coverage.json"],
            },
            "speech": {
                "held_out": speech["direct"]["n"],
                "coverage": speech["coverage"],
                "accepted": speech["accepted"],
                "rejected": speech["rejected"],
                "direct_roc_auc": speech["direct"]["roc_auc"],
                "accepted_subset_roc_auc": speech["adapter_aware"]["roc_auc"],
                "evidence": ["speech.json"],
            },
            "metadata_context": {
                "held_out": metadata["n"],
                "direct_roc_auc": metadata["metrics"]["roc_auc"],
                "precision": metadata["metrics"]["precision"],
                "evidence": ["metadata_context.json"],
            },
            "modality_combinations": {
                "combination_count": data["missing_modalities.json"]["combination_count"],
                "real_check_count": data["missing_modalities.json"]["real_check_count"],
                "robustness_non_empty_combinations": len(data["robustness.json"]["non_empty_combinations"]),
                "evidence": ["missing_modalities.json", "robustness.json"],
            },
            "corruption": {
                "scenario_count": corruption["count"],
                "passed": corruption["passed"],
                "all_passed": corruption["passed"] == corruption["count"],
                "evidence": ["system_corruption_summary.json", "corruption.json"],
            },
        },
    }


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline()
    OUTPUT_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
