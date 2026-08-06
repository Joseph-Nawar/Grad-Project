"""Validate a completed clinical-safety review without fabricating approval."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

def validate_review(input_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    source = Path(input_path); destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    response = json.loads(source.read_text(encoding="utf-8"))
    required = ("reviewer_name", "reviewer_role", "reviewer_organisation", "decision", "reviewed_at_utc", "responses")
    missing = [field for field in required if not response.get(field)]
    decision = response.get("decision")
    status = "PENDING" if missing or decision not in {"approved", "approved_with_changes", "not_approved"} else decision.upper()
    status_data = {"status": status, "validated_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(source), "missing_fields": missing}
    status_path = destination / "clinical_review_status.json"; status_path.write_text(json.dumps(status_data, indent=2) + "\n", encoding="utf-8")
    log_path = destination / "clinical_review_action_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "action", "status"]); writer.writeheader(); writer.writerow({"timestamp_utc": status_data["validated_at_utc"], "action": "validated", "status": status})
    return status_path, log_path
