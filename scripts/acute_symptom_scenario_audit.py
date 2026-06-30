from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.modules.acute_symptom_module import (
    assess_acute_stroke_symptoms,
)


RISK_ORDER = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "URGENT": 3,
}

OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "acute_symptom_scenario_audit.csv"
)


def make_scenario(
    scenario_name: str,
    expected_min_band: str,
    **symptom_kwargs: bool | int | None,
) -> dict[str, object]:
    return {
        "scenario_name": scenario_name,
        "expected_min_band": expected_min_band,
        "symptom_kwargs": symptom_kwargs,
    }


def build_scenarios() -> list[dict[str, object]]:
    return [
        make_scenario("no_symptoms", "LOW"),
        make_scenario(
            "only_onset_known_within_window",
            "LOW",
            symptom_onset_minutes=120,
        ),
        make_scenario(
            "only_onset_known_late",
            "LOW",
            symptom_onset_minutes=420,
        ),
        make_scenario(
            "resolved_no_current_symptoms",
            "LOW",
            symptoms_resolved=True,
        ),
        make_scenario("face_only", "MODERATE", face_drooping=True),
        make_scenario("arm_only", "MODERATE", arm_weakness=True),
        make_scenario("speech_only", "MODERATE", speech_difficulty=True),
        make_scenario(
            "face_only_within_window",
            "MODERATE",
            face_drooping=True,
            symptom_onset_minutes=90,
        ),
        make_scenario(
            "face_only_late",
            "MODERATE",
            face_drooping=True,
            symptom_onset_minutes=420,
        ),
        make_scenario(
            "face_only_resolved",
            "MODERATE",
            face_drooping=True,
            symptoms_resolved=True,
        ),
        make_scenario(
            "face_arm",
            "URGENT",
            face_drooping=True,
            arm_weakness=True,
        ),
        make_scenario(
            "face_speech",
            "URGENT",
            face_drooping=True,
            speech_difficulty=True,
        ),
        make_scenario(
            "arm_speech",
            "URGENT",
            arm_weakness=True,
            speech_difficulty=True,
        ),
        make_scenario(
            "face_arm_speech",
            "URGENT",
            face_drooping=True,
            arm_weakness=True,
            speech_difficulty=True,
        ),
        make_scenario(
            "face_arm_resolved",
            "URGENT",
            face_drooping=True,
            arm_weakness=True,
            symptoms_resolved=True,
        ),
        make_scenario(
            "face_arm_within_window",
            "URGENT",
            face_drooping=True,
            arm_weakness=True,
            symptom_onset_minutes=60,
        ),
        make_scenario(
            "balance_only",
            "LOW",
            balance_or_coordination_loss=True,
        ),
        make_scenario(
            "vision_only",
            "LOW",
            vision_disturbance=True,
        ),
        make_scenario(
            "confusion_only",
            "LOW",
            confusion_or_understanding_difficulty=True,
        ),
        make_scenario(
            "balance_vision",
            "MODERATE",
            balance_or_coordination_loss=True,
            vision_disturbance=True,
        ),
        make_scenario(
            "balance_vision_within_window",
            "MODERATE",
            balance_or_coordination_loss=True,
            vision_disturbance=True,
            symptom_onset_minutes=180,
        ),
        make_scenario(
            "balance_vision_late",
            "MODERATE",
            balance_or_coordination_loss=True,
            vision_disturbance=True,
            symptom_onset_minutes=420,
        ),
        make_scenario(
            "balance_vision_headache",
            "MODERATE",
            balance_or_coordination_loss=True,
            vision_disturbance=True,
            sudden_severe_headache=True,
        ),
        make_scenario(
            "confusion_vision",
            "MODERATE",
            confusion_or_understanding_difficulty=True,
            vision_disturbance=True,
        ),
        make_scenario(
            "balance_vision_confusion",
            "MODERATE",
            balance_or_coordination_loss=True,
            vision_disturbance=True,
            confusion_or_understanding_difficulty=True,
        ),
        make_scenario(
            "resolved_balance_vision",
            "MODERATE",
            balance_or_coordination_loss=True,
            vision_disturbance=True,
            symptoms_resolved=True,
        ),
        make_scenario(
            "face_plus_balance",
            "HIGH",
            face_drooping=True,
            balance_or_coordination_loss=True,
        ),
        make_scenario(
            "arm_plus_balance",
            "HIGH",
            arm_weakness=True,
            balance_or_coordination_loss=True,
        ),
        make_scenario(
            "speech_plus_confusion",
            "HIGH",
            speech_difficulty=True,
            confusion_or_understanding_difficulty=True,
        ),
        make_scenario(
            "speech_plus_vision",
            "HIGH",
            speech_difficulty=True,
            vision_disturbance=True,
        ),
        make_scenario(
            "speech_plus_balance_within_window",
            "HIGH",
            speech_difficulty=True,
            balance_or_coordination_loss=True,
            symptom_onset_minutes=120,
        ),
        make_scenario(
            "speech_plus_arm",
            "URGENT",
            speech_difficulty=True,
            arm_weakness=True,
        ),
        make_scenario(
            "speech_plus_face",
            "URGENT",
            speech_difficulty=True,
            face_drooping=True,
        ),
    ]


def audit_scenarios() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scenario in build_scenarios():
        symptoms = AcuteStrokeSymptoms(**scenario["symptom_kwargs"])
        result = assess_acute_stroke_symptoms(symptoms)
        row = {
            "scenario_name": scenario["scenario_name"],
            "input_symptoms": json.dumps(symptoms.model_dump(), sort_keys=True),
            "acute_symptom_score": result.acute_symptom_score,
            "risk_band": result.risk_band,
            "hard_escalation": result.hard_escalation,
            "evidence": json.dumps(result.evidence),
            "warnings": json.dumps(result.warnings),
            "expected_min_band": scenario["expected_min_band"],
        }
        row["passes_expectation"] = (
            RISK_ORDER[row["risk_band"]] >= RISK_ORDER[row["expected_min_band"]]
        )
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    audit_df = audit_scenarios()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(OUTPUT_PATH, index=False)

    failing = audit_df.loc[~audit_df["passes_expectation"], ["scenario_name", "risk_band", "expected_min_band"]]

    print(f"Scenarios tested: {len(audit_df)}")
    print(f"Scenarios passing expectation: {int(audit_df['passes_expectation'].sum())}")
    print(f"Scenarios failing expectation: {len(failing)}")
    if failing.empty:
        print("Failing scenarios: none")
    else:
        print("Failing scenarios:")
        print(failing.to_string(index=False))

    print(f"Saved audit CSV to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
