from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rural_stroke_assist.modules.fusion_module import (  # noqa: E402
    DEFAULT_FUSION_WEIGHTS,
    FusionInput,
    fuse_multimodal_scores,
)


OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "fusion_scenario_audit.csv"
)

RISK_ORDER = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "URGENT": 3,
}

WEIGHT_CONFIGS = {
    "A_current": DEFAULT_FUSION_WEIGHTS,
    "B_equal_acute_low_metadata": {
        "face": 0.30,
        "speech": 0.30,
        "acute_symptoms": 0.30,
        "metadata_context": 0.10,
    },
    "C_symptom_heavier": {
        "face": 0.30,
        "speech": 0.25,
        "acute_symptoms": 0.35,
        "metadata_context": 0.10,
    },
    "D_face_speech_heavier": {
        "face": 0.40,
        "speech": 0.35,
        "acute_symptoms": 0.15,
        "metadata_context": 0.10,
    },
}

THRESHOLD_CONFIGS = {
    "M45_H75": {"moderate_threshold": 0.45, "high_threshold": 0.75},
    "M50_H75": {"moderate_threshold": 0.50, "high_threshold": 0.75},
    "M50_H70": {"moderate_threshold": 0.50, "high_threshold": 0.70},
}


def assign_risk_band(
    fused_score: float,
    *,
    hard_escalation: bool,
    moderate_threshold: float,
    high_threshold: float,
) -> str:
    if hard_escalation:
        return "URGENT"
    if fused_score >= high_threshold:
        return "HIGH"
    if fused_score >= moderate_threshold:
        return "MODERATE"
    return "LOW"


def make_scenario(
    scenario_name: str,
    expected_min_band: str,
    *,
    face_acute_score: float | None = None,
    speech_acute_score: float | None = None,
    acute_symptom_score: float | None = None,
    metadata_contextual_risk_score: float | None = None,
    acute_symptom_hard_escalation: bool = False,
) -> dict[str, object]:
    return {
        "scenario_name": scenario_name,
        "expected_min_band": expected_min_band,
        "input": FusionInput(
            face_acute_score=face_acute_score,
            speech_acute_score=speech_acute_score,
            acute_symptom_score=acute_symptom_score,
            metadata_contextual_risk_score=metadata_contextual_risk_score,
            acute_symptom_hard_escalation=acute_symptom_hard_escalation,
        ),
    }


def build_scenarios() -> list[dict[str, object]]:
    return [
        make_scenario(
            "all_scores_low",
            "LOW",
            face_acute_score=0.10,
            speech_acute_score=0.10,
            acute_symptom_score=0.10,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "metadata_high_only_acute_low",
            "LOW",
            face_acute_score=0.10,
            speech_acute_score=0.12,
            acute_symptom_score=0.08,
            metadata_contextual_risk_score=0.90,
        ),
        make_scenario(
            "missing_one_modality_remaining_low",
            "LOW",
            face_acute_score=None,
            speech_acute_score=0.12,
            acute_symptom_score=0.10,
            metadata_contextual_risk_score=0.25,
        ),
        make_scenario(
            "face_high_only",
            "LOW",
            face_acute_score=0.88,
            speech_acute_score=0.10,
            acute_symptom_score=0.10,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "speech_high_only",
            "LOW",
            face_acute_score=0.10,
            speech_acute_score=0.92,
            acute_symptom_score=0.10,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "acute_symptoms_high_only",
            "LOW",
            face_acute_score=0.10,
            speech_acute_score=0.12,
            acute_symptom_score=0.88,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "metadata_high_only",
            "LOW",
            face_acute_score=0.05,
            speech_acute_score=0.05,
            acute_symptom_score=0.05,
            metadata_contextual_risk_score=0.95,
        ),
        make_scenario(
            "two_acute_modalities_moderate",
            "MODERATE",
            face_acute_score=0.55,
            speech_acute_score=0.55,
            acute_symptom_score=0.20,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "face_and_speech_high",
            "HIGH",
            face_acute_score=0.88,
            speech_acute_score=0.81,
            acute_symptom_score=0.35,
            metadata_contextual_risk_score=0.55,
        ),
        make_scenario(
            "face_high_plus_symptoms_high",
            "HIGH",
            face_acute_score=0.86,
            speech_acute_score=0.20,
            acute_symptom_score=0.82,
            metadata_contextual_risk_score=0.25,
        ),
        make_scenario(
            "speech_high_plus_symptoms_high",
            "HIGH",
            face_acute_score=0.20,
            speech_acute_score=0.89,
            acute_symptom_score=0.82,
            metadata_contextual_risk_score=0.25,
        ),
        make_scenario(
            "three_acute_sources_elevated",
            "HIGH",
            face_acute_score=0.70,
            speech_acute_score=0.68,
            acute_symptom_score=0.62,
            metadata_contextual_risk_score=0.30,
        ),
        make_scenario(
            "face_high_speech_moderate",
            "MODERATE",
            face_acute_score=0.80,
            speech_acute_score=0.55,
            acute_symptom_score=0.20,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "speech_very_high_symptoms_moderate",
            "MODERATE",
            face_acute_score=0.25,
            speech_acute_score=0.92,
            acute_symptom_score=0.35,
            metadata_contextual_risk_score=0.40,
        ),
        make_scenario(
            "face_very_high_speech_low_symptoms_low",
            "LOW",
            face_acute_score=0.80,
            speech_acute_score=0.20,
            acute_symptom_score=0.20,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "face_speech_moderate_symptoms_mild",
            "MODERATE",
            face_acute_score=0.55,
            speech_acute_score=0.55,
            acute_symptom_score=0.35,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "urgent_hard_escalation_low_models",
            "URGENT",
            face_acute_score=0.15,
            speech_acute_score=0.10,
            acute_symptom_score=0.40,
            metadata_contextual_risk_score=0.20,
            acute_symptom_hard_escalation=True,
        ),
        make_scenario(
            "urgent_hard_escalation_high_models",
            "URGENT",
            face_acute_score=0.70,
            speech_acute_score=0.68,
            acute_symptom_score=0.85,
            metadata_contextual_risk_score=0.50,
            acute_symptom_hard_escalation=True,
        ),
        make_scenario(
            "missing_face_strong_remaining_acute",
            "MODERATE",
            face_acute_score=None,
            speech_acute_score=0.82,
            acute_symptom_score=0.52,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "missing_speech_strong_remaining_acute",
            "MODERATE",
            face_acute_score=0.82,
            speech_acute_score=None,
            acute_symptom_score=0.52,
            metadata_contextual_risk_score=0.20,
        ),
        make_scenario(
            "missing_metadata_strong_acute_pair",
            "MODERATE",
            face_acute_score=0.76,
            speech_acute_score=0.72,
            acute_symptom_score=0.20,
            metadata_contextual_risk_score=None,
        ),
        make_scenario(
            "only_symptoms_available",
            "LOW",
            acute_symptom_score=0.55,
        ),
        make_scenario(
            "only_face_and_speech_available_high_pair",
            "HIGH",
            face_acute_score=0.88,
            speech_acute_score=0.81,
        ),
    ]


def analyze_configuration(
    config_name: str,
    weights: dict[str, float],
    threshold_name: str,
    moderate_threshold: float,
    high_threshold: float,
    scenarios: list[dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        fusion_input: FusionInput = scenario["input"]
        base_result = fuse_multimodal_scores(fusion_input, weights=weights)
        risk_band = assign_risk_band(
            fused_score=base_result.fused_score,
            hard_escalation=fusion_input.acute_symptom_hard_escalation,
            moderate_threshold=moderate_threshold,
            high_threshold=high_threshold,
        )
        row = {
            "config_name": config_name,
            "threshold_name": threshold_name,
            "moderate_threshold": moderate_threshold,
            "high_threshold": high_threshold,
            "scenario_name": scenario["scenario_name"],
            "face_acute_score": fusion_input.face_acute_score,
            "speech_acute_score": fusion_input.speech_acute_score,
            "acute_symptom_score": fusion_input.acute_symptom_score,
            "metadata_contextual_risk_score": fusion_input.metadata_contextual_risk_score,
            "hard_escalation": fusion_input.acute_symptom_hard_escalation,
            "fused_score": base_result.fused_score,
            "risk_band": risk_band,
            "normalized_weights_used": json.dumps(
                base_result.normalized_weights_used,
                sort_keys=True,
            ),
            "modality_contributions": json.dumps(
                base_result.modality_contributions,
                sort_keys=True,
            ),
            "warnings": json.dumps(base_result.warnings),
            "expected_min_band": scenario["expected_min_band"],
        }
        row["passes_expectation"] = (
            RISK_ORDER[row["risk_band"]] >= RISK_ORDER[row["expected_min_band"]]
        )
        rows.append(row)

    return pd.DataFrame(rows)


def print_configuration_summary(audit_df: pd.DataFrame) -> None:
    grouped = audit_df.groupby(
        ["config_name", "threshold_name", "moderate_threshold", "high_threshold"],
        sort=False,
    )

    for group_key, group_df in grouped:
        config_name, threshold_name, moderate_threshold, high_threshold = group_key
        failing = group_df.loc[
            ~group_df["passes_expectation"],
            ["scenario_name", "risk_band", "expected_min_band"],
        ]
        metadata_cases_low = bool(
            (
                group_df.loc[
                    group_df["scenario_name"].isin(
                        ["metadata_high_only", "metadata_high_only_acute_low"]
                    ),
                    "risk_band",
                ]
                == "LOW"
            ).all()
        )
        urgent_cases_urgent = bool(
            (
                group_df.loc[
                    group_df["scenario_name"].str.startswith("urgent_hard_escalation"),
                    "risk_band",
                ]
                == "URGENT"
            ).all()
        )
        borderline_snapshot = group_df.loc[
            group_df["scenario_name"].isin(
                [
                    "speech_very_high_symptoms_moderate",
                    "face_high_speech_moderate",
                    "face_speech_moderate_symptoms_mild",
                ]
            ),
            ["scenario_name", "fused_score", "risk_band"],
        ]

        print(
            f"{config_name} + {threshold_name} "
            f"(moderate={moderate_threshold:.2f}, high={high_threshold:.2f})"
        )
        print(
            "  pass_count="
            f"{int(group_df['passes_expectation'].sum())}/{len(group_df)}"
        )
        print(f"  metadata_only_cases_remain_low={metadata_cases_low}")
        print(f"  urgent_cases_remain_urgent={urgent_cases_urgent}")
        if failing.empty:
            print("  failing_scenarios=none")
        else:
            print(
                "  failing_scenarios="
                + ", ".join(
                    f"{row.scenario_name} ({row.risk_band} < {row.expected_min_band})"
                    for row in failing.itertuples()
                )
            )
        print("  borderline_snapshot:")
        for row in borderline_snapshot.itertuples():
            print(
                f"    {row.scenario_name}: "
                f"score={row.fused_score:.3f}, band={row.risk_band}"
            )


def main() -> int:
    scenarios = build_scenarios()
    audit_frames = []

    for (config_name, weights), (threshold_name, thresholds) in itertools.product(
        WEIGHT_CONFIGS.items(),
        THRESHOLD_CONFIGS.items(),
    ):
        audit_frames.append(
            analyze_configuration(
                config_name=config_name,
                weights=weights,
                threshold_name=threshold_name,
                moderate_threshold=thresholds["moderate_threshold"],
                high_threshold=thresholds["high_threshold"],
                scenarios=scenarios,
            )
        )

    audit_df = pd.concat(audit_frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(OUTPUT_PATH, index=False)

    print_configuration_summary(audit_df)
    print(f"Saved audit CSV to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
