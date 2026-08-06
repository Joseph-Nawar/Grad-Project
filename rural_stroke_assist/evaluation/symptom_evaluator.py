"""Deterministic scenario coverage for the acute symptom rules."""
from __future__ import annotations
from typing import Any
from rural_stroke_assist.capture.schemas import AcuteStrokeSymptoms
from rural_stroke_assist.inference.symptom_adapter import SymptomAdapter


def evaluate_symptoms() -> dict[str, Any]:
    scenarios = {
        "none": AcuteStrokeSymptoms(),
        "face": AcuteStrokeSymptoms(face_drooping=True),
        "speech": AcuteStrokeSymptoms(speech_difficulty=True),
        "two_core": AcuteStrokeSymptoms(face_drooping=True, speech_difficulty=True, symptom_onset_minutes=30),
        "resolved": AcuteStrokeSymptoms(face_drooping=True, symptoms_resolved=True, symptom_onset_minutes=30),
        "unknown_onset": AcuteStrokeSymptoms(face_drooping=True, symptom_onset_minutes=None),
        "all_fast": AcuteStrokeSymptoms(face_drooping=True, arm_weakness=True, speech_difficulty=True, symptom_onset_minutes=10),
    }
    adapter = SymptomAdapter()
    results = {}
    for name, value in scenarios.items():
        evidence = adapter.infer(value)
        results[name] = {"score": evidence.score, "label": evidence.label, "hard_escalation": evidence.details.get("hard_escalation"), "warnings": evidence.warnings}
    return {"scenario_count": len(results), "scenarios": results, "rule_verification_only": True}
