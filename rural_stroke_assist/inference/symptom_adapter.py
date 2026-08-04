"""Adapter for the existing deterministic acute symptom rules."""

from __future__ import annotations

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.inference.contracts import ModalityEvidence, QualityStatus
from rural_stroke_assist.inference.registry import BaselineRegistry, load_baseline_registry
from rural_stroke_assist.modules.acute_symptom_module import assess_acute_stroke_symptoms


class SymptomAdapter:
    def __init__(self, *, registry: BaselineRegistry | None = None) -> None:
        self.registry = registry or load_baseline_registry()

    def infer(self, input_data: AcuteStrokeSymptoms | None) -> ModalityEvidence:
        component = self.registry.component("acute_symptoms")
        if input_data is None:
            return ModalityEvidence.unavailable(
                modality="acute_symptoms", score_semantics="deterministic_acute_symptom_evidence",
                provenance=component.path, warning="No acute symptom assessment was provided.",
            )
        result = assess_acute_stroke_symptoms(input_data)
        return ModalityEvidence(
            modality="acute_symptoms", available=True, score=result.acute_symptom_score,
            score_semantics="deterministic_acute_symptom_evidence", label=result.risk_band,
            confidence=None, quality_status=QualityStatus.PASS, quality_findings=(),
            warnings=tuple(result.warnings), provenance=component.path,
            details={"hard_escalation": result.hard_escalation, "evidence": tuple(result.evidence)},
        )
