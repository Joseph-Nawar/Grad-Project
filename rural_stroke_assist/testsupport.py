"""Small deterministic fixture helpers used by UI tests."""

from rural_stroke_assist.assessment.contracts import AssessmentResult, AssessmentStatus, AssessmentTimings, FusionEvidence, ModalityExecution
from rural_stroke_assist.inference.contracts import ModalityEvidence


def fake_assessment_result() -> AssessmentResult:
    evidence = ModalityEvidence(modality="acute_symptoms", available=True, score=0.85, score_semantics="deterministic_acute_symptom_evidence", label="URGENT", provenance="rural_stroke_assist/modules/acute_symptom_module.py")
    return AssessmentResult(status=AssessmentStatus.PARTIAL, modality_executions={"acute_symptoms": ModalityExecution("acute_symptoms", evidence, 1)}, fusion=FusionEvidence(0.85, "URGENT", {"acute_symptoms": 0.85}, {"acute_symptoms": 1.0}, (), "rural_stroke_assist/modules/fusion_module.py"), explanations=(), warnings=(), timings=AssessmentTimings({"acute_symptoms": 1}, 1, 1, 3), provenance=("source",))
