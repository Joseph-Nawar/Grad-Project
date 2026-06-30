from dataclasses import dataclass

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms


@dataclass(frozen=True)
class AcuteSymptomAssessmentResult:
    """
    Output from the acute symptom assessment module.

    acute_symptom_score:
        Normalized score in [0, 1]. Higher means more acute stroke-like symptoms.

    risk_band:
        LOW, MODERATE, HIGH, or URGENT.

    evidence:
        Human-readable reasons supporting the score.

    warnings:
        Safety and interpretation warnings.
    """

    acute_symptom_score: float
    risk_band: str
    evidence: list[str]
    warnings: list[str]
    hard_escalation: bool


def _clip_score(score: float) -> float:
    """Clip score into [0, 1]."""
    return max(0.0, min(1.0, score))


def assess_acute_stroke_symptoms(
    symptoms: AcuteStrokeSymptoms,
) -> AcuteSymptomAssessmentResult:
    """
    Deterministic FAST/BE-FAST-inspired acute symptom assessment.

    This module does NOT diagnose stroke.
    It produces an auditable acute symptom concern score for fusion.

    Design logic:
    - Core FAST signs carry the highest weight.
    - BE-FAST extension signs add supporting acute evidence.
    - Any core FAST sign creates at least moderate concern.
    - Multiple core FAST signs or severe associated symptoms trigger urgent concern.
    """
    evidence: list[str] = []
    warnings: list[str] = []

    score = 0.0
    hard_escalation = False

    # Core FAST indicators.
    if symptoms.face_drooping:
        score += 0.25
        evidence.append("Face drooping/asymmetry reported.")

    if symptoms.arm_weakness:
        score += 0.25
        evidence.append("Arm weakness or arm drift reported.")

    if symptoms.speech_difficulty:
        score += 0.25
        evidence.append("Speech difficulty reported.")

    # BE-FAST / additional warning indicators.
    if symptoms.balance_or_coordination_loss:
        score += 0.10
        evidence.append("Sudden balance or coordination problem reported.")

    if symptoms.vision_disturbance:
        score += 0.10
        evidence.append("Sudden vision disturbance reported.")

    if symptoms.confusion_or_understanding_difficulty:
        score += 0.10
        evidence.append("Sudden confusion or understanding difficulty reported.")

    if symptoms.sudden_severe_headache:
        score += 0.10
        evidence.append("Sudden severe headache reported.")

    core_fast_count = sum(
        [
            symptoms.face_drooping,
            symptoms.arm_weakness,
            symptoms.speech_difficulty,
        ]
    )

    extended_count = sum(
        [
            symptoms.balance_or_coordination_loss,
            symptoms.vision_disturbance,
            symptoms.confusion_or_understanding_difficulty,
            symptoms.sudden_severe_headache,
        ]
    )

    if core_fast_count >= 2:
        hard_escalation = True
        warnings.append(
            "Multiple FAST warning signs reported. Treat as urgent screening concern."
        )

    if symptoms.symptom_onset_minutes is None:
        warnings.append("Symptom onset time is unknown.")
    elif symptoms.symptom_onset_minutes <= 270:
        evidence.append("Symptom onset is within 4.5 hours.")
        warnings.append(
            "Reported onset is within a commonly referenced emergency treatment window."
        )
        score += 0.05
    else:
        evidence.append("Symptom onset is reported beyond 4.5 hours.")

    if symptoms.symptoms_resolved:
        warnings.append(
            "Symptoms are reported as resolved. Transient symptoms may still require urgent assessment."
        )

    score = _clip_score(score)

    if hard_escalation:
        risk_band = "URGENT"
        score = max(score, 0.85)
    elif core_fast_count >= 1 and score >= 0.35:
        risk_band = "HIGH"
    elif core_fast_count >= 1 or extended_count >= 2:
        risk_band = "MODERATE"
        score = max(score, 0.35)
    elif score >= 0.70:
        risk_band = "HIGH"
    elif score >= 0.35:
        risk_band = "MODERATE"
    else:
        risk_band = "LOW"

    if not evidence:
        evidence.append("No acute stroke warning symptoms were reported.")

    warnings.append(
        "This module is a research screening component and is not a clinical diagnosis."
    )

    return AcuteSymptomAssessmentResult(
        acute_symptom_score=score,
        risk_band=risk_band,
        evidence=evidence,
        warnings=warnings,
        hard_escalation=hard_escalation,
    )