from pydantic import BaseModel

from rural_stroke_assist.capture.schemas import PatientMetadata


class MetadataModuleResult(BaseModel):
    contextual_risk_score: float
    confidence: float
    evidence: list[str]
    warnings: list[str]


def analyze_metadata(metadata: PatientMetadata) -> MetadataModuleResult:
    """
    Legacy rule-based metadata analysis.

    This path is retained for compatibility and is not the canonical metadata
    adapter. It is not a diagnosis model.
    It only flags contextual risk factors.
    """
    score = 0.0
    evidence = []

    if metadata.age >= 60:
        score += 0.25
        evidence.append("Age is above 60.")

    if metadata.hypertension:
        score += 0.20
        evidence.append("Hypertension reported.")

    if metadata.diabetes:
        score += 0.15
        evidence.append("Diabetes reported.")

    if metadata.previous_stroke:
        score += 0.25
        evidence.append("Previous stroke reported.")

    if metadata.symptom_onset_minutes is not None and metadata.symptom_onset_minutes <= 270:
        score += 0.15
        evidence.append("Symptoms started within an urgent time window.")

    score = min(score, 1.0)

    return MetadataModuleResult(
        contextual_risk_score=score,
        confidence=0.80,
        evidence=evidence,
        warnings=[],
    )
