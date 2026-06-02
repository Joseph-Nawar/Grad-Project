from pydantic import BaseModel

from rural_stroke_assist.modules.face_module import FaceModuleResult
from rural_stroke_assist.modules.speech_module import SpeechModuleResult
from rural_stroke_assist.modules.metadata_module import MetadataModuleResult


class FusionResult(BaseModel):
    final_risk_score: float
    triage_level: str
    confidence: float
    evidence: list[str]
    warnings: list[str]


def determine_triage_level(score: float) -> str:
    """Convert numeric risk score into triage category."""
    if score >= 0.70:
        return "High concern"
    if score >= 0.40:
        return "Moderate concern"
    return "Low concern"


def fuse_results(
    face_result: FaceModuleResult,
    speech_result: SpeechModuleResult,
    metadata_result: MetadataModuleResult,
) -> FusionResult:
    """
    Fuse outputs from face, speech, and metadata modules.

    MVP approach:
    - weighted late fusion
    - face and speech are more important than metadata
    """
    face_weight = 0.45
    speech_weight = 0.35
    metadata_weight = 0.20

    final_score = (
        face_result.facial_asymmetry_score * face_weight
        + speech_result.speech_abnormality_score * speech_weight
        + metadata_result.contextual_risk_score * metadata_weight
    )

    confidence = (
        face_result.confidence * face_weight
        + speech_result.confidence * speech_weight
        + metadata_result.confidence * metadata_weight
    )

    evidence = (
        face_result.evidence
        + speech_result.evidence
        + metadata_result.evidence
    )

    warnings = (
        face_result.warnings
        + speech_result.warnings
        + metadata_result.warnings
    )

    return FusionResult(
        final_risk_score=round(final_score, 3),
        triage_level=determine_triage_level(final_score),
        confidence=round(confidence, 3),
        evidence=evidence,
        warnings=warnings,
    )