from pathlib import Path
from pydantic import BaseModel

from rural_stroke_assist.inference.speech_adapter import SpeechAdapter


class SpeechModuleResult(BaseModel):
    speech_abnormality_score: float | None
    confidence: float | None
    evidence: list[str]
    warnings: list[str]


def analyze_speech_audio(audio_path: Path | None) -> SpeechModuleResult:
    """Compatibility façade over the canonical artifact-backed speech adapter."""
    result = SpeechAdapter().infer(audio_path)
    return SpeechModuleResult(
        speech_abnormality_score=result.score,
        confidence=result.confidence,
        evidence=[result.label] if result.label else [],
        warnings=list(result.warnings),
    )
