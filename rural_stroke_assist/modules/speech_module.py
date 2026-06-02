from pathlib import Path
from pydantic import BaseModel


class SpeechModuleResult(BaseModel):
    speech_abnormality_score: float
    confidence: float
    evidence: list[str]
    warnings: list[str]


def analyze_speech_audio(audio_path: Path | None) -> SpeechModuleResult:
    """
    Placeholder speech analysis module.

    Later, this will:
    - validate audio quality
    - extract acoustic features
    - analyze speech clarity
    """
    if audio_path is None:
        return SpeechModuleResult(
            speech_abnormality_score=0.0,
            confidence=0.0,
            evidence=[],
            warnings=["No speech audio provided."],
        )

    return SpeechModuleResult(
        speech_abnormality_score=0.50,
        confidence=0.50,
        evidence=["Placeholder speech module result."],
        warnings=["Real speech analysis not implemented yet."],
    )