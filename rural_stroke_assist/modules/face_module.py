from pathlib import Path
from pydantic import BaseModel

from rural_stroke_assist.inference.face_adapter import FaceAdapter


class FaceModuleResult(BaseModel):
    facial_asymmetry_score: float | None
    confidence: float | None
    evidence: list[str]
    warnings: list[str]


def analyze_face_image(image_path: Path | None) -> FaceModuleResult:
    """Compatibility façade over the canonical artifact-backed face adapter."""
    result = FaceAdapter().infer(image_path)
    return FaceModuleResult(
        facial_asymmetry_score=result.score,
        confidence=result.confidence,
        evidence=[result.label] if result.label else [],
        warnings=list(result.warnings),
    )
