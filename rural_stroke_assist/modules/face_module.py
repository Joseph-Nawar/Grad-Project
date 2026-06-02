from pathlib import Path
from pydantic import BaseModel


class FaceModuleResult(BaseModel):
    facial_asymmetry_score: float
    confidence: float
    evidence: list[str]
    warnings: list[str]


def analyze_face_image(image_path: Path) -> FaceModuleResult:
    """
    Placeholder face analysis module.

    Later, this will:
    - detect face
    - extract landmarks
    - compute left/right asymmetry
    - return evidence
    """
    return FaceModuleResult(
        facial_asymmetry_score=0.50,
        confidence=0.50,
        evidence=["Placeholder face module result."],
        warnings=["Real facial landmark analysis not implemented yet."],
    )