"""Pluggable, conservative face-image quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from rural_stroke_assist.inference.contracts import QualityFinding, QualityStatus


@dataclass(frozen=True)
class FaceQualityAssessment:
    status: QualityStatus
    findings: tuple[QualityFinding, ...]


class FaceQualityAssessor(Protocol):
    def __call__(self, image: Image.Image) -> FaceQualityAssessment:
        ...


class OpenCVFaceQualityAssessor:
    """Haar-based presence/count/size checks with explicit pose non-assessment."""

    def __init__(
        self,
        *,
        min_face_size_px: int = 80,
        min_image_dimension_px: int = 160,
        blur_variance_threshold: float = 100.0,
        min_luminance: float = 35.0,
        max_luminance: float = 225.0,
    ) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError(f"Unable to load face detector cascade: {cascade_path}")
        self.min_face_size_px = min_face_size_px
        self.min_image_dimension_px = min_image_dimension_px
        self.blur_variance_threshold = blur_variance_threshold
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance

    def __call__(self, image: Image.Image) -> FaceQualityAssessment:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        findings: list[QualityFinding] = [
            QualityFinding(
                code="pose_not_assessed",
                message="Pose was not assessed because no landmark backend is configured.",
                status=QualityStatus.NOT_ASSESSED,
            )
        ]

        if min(rgb.shape[:2]) < self.min_image_dimension_px:
            findings.append(QualityFinding("dimensions", "Image dimensions are below the minimum quality threshold.", QualityStatus.REJECT))

        if len(faces) == 0:
            findings.append(QualityFinding("no_face", "No face detected.", QualityStatus.REJECT))
            return FaceQualityAssessment(QualityStatus.REJECT, tuple(findings))
        if len(faces) > 1:
            findings.append(QualityFinding("multiple_faces", "Multiple faces detected; input is ambiguous.", QualityStatus.REJECT))
            return FaceQualityAssessment(QualityStatus.REJECT, tuple(findings))

        x, y, width, height = faces[0]
        if min(width, height) < self.min_face_size_px:
            findings.append(QualityFinding("face_too_small", "Detected face is below the minimum size.", QualityStatus.REJECT))

        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < self.blur_variance_threshold:
            findings.append(QualityFinding("blur", f"Image sharpness is low (variance={blur:.1f}).", QualityStatus.WARN))

        luminance = float(gray.mean())
        if luminance < self.min_luminance or luminance > self.max_luminance:
            findings.append(QualityFinding("lighting", f"Mean luminance is outside the supported range ({luminance:.1f}).", QualityStatus.WARN))

        status = QualityStatus.REJECT if any(item.status is QualityStatus.REJECT for item in findings) else QualityStatus.WARN if any(item.status is QualityStatus.WARN for item in findings) else QualityStatus.PASS
        return FaceQualityAssessment(status=status, findings=tuple(findings))
