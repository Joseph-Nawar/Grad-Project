"""Production construction for an explicit assessment runtime profile."""

from __future__ import annotations

import os

from rural_stroke_assist.assessment.fusion_strategy import CanonicalLateFusionStrategy
from rural_stroke_assist.assessment.service import AssessmentService
from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.metadata_adapter import MetadataAdapter
from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
from rural_stroke_assist.inference.symptom_adapter import SymptomAdapter
from rural_stroke_assist.inference.runners import OriginalFaceRunner, OriginalTabularRunner
from rural_stroke_assist.inference.registry import load_baseline_registry


def create_default_assessment_service(profile: str | None = None) -> AssessmentService:
    selected = profile or os.environ.get("RURALSTROKE_EDGE_RUNTIME_PROFILE", "original")
    if selected not in {"original", "optimized"}:
        raise ValueError("RURALSTROKE_EDGE_RUNTIME_PROFILE must be original or optimized")
    if selected == "optimized":
        from rural_stroke_assist.inference.edge_registry import build_optimized_runners

        runners = build_optimized_runners()
    else:
        registry = load_baseline_registry()
        runners = {
            "face": OriginalFaceRunner(registry.path_for("face")),
            "speech": OriginalTabularRunner(registry.path_for("speech")),
            "metadata_context": OriginalTabularRunner(registry.path_for("metadata_context")),
        }
    return AssessmentService(
        adapters={
            "face": FaceAdapter(runner=runners["face"]),
            "speech": SpeechAdapter(runner=runners["speech"]),
            "metadata_context": MetadataAdapter(runner=runners["metadata_context"]),
            "acute_symptoms": SymptomAdapter(),
        },
        fusion_strategy=CanonicalLateFusionStrategy(),
        runtime_profile=selected,
    )
