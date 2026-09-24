"""Production construction for an explicit assessment runtime profile."""

from __future__ import annotations

import os

from rural_stroke_assist.assessment.fusion_strategy import CanonicalLateFusionStrategy
from rural_stroke_assist.assessment.service import AssessmentService
from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.metadata_adapter import MetadataAdapter
from rural_stroke_assist.inference.registry import load_baseline_registry
from rural_stroke_assist.inference.runners import OriginalFaceRunner, OriginalTabularRunner
from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
from rural_stroke_assist.inference.symptom_adapter import SymptomAdapter


def create_default_assessment_service(profile: str | None = None) -> AssessmentService:
    selected = profile or os.environ.get("RURALSTROKE_EDGE_RUNTIME_PROFILE", "original")
    if selected not in {"original", "optimized", "pretrained_reference"}:
        raise ValueError(
            "RURALSTROKE_EDGE_RUNTIME_PROFILE must be original, optimized, "
            "or explicitly pretrained_reference"
        )
    if selected == "pretrained_reference":
        from rural_stroke_assist.inference.reference_runtime import (
            build_pretrained_reference_adapters,
            build_pretrained_reference_runtime,
        )

        reference_registry, runners = build_pretrained_reference_runtime()
        adapters = build_pretrained_reference_adapters(reference_registry, runners)
    elif selected == "optimized":
        from rural_stroke_assist.inference.edge_registry import build_optimized_runners

        runners = build_optimized_runners()
        adapters = {
            "face": FaceAdapter(runner=runners["face"]),
            "speech": SpeechAdapter(runner=runners["speech"]),
            "metadata_context": MetadataAdapter(runner=runners["metadata_context"]),
            "acute_symptoms": SymptomAdapter(),
        }
    else:
        registry = load_baseline_registry()
        runners = {
            "face": OriginalFaceRunner(registry.path_for("face")),
            "speech": OriginalTabularRunner(registry.path_for("speech")),
            "metadata_context": OriginalTabularRunner(registry.path_for("metadata_context")),
        }
        adapters = {
            "face": FaceAdapter(runner=runners["face"]),
            "speech": SpeechAdapter(runner=runners["speech"]),
            "metadata_context": MetadataAdapter(runner=runners["metadata_context"]),
            "acute_symptoms": SymptomAdapter(),
        }
    return AssessmentService(
        adapters=adapters,
        fusion_strategy=CanonicalLateFusionStrategy(),
        runtime_profile=selected,
    )
