"""Construction helpers for the explicit pretrained reference profile."""

from __future__ import annotations

from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.reference_adapters import (
    ReferenceMetadataAdapter,
    ReferenceSpeechAdapter,
)
from rural_stroke_assist.inference.reference_registry import ReferenceProfileRegistry
from rural_stroke_assist.inference.reference_runners import DistilHuBERTRunner, TabPFNRunner
from rural_stroke_assist.inference.runners import OriginalFaceRunner
from rural_stroke_assist.inference.symptom_adapter import SymptomAdapter


def build_pretrained_reference_runtime() -> tuple[ReferenceProfileRegistry, dict[str, object]]:
    """Build all pinned reference runners after validating local resources.

    This is intentionally a separate opt-in construction path.  The caller must
    choose ``pretrained_reference``; the default/edge factory paths do not call
    this function.
    """

    registry = ReferenceProfileRegistry.from_file()
    required = {"face", "speech", "metadata_context", "acute_symptoms"}
    declared = set(registry.data.get("modalities", {}))
    missing = required - declared
    if missing:
        raise ValueError(
            "pretrained_reference registry is incomplete; missing modality(s): "
            + ", ".join(sorted(missing))
        )
    registry.validate_project_artifacts()
    registry.validate_external_resources(local_only=True)
    runners = {
        "face": OriginalFaceRunner(registry.path_for("face")),
        "speech": DistilHuBERTRunner(registry),
        "metadata_context": TabPFNRunner(registry),
    }
    return registry, runners


def build_pretrained_reference_adapters(
    registry: ReferenceProfileRegistry, runners: dict[str, object]
) -> dict[str, object]:
    """Bind the reference runners to the existing adapter/fusion service."""

    return {
        "face": FaceAdapter(registry=registry, runner=runners["face"]),
        "speech": ReferenceSpeechAdapter(registry=registry, runner=runners["speech"]),
        "metadata_context": ReferenceMetadataAdapter(
            registry=registry, runner=runners["metadata_context"]
        ),
        "acute_symptoms": SymptomAdapter(registry=registry),
    }
