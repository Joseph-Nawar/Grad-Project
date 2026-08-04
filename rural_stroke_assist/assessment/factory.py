"""Production construction for the default assessment service."""

from rural_stroke_assist.assessment.fusion_strategy import CanonicalLateFusionStrategy
from rural_stroke_assist.assessment.service import AssessmentService
from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.metadata_adapter import MetadataAdapter
from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
from rural_stroke_assist.inference.symptom_adapter import SymptomAdapter


def create_default_assessment_service() -> AssessmentService:
    return AssessmentService(
        adapters={"face": FaceAdapter(), "speech": SpeechAdapter(), "metadata_context": MetadataAdapter(), "acute_symptoms": SymptomAdapter()},
        fusion_strategy=CanonicalLateFusionStrategy(),
    )
