from pathlib import Path
from uuid import uuid4

from rural_stroke_assist.capture.schemas import AssessmentInput, PatientMetadata


def create_assessment_input(
    metadata: PatientMetadata,
    face_video_path: Path | None = None,
    speech_audio_path: Path | None = None,
    face_image_path: Path | None = None,
) -> AssessmentInput:
    """
    Create a structured assessment input object.

    This represents the first system architecture step:
    gathering patient input.
    """
    return AssessmentInput(
        session_id=str(uuid4()),
        face_video_path=face_video_path,
        speech_audio_path=speech_audio_path,
        face_image_path=face_image_path,
        metadata=metadata,
    )