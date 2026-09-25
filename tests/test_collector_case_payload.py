from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.server.api.schemas.cases import CaseCreateRequest
from rural_stroke_assist.ui.payloads import build_case_assessment_input_payload


def test_collector_case_payload_matches_strict_create_schema() -> None:
    assessment_input = AssessmentInput(
        session_id="collector-session",
        face_video_path="unused-video.mp4",
        speech_audio_path=None,
        face_image_path="unused-face.jpg",
        metadata=MetadataInput(
            age=60,
            hypertension=0,
            heart_disease=0,
            avg_glucose_level=100.0,
            bmi=25.0,
            gender="Female",
            ever_married="No",
            work_type="Private",
            Residence_type="Rural",
            smoking_status="Unknown",
        ),
        acute_symptoms=AcuteStrokeSymptoms(),
    )
    payload = {
        "id": str(uuid4()),
        "facility": "Local health post",
        "patient_code": None,
        "assessment_input": build_case_assessment_input_payload(assessment_input),
    }

    parsed = CaseCreateRequest.model_validate(payload)

    assert set(payload["assessment_input"]) == {
        "session_id",
        "metadata",
        "acute_symptoms",
    }
    assert parsed.assessment_input.session_id == "collector-session"
    assert parsed.assessment_input.metadata is not None
    assert parsed.assessment_input.metadata.residence_type == "Rural"

    invalid_payload = {
        **payload,
        "assessment_input": {
            **payload["assessment_input"],
            "face_video_path": None,
        },
    }
    with pytest.raises(ValidationError):
        CaseCreateRequest.model_validate(invalid_payload)
