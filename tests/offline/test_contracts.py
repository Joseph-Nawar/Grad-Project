from __future__ import annotations

import json
from pathlib import Path

import pytest

from rural_stroke_assist.offline.contracts import (
    AssessmentEnvelope,
    AttachmentRef,
    OutboxEventType,
    OutboxState,
    SyncState,
    WorkflowState,
)
from rural_stroke_assist.offline.envelope import canonical_assessment_hash, canonical_json


def _envelope() -> AssessmentEnvelope:
    return AssessmentEnvelope(
        schema_version=1,
        case_id="00000000-0000-0000-0000-000000000001",
        assessment_id="00000000-0000-0000-0000-000000000002",
        request={
            "session_id": "collector-session",
            "metadata": {"age": 60, "Residence_type": "Rural"},
            "acute_symptoms": {"face_drooping": True},
            "attachments": [
                AttachmentRef(
                    attachment_id="00000000-0000-0000-0000-000000000003",
                    kind="face",
                    media_type="image/png",
                    size_bytes=3,
                    sha256="a" * 64,
                )
            ],
        },
        result={"status": "partial", "provenance": ["rural_stroke_assist/modules/fusion_module.py"]},
        provenance={
            "schema_version": 1,
            "model_ids": {"face": "stage0-face"},
            "fusion_id": "canonical-late-fusion",
        },
    )


def test_canonical_envelope_is_path_free_and_order_stable() -> None:
    envelope = _envelope()
    encoded = canonical_json(envelope)

    assert "collector-session" in encoded
    assert "/" not in json.loads(encoded)["request"].get("face_image_path", "")
    assert "absolute" not in encoded
    assert canonical_assessment_hash(envelope) == canonical_assessment_hash(
        AssessmentEnvelope.model_validate(json.loads(encoded))
    )


def test_envelope_rejects_absolute_paths_and_raw_media() -> None:
    with pytest.raises(ValueError, match="path-free"):
        AssessmentEnvelope.model_validate(
            {
                **_envelope().model_dump(mode="json"),
                "request": {"face_image_path": str(Path("C:/secret/photo.jpg"))},
            }
        )


def test_workflow_and_sync_states_are_separate_and_outbox_states_are_explicit() -> None:
    assert WorkflowState.QUEUED.value == "QUEUED"
    assert SyncState.BLOCKED_AUTH.value == "BLOCKED_AUTH"
    assert OutboxState.DEAD_LETTER.value == "DEAD_LETTER"
    assert [event.value for event in OutboxEventType] == [
        "CREATE_CASE",
        "UPLOAD_ATTACHMENT",
        "IMPORT_ASSESSMENT",
        "SUBMIT_CASE",
    ]
