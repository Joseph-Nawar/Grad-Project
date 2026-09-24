from __future__ import annotations

from pathlib import Path

import pytest

from rural_stroke_assist.assessment.contracts import AssessmentStatus
from rural_stroke_assist.assessment.fusion_strategy import CanonicalLateFusionStrategy
from rural_stroke_assist.assessment.service import AssessmentService
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.contracts import ModalityEvidence
from rural_stroke_assist.inference.exceptions import ArtifactConfigurationError


SEMANTICS = {
    "face": "visual_proxy_evidence",
    "speech": "dysarthria_proxy_evidence",
    "metadata_context": "contextual_risk_evidence",
    "acute_symptoms": "deterministic_acute_symptom_evidence",
}


class FixedAdapter:
    def __init__(self, result: ModalityEvidence):
        self.result = result

    def infer(self, _value: object) -> ModalityEvidence:
        return self.result


def reference_evidence(modality: str, score: float) -> ModalityEvidence:
    return ModalityEvidence(
        modality=modality,
        available=True,
        score=score,
        score_semantics=SEMANTICS[modality],
        label="reference",
        provenance=f"profile=pretrained_reference; modality={modality}",
        details={"profile": "pretrained_reference"},
    )


def service() -> AssessmentService:
    scores = {"face": 0.7, "speech": 0.8, "metadata_context": 0.2, "acute_symptoms": 0.1}
    return AssessmentService(
        adapters={
            name: FixedAdapter(reference_evidence(name, score))
            for name, score in scores.items()
        },
        fusion_strategy=CanonicalLateFusionStrategy(),
        runtime_profile="pretrained_reference",
    )


def complete_input() -> AssessmentInput:
    return AssessmentInput(
        session_id="reference-contract",
        face_image_path=Path("face.png"),
        speech_audio_path=Path("speech.wav"),
        metadata={
            "age": 68,
            "hypertension": 1,
            "heart_disease": 0,
            "avg_glucose_level": 130.0,
            "bmi": 25.0,
            "gender": "Male",
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "smoking_status": "never smoked",
        },
        acute_symptoms=AcuteStrokeSymptoms(),
    )


def test_reference_evidence_enters_unchanged_canonical_late_fusion() -> None:
    result = service().assess(complete_input())

    assert result.status is AssessmentStatus.COMPLETE
    assert result.fusion is not None
    assert result.fusion.evidence_score == pytest.approx(0.53)
    assert result.fusion.normalized_weights_used == pytest.approx(
        {
            "face": 0.35,
            "speech": 0.3,
            "metadata_context": 0.1,
            "acute_symptoms": 0.25,
        }
    )


def test_reference_missing_speech_and_metadata_remain_a_valid_partial_assessment() -> None:
    value = complete_input()
    value.speech_audio_path = None
    value.metadata = None

    result = service().assess(value)

    assert result.status is AssessmentStatus.PARTIAL
    assert result.fusion is not None
    assert result.modality_executions["speech"].evidence.available is False
    assert result.modality_executions["metadata_context"].evidence.available is False
    assert result.fusion.normalized_weights_used == {
        "face": 0.35 / 0.6,
        "acute_symptoms": 0.25 / 0.6,
    }


def test_reference_acute_hard_escalation_is_unchanged() -> None:
    value = complete_input()
    value.acute_symptoms = AcuteStrokeSymptoms(face_drooping=True, arm_weakness=True)
    assessment_service = service()
    assessment_service.adapters["acute_symptoms"] = FixedAdapter(
        ModalityEvidence(
            modality="acute_symptoms",
            available=True,
            score=0.85,
            score_semantics=SEMANTICS["acute_symptoms"],
            label="URGENT",
            provenance="profile=pretrained_reference; modality=acute_symptoms",
            details={"profile": "pretrained_reference", "hard_escalation": True},
        )
    )
    result = assessment_service.assess(value)

    assert result.fusion is not None
    assert result.fusion.risk_band == "URGENT"


def test_reference_no_usable_evidence_is_insufficient() -> None:
    result = service().assess(AssessmentInput(session_id="reference-empty"))

    assert result.status is AssessmentStatus.INSUFFICIENT_EVIDENCE
    assert result.fusion is None


def test_reference_builder_does_not_fallback_when_external_resources_fail(monkeypatch) -> None:
    import rural_stroke_assist.inference.reference_runtime as reference_runtime

    class Registry:
        data = {
            "modalities": {
                "face": {},
                "speech": {},
                "metadata_context": {},
                "acute_symptoms": {},
            }
        }

        def validate_project_artifacts(self):
            return None

        def validate_external_resources(self, *, local_only):
            assert local_only is True
            raise ArtifactConfigurationError("prepared DistilHuBERT resource is missing")

    monkeypatch.setattr(
        reference_runtime.ReferenceProfileRegistry,
        "from_file",
        staticmethod(lambda: Registry()),
    )

    try:
        reference_runtime.build_pretrained_reference_runtime()
    except ArtifactConfigurationError as exc:
        assert "prepared" in str(exc)
    else:
        raise AssertionError("pretrained_reference unexpectedly fell back or constructed")
