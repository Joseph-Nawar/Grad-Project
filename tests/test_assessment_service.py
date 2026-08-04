from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from rural_stroke_assist.assessment.contracts import AssessmentStatus, FusionEvidence
from rural_stroke_assist.assessment.exceptions import AssessmentExecutionError
from rural_stroke_assist.assessment.fusion_strategy import CanonicalLateFusionStrategy
from rural_stroke_assist.assessment.service import AssessmentService
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.contracts import ModalityEvidence, QualityStatus
from rural_stroke_assist.inference.exceptions import FeatureContractError
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.modules.fusion_module import FusionInput


SEMANTICS = {
    "face": "visual_proxy_evidence",
    "speech": "dysarthria_proxy_evidence",
    "metadata_context": "contextual_risk_evidence",
    "acute_symptoms": "deterministic_acute_symptom_evidence",
}


class FakeAdapter:
    def __init__(self, evidence: ModalityEvidence | Exception):
        self.evidence = evidence
        self.calls: list[object] = []

    def infer(self, value: object) -> ModalityEvidence:
        self.calls.append(value)
        if isinstance(self.evidence, Exception):
            raise self.evidence
        return self.evidence


def evidence(modality: str, score: float = 0.8, *, warning: str = "") -> ModalityEvidence:
    return ModalityEvidence(
        modality=modality, available=True, score=score,
        score_semantics=SEMANTICS[modality], label="test", provenance=f"models/{modality}",
        warnings=(warning,) if warning else (),
    )


def input_data() -> AssessmentInput:
    return AssessmentInput(
        session_id="test-session",
        face_image_path=Path("face.jpg"),
        speech_audio_path=Path("speech.wav"),
        metadata=MetadataInput(
            age=70, hypertension=1, heart_disease=0, avg_glucose_level=120,
            bmi=25, gender="Male", ever_married="Yes", work_type="Private",
            Residence_type="Urban", smoking_status="never smoked",
        ),
        acute_symptoms=AcuteStrokeSymptoms(face_drooping=True, arm_weakness=True),
    )


def service_with(adapters: dict[str, FakeAdapter]) -> AssessmentService:
    return AssessmentService(
        adapters=adapters,
        fusion_strategy=CanonicalLateFusionStrategy(),
    )


def test_assessment_executes_all_modalities_and_returns_complete_result() -> None:
    adapters = {name: FakeAdapter(evidence(name)) for name in SEMANTICS}
    result = service_with(adapters).assess(input_data())

    assert result.status is AssessmentStatus.COMPLETE
    assert result.fusion is not None
    assert result.fusion.evidence_score == 0.8
    assert set(result.modality_executions) == set(SEMANTICS)
    assert result.timings.total_duration_ns >= 0
    assert result.provenance


def test_missing_modalities_are_unavailable_and_remaining_weights_renormalize() -> None:
    adapters = {name: FakeAdapter(evidence(name)) for name in SEMANTICS}
    result = service_with(adapters).assess(
        AssessmentInput(session_id="partial", metadata=None, acute_symptoms=None)
    )

    assert result.status is AssessmentStatus.INSUFFICIENT_EVIDENCE
    assert result.modality_executions["face"].evidence.available is False
    assert result.modality_executions["speech"].evidence.available is False
    assert result.fusion is None


def test_known_adapter_failure_isolated_and_assessment_is_partial() -> None:
    adapters = {name: FakeAdapter(evidence(name)) for name in SEMANTICS}
    adapters["speech"] = FakeAdapter(FeatureContractError("bad features"))

    result = service_with(adapters).assess(input_data())

    assert result.status is AssessmentStatus.PARTIAL
    assert result.modality_executions["speech"].failure is not None
    assert result.fusion is not None
    assert "bad features" not in result.warnings
    assert any("speech" in item.message.lower() for item in result.explanations)


def test_unexpected_adapter_error_is_re_raised_as_assessment_execution_error() -> None:
    adapters = {name: FakeAdapter(evidence(name)) for name in SEMANTICS}
    adapters["face"] = FakeAdapter(RuntimeError("programming bug"))

    with pytest.raises(AssessmentExecutionError):
        service_with(adapters).assess(input_data())


def test_no_usable_modalities_skips_fusion() -> None:
    adapters = {name: FakeAdapter(evidence(name)) for name in SEMANTICS}
    result = service_with(adapters).assess(
        AssessmentInput(session_id="empty", metadata=None, acute_symptoms=None)
    )
    assert result.status is AssessmentStatus.INSUFFICIENT_EVIDENCE
    assert result.fusion is None


@pytest.mark.parametrize("available_name", tuple(SEMANTICS))
def test_each_single_available_modality_can_be_fused(available_name: str) -> None:
    adapters = {name: FakeAdapter(evidence(name)) for name in SEMANTICS}
    value = AssessmentInput(session_id="single", metadata=None, acute_symptoms=None)
    if available_name == "face":
        value.face_image_path = Path("face.jpg")
    elif available_name == "speech":
        value.speech_audio_path = Path("speech.wav")
    elif available_name == "metadata_context":
        value.metadata = input_data().metadata
    else:
        value.acute_symptoms = AcuteStrokeSymptoms(face_drooping=True)
    result = service_with(adapters).assess(value)
    assert result.fusion is not None
    assert result.status is AssessmentStatus.PARTIAL


def test_invalid_mapping_is_rejected_before_execution() -> None:
    with pytest.raises(Exception) as caught:
        service_with({name: FakeAdapter(evidence(name)) for name in SEMANTICS}).assess({})
    assert caught.type.__name__ == "AssessmentInputError"


def test_legacy_metadata_is_not_silently_mapped_to_canonical_contract() -> None:
    value = AssessmentInput(session_id="legacy", metadata={"age": 70, "sex": "male"})
    adapters = {name: FakeAdapter(evidence(name)) for name in SEMANTICS}
    adapters["metadata_context"] = FakeAdapter(FeatureContractError("legacy schema"))
    result = service_with(adapters).assess(value)
    assert result.status is AssessmentStatus.INSUFFICIENT_EVIDENCE
    assert result.modality_executions["metadata_context"].failure is not None


def test_warning_deduplication_and_deterministic_explanation() -> None:
    adapters = {
        name: FakeAdapter(evidence(name, warning="same warning")) for name in SEMANTICS
    }
    first = service_with(adapters).assess(input_data())
    second = service_with({name: FakeAdapter(evidence(name, warning="same warning")) for name in SEMANTICS}).assess(input_data())

    assert first.warnings.count("same warning") == 1
    assert first.explanations == second.explanations


def test_urgent_symptoms_are_machine_readable_and_explained() -> None:
    symptom = ModalityEvidence(
        modality="acute_symptoms", available=True, score=0.85,
        score_semantics=SEMANTICS["acute_symptoms"], label="URGENT",
        provenance="rural_stroke_assist/modules/acute_symptom_module.py",
        details={"hard_escalation": True},
    )
    adapters = {name: FakeAdapter(evidence(name, 0.1)) for name in SEMANTICS}
    adapters["acute_symptoms"] = FakeAdapter(symptom)
    result = service_with(adapters).assess(input_data())

    assert result.fusion is not None
    assert result.fusion.risk_band == "URGENT"
    assert any("urgent" in item.message.lower() for item in result.explanations)


def test_canonical_fusion_matches_registry_defaults_and_reports_contributions() -> None:
    strategy = CanonicalLateFusionStrategy()
    result = strategy.fuse({name: evidence(name, 0.5) for name in SEMANTICS})

    assert isinstance(result, FusionEvidence)
    assert result.evidence_score == pytest.approx(0.5)
    assert result.modality_contributions["face"] == pytest.approx(0.175)
    assert result.normalized_weights_used["metadata_context"] == pytest.approx(0.1)


def test_canonical_fusion_renormalizes_missing_modalities() -> None:
    result = CanonicalLateFusionStrategy().fuse({"speech": evidence("speech", 0.2)})
    assert result.evidence_score == pytest.approx(0.2)
    assert result.normalized_weights_used == {"speech": 1.0}


def test_canonical_fusion_applies_urgent_override_and_corroboration_floor() -> None:
    urgent = evidence("acute_symptoms", 0.85)
    urgent = ModalityEvidence(
        modality=urgent.modality, available=True, score=urgent.score,
        score_semantics=urgent.score_semantics, label="URGENT", provenance=urgent.provenance,
        details={"hard_escalation": True},
    )
    result = CanonicalLateFusionStrategy().fuse({"face": evidence("face", 0.8), "acute_symptoms": urgent})
    assert result.risk_band == "URGENT"

    floor = CanonicalLateFusionStrategy().fuse({"face": evidence("face", 0.75), "speech": evidence("speech", 0.75), "metadata_context": evidence("metadata_context", 0.0)})
    assert floor.evidence_score >= 0.5
    assert any("corroboration floor" in note.lower() for note in floor.evidence_notes)


def test_canonical_fusion_rejects_out_of_range_strategy_output() -> None:
    class BadFusion:
        def fuse(self, evidence_map):
            return FusionEvidence(
                evidence_score=1.5, risk_band="HIGH", modality_contributions={},
                normalized_weights_used={}, warnings=(), provenance="fusion",
            )

    adapters = {name: FakeAdapter(evidence(name)) for name in SEMANTICS}
    with pytest.raises(AssessmentExecutionError):
        AssessmentService(adapters=adapters, fusion_strategy=BadFusion()).assess(input_data())


def test_legacy_fusion_emits_deprecation_warning() -> None:
    from rural_stroke_assist.fusion.fusion_engine import fuse_results
    from rural_stroke_assist.modules.face_module import FaceModuleResult
    from rural_stroke_assist.modules.speech_module import SpeechModuleResult
    from rural_stroke_assist.modules.metadata_module import MetadataModuleResult

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fuse_results(
            FaceModuleResult(facial_asymmetry_score=0.1, confidence=0.9, evidence=[], warnings=[]),
            SpeechModuleResult(speech_abnormality_score=0.1, confidence=0.9, evidence=[], warnings=[]),
            MetadataModuleResult(contextual_risk_score=0.1, confidence=0.9, evidence=[], warnings=[]),
        )
    assert any(item.category is DeprecationWarning for item in caught)
