"""Sequential end-to-end assessment service."""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

from pydantic import ValidationError

from rural_stroke_assist.assessment.contracts import (
    AssessmentResult,
    AssessmentStatus,
    AssessmentTimings,
    FusionEvidence,
    ModalityExecution,
    StructuredAdapterFailure,
)
from rural_stroke_assist.assessment.exceptions import (
    AssessmentExecutionError,
    AssessmentInputError,
)
from rural_stroke_assist.assessment.explanation import build_explanations
from rural_stroke_assist.assessment.fusion_strategy import FusionStrategy
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.contracts import ModalityEvidence, QualityStatus
from rural_stroke_assist.inference.exceptions import AdapterError

ORDER = ("face", "speech", "metadata_context", "acute_symptoms")
SEMANTICS = {
    "face": "visual_proxy_evidence",
    "speech": "dysarthria_proxy_evidence",
    "metadata_context": "contextual_risk_evidence",
    "acute_symptoms": "deterministic_acute_symptom_evidence",
}


class AssessmentService:
    def __init__(
        self,
        *,
        adapters: Mapping[str, Any],
        fusion_strategy: FusionStrategy,
        runtime_profile: str = "original",
    ) -> None:
        self.adapters = dict(adapters)
        self.fusion_strategy = fusion_strategy
        self.runtime_profile = runtime_profile

    @staticmethod
    def _validate_input(value: AssessmentInput | Mapping[str, Any]) -> AssessmentInput:
        if isinstance(value, AssessmentInput):
            return value
        try:
            return AssessmentInput.model_validate(value)
        except ValidationError as exc:
            raise AssessmentInputError("Assessment input validation failed.") from exc

    @staticmethod
    def _input_for(name: str, value: AssessmentInput) -> object:
        if name == "face":
            return value.face_image_path
        if name == "speech":
            return value.speech_audio_path
        if name == "metadata_context":
            return value.metadata
        return value.acute_symptoms

    def assess(self, assessment_input: AssessmentInput | Mapping[str, Any]) -> AssessmentResult:
        return self._assess(assessment_input)

    def _assess(self, assessment_input: AssessmentInput | Mapping[str, Any]) -> AssessmentResult:
        started = time.perf_counter_ns()
        validated = self._validate_input(assessment_input)
        executions: dict[str, ModalityExecution] = {}
        warnings: list[str] = []
        provenance: list[str] = [f"profile={self.runtime_profile}"]
        usable: dict[str, ModalityEvidence] = {}
        for name in ORDER:
            modality_started = time.perf_counter_ns()
            input_value = self._input_for(name, validated)
            if input_value is None:
                evidence = ModalityEvidence.unavailable(
                    modality=name,
                    score_semantics=SEMANTICS[name],
                    provenance=f"input:{name}",
                    warning=f"No {name} input was provided.",
                )
                executions[name] = ModalityExecution(
                    name, evidence, time.perf_counter_ns() - modality_started
                )
                warnings.extend(evidence.warnings)
                provenance.append(evidence.provenance)
                continue
            try:
                evidence = self.adapters[name].infer(input_value)
                if not isinstance(evidence, ModalityEvidence):
                    raise AssessmentExecutionError(
                        f"Adapter {name} returned an invalid result type."
                    )
                if evidence.modality != name:
                    raise AssessmentExecutionError(
                        f"Adapter {name} returned modality {evidence.modality}."
                    )
                if evidence.available and evidence.score is not None:
                    usable[name] = evidence
                warnings.extend(evidence.warnings)
                provenance.append(evidence.provenance)
                executions[name] = ModalityExecution(
                    name, evidence, time.perf_counter_ns() - modality_started
                )
            except AdapterError as exc:
                evidence = ModalityEvidence.unavailable(
                    modality=name,
                    score_semantics=SEMANTICS[name],
                    provenance=f"adapter:{name}",
                    warning=f"{name} adapter failed: {type(exc).__name__}.",
                    quality_status=QualityStatus.REJECT,
                )
                failure = StructuredAdapterFailure(name, type(exc).__name__, str(exc))
                executions[name] = ModalityExecution(
                    name, evidence, time.perf_counter_ns() - modality_started, failure
                )
                warnings.append(f"{name} adapter failed: {type(exc).__name__}.")
            except AssessmentExecutionError:
                raise
            except Exception as exc:
                raise AssessmentExecutionError(f"Unexpected {name} adapter error.") from exc

        fusion_started = time.perf_counter_ns()
        fusion = None
        if usable:
            try:
                fusion = self.fusion_strategy.fuse(usable)
                if not isinstance(fusion, FusionEvidence):
                    raise ValueError("Fusion strategy returned an invalid result type.")
            except (ValueError, TypeError) as exc:
                raise AssessmentExecutionError("Fusion failed validation.") from exc
        fusion_duration = time.perf_counter_ns() - fusion_started
        explanation_started = time.perf_counter_ns()
        explanations = build_explanations(executions, fusion)
        explanation_duration = time.perf_counter_ns() - explanation_started
        if fusion:
            warnings.extend(fusion.warnings)
            provenance.append(fusion.provenance)
        status = (
            AssessmentStatus.INSUFFICIENT_EVIDENCE if not usable else AssessmentStatus.COMPLETE
        )
        if usable and any(
            item.failure is not None or not item.evidence.available for item in executions.values()
        ):
            status = AssessmentStatus.PARTIAL
        return AssessmentResult(
            status=status,
            modality_executions=executions,
            fusion=fusion,
            explanations=explanations,
            warnings=tuple(dict.fromkeys(warnings)),
            timings=AssessmentTimings(
                modality_duration_ns={name: executions[name].duration_ns for name in ORDER},
                fusion_duration_ns=fusion_duration,
                explanation_duration_ns=explanation_duration,
                total_duration_ns=time.perf_counter_ns() - started,
            ),
            provenance=tuple(dict.fromkeys(provenance)),
        )
