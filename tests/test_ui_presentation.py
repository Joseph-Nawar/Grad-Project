from __future__ import annotations

from copy import deepcopy

from rural_stroke_assist.ui.common import render_assessment_result
from rural_stroke_assist.ui.presentation import (
    band_guidance,
    build_context_rows,
    build_modality_summaries,
    group_warnings,
    humanize_explanation,
    humanize_key,
    humanize_value,
    model_label,
    profile_label,
    runtime_profile_for_result,
    short_case_reference,
)


class _CaptureColumn:
    def __init__(self, app: "_CaptureStreamlit") -> None:
        self.app = app

    def __enter__(self) -> "_CaptureStreamlit":
        return self.app

    def __exit__(self, *_: object) -> bool:
        return False


class _CaptureStreamlit:
    def __init__(self) -> None:
        self.markdown_calls: list[str] = []

    def markdown(self, value: str, **_: object) -> None:
        self.markdown_calls.append(value)

    def columns(self, count: int, **_: object) -> list[_CaptureColumn]:
        return [_CaptureColumn(self) for _ in range(count)]

    def caption(self, _: str) -> None:
        pass


def result_snapshot() -> dict:
    return {
        "warnings": [
            "This is a research screening estimate, not a clinical diagnosis.",
            "This is a research screening estimate, not a clinical diagnosis.",
            "Face score is visual proxy evidence from a public dataset.",
            "Audio has clipping.",
        ],
        "modality_executions": {
            "face": {
                "available": True,
                "score": 0.2,
                "score_semantics": "visual_proxy_evidence",
                "quality_status": "WARN",
                "quality_findings": [{"code": "blur", "message": "Image is blurry.", "status": "WARN"}],
                "failure": None,
            },
            "metadata_context": {
                "available": True,
                "score": 0.8,
                "score_semantics": "contextual_risk_evidence",
                "quality_status": "PASS",
                "quality_findings": [],
                "failure": None,
            },
            "speech": {
                "available": False,
                "score": None,
                "score_semantics": "dysarthria_proxy_evidence",
                "quality_status": "UNAVAILABLE",
                "quality_findings": [],
                "failure": {"error_type": "FeatureContractError", "message": "bad feature"},
            },
        },
        "fusion": {
            "risk_band": "LOW",
            "modality_contributions": {"speech": 0.1, "metadata_context": 0.2},
        },
        "explanations": [{"code": "strongest_contributions", "message": "Strongest evidence contributions: speech, metadata_context."}],
    }


def test_humanization_and_unknown_fallback() -> None:
    assert humanize_key("avg_glucose_level") == "Average glucose"
    assert humanize_key("metadata_context") == "Background risk"
    assert humanize_key("unexpected_field") == "Unexpected field"
    assert humanize_value(False) == "No"
    assert humanize_value(None) == "Unknown"
    assert humanize_value("Govt_job") == "Government employment"


def test_short_reference_is_stable_and_does_not_change_identifier() -> None:
    identifier = "2c809032-1234-5678-9abc-def012345678"
    assert short_case_reference(identifier) == "Case 2C809032"
    assert identifier == "2c809032-1234-5678-9abc-def012345678"


def test_warning_groups_deduplicate_and_separate_quality() -> None:
    groups = group_warnings(result_snapshot())
    assert groups.assessment_limitations == ("This is a research screening result, not a clinical diagnosis or calibrated clinical probability.",)
    assert groups.input_quality_findings == ("Face: Image is blurry.", "Audio has clipping.")
    assert "Speech: the speech adapter failed (FeatureContractError)." in groups.modality_limitations
    assert groups.technical_warnings == ("This is a research screening estimate, not a clinical diagnosis.", "Face score is visual proxy evidence from a public dataset.", "Audio has clipping.")


def test_low_band_guidance_contains_no_rule_out_safety_wording() -> None:
    assert "does not rule out stroke" in band_guidance("LOW")
    assert "clinical probability" not in band_guidance("LOW").lower()
    assert "specialist review" in band_guidance("HIGH").lower()


def test_modality_summaries_and_explanation_are_human_readable() -> None:
    summaries = build_modality_summaries(result_snapshot())
    speech = next(item for item in summaries if item.name == "Speech")
    assert speech.availability == "Unavailable"
    assert "failed" in speech.interpretation
    assert "Background risk" in next(item.interpretation for item in summaries if item.name == "Background risk")
    explanation = humanize_explanation(result_snapshot()["explanations"][0], result_snapshot())
    assert "speech" in explanation.lower() and "dysarthria proxy" in explanation.lower()
    assert "background risk" in explanation.lower()


def test_context_rows_omit_internal_fields_and_humanize_values_without_mutation() -> None:
    payload = {
        "session_id": "secret-session",
        "face_image_path": "cases/id/face.jpg",
        "metadata": {"avg_glucose_level": 100, "work_type": "Govt_job", "hypertension": False},
        "acute_symptoms": {"face_drooping": False, "symptom_onset_minutes": None},
    }
    before = deepcopy(payload)
    context, symptoms = build_context_rows(payload)
    assert [(row.label, row.value) for row in context] == [("Average glucose", "100"), ("Work type", "Government employment"), ("Hypertension", "No")]
    assert [(row.label, row.value) for row in symptoms] == [("Face drooping", "No"), ("Symptom onset minutes", "Unknown")]
    assert all("session" not in row.label.lower() for row in context + symptoms)
    assert all("path" not in row.label.lower() for row in context + symptoms)
    assert payload == before


def test_runtime_profile_and_model_labels_follow_actual_case_provenance() -> None:
    reference = {
        "provenance": ["profile=pretrained_reference"],
        "modality_executions": {
            "face": {"provenance": "artifact=...mobilenetv2..."},
            "speech": {"provenance": "model_id=ntu-spml/distilhubert", "details": {"upstream_model_id": "ntu-spml/distilhubert"}},
            "metadata_context": {"provenance": "model_id=TabPFN v2", "details": {"model_version": "ModelVersion.V2"}},
            "acute_symptoms": {},
        },
    }
    assert runtime_profile_for_result(reference) == "pretrained_reference"
    assert profile_label("pretrained_reference") == "Research reference profile"
    assert model_label("face", reference["modality_executions"]["face"], "pretrained_reference") == "ImageNet-pretrained MobileNetV2"
    assert model_label("speech", reference["modality_executions"]["speech"], "pretrained_reference") == "DistilHuBERT"
    assert model_label("metadata_context", reference["modality_executions"]["metadata_context"], "pretrained_reference") == "TabPFN v2"
    assert model_label("acute_symptoms", {}, "pretrained_reference") == "Deterministic symptom rules"


def test_deployment_profile_model_names_and_edge_profile_are_not_mislabeled() -> None:
    result = {"provenance": ["profile=original"]}
    assert runtime_profile_for_result(result) == "original"
    assert profile_label("original") == "Deployment-oriented profile · original runtime"
    assert model_label("speech", {"provenance": "trial_001_mfcc_random_forest/model.pkl"}, "original") == "MFCC + Random Forest"
    assert model_label("metadata_context", {"provenance": "mvp_metadata_risk_model.pkl"}, "original") == "Logistic Regression"
    assert runtime_profile_for_result({}, {"profile": "optimized"}) == "optimized"
    assert profile_label("optimized") == "Deployment-oriented profile · optimized runtime"


def test_urgent_partial_result_keeps_assessment_completeness_visible() -> None:
    result = {
        "status": "partial",
        "provenance": ["profile=original"],
        "fusion": {
            "risk_band": "URGENT",
            "evidence_score": 0.85,
            "modality_contributions": {"acute_symptoms": 0.85},
            "normalized_weights_used": {"acute_symptoms": 1.0},
        },
        "modality_executions": {
            "acute_symptoms": {
                "available": True,
                "score": 0.85,
                "quality_status": "PASS",
                "score_semantics": "deterministic_acute_symptom_evidence",
                "details": {"hard_escalation": True, "evidence": ["Face drooping"]},
                "provenance": "acute_symptom_module.py",
            }
        },
    }
    app = _CaptureStreamlit()

    render_assessment_result(app, result, include_limitations=False)

    hero = app.markdown_calls[0]
    assert "rsa-status-pill urgent'>URGENT</span>" in hero
    assert "rsa-status-pill partial'>PARTIAL</span>" in hero
