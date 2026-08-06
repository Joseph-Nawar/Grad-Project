from __future__ import annotations

from copy import deepcopy

from rural_stroke_assist.ui.presentation import (
    band_guidance,
    build_context_rows,
    build_modality_summaries,
    group_warnings,
    humanize_explanation,
    humanize_key,
    humanize_value,
    short_case_reference,
)


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
