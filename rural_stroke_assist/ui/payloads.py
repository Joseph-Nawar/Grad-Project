"""Path-free payload builders shared by the Streamlit client and contract tests."""

from __future__ import annotations

from typing import Any

from rural_stroke_assist.capture.schemas import AssessmentInput


def build_case_assessment_input_payload(
    assessment_input: AssessmentInput,
) -> dict[str, Any]:
    """Return only the fields accepted by the strict case-create draft schema."""

    return assessment_input.model_dump(
        mode="json",
        by_alias=True,
        include={"session_id", "metadata", "acute_symptoms"},
        exclude_none=True,
    )
