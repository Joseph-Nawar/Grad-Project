"""Pure deterministic explanation construction."""

from __future__ import annotations

from typing import Mapping

from rural_stroke_assist.assessment.contracts import ExplanationItem, FusionEvidence, ModalityExecution


def build_explanations(
    executions: Mapping[str, ModalityExecution], fusion: FusionEvidence | None,
) -> tuple[ExplanationItem, ...]:
    items: list[ExplanationItem] = []
    if fusion is None:
        items.append(ExplanationItem("insufficient_evidence", "No modality produced usable evidence; fusion was skipped."))
    else:
        items.append(ExplanationItem("evidence_band", f"Final research evidence band: {fusion.risk_band}."))
        strongest = sorted(fusion.modality_contributions.items(), key=lambda pair: (-pair[1], pair[0]))
        if strongest:
            names = ", ".join(name for name, _ in strongest[:2])
            items.append(ExplanationItem("strongest_contributions", f"Strongest evidence contributions: {names}."))
        if len(fusion.normalized_weights_used) < 4:
            items.append(ExplanationItem("renormalized_weights", "Available modality weights were renormalized after missing or failed inputs."))
        if any("hard escalation" in warning.lower() for warning in fusion.warnings):
            items.append(ExplanationItem("urgent_override", "The acute symptom rules triggered the canonical urgent override."))
        if any("corroboration floor" in note.lower() for note in fusion.evidence_notes):
            items.append(ExplanationItem("corroboration_floor", "The canonical acute-evidence corroboration floor was applied."))
    for name in ("face", "speech", "metadata_context", "acute_symptoms"):
        execution = executions[name]
        if execution.failure is not None:
            items.append(ExplanationItem("modality_failure", f"{name} failed with a known adapter error; its evidence was excluded."))
        elif not execution.evidence.available:
            items.append(ExplanationItem("modality_missing", f"{name} was unavailable and contributed no score."))
        if execution.evidence.quality_status.value in {"WARN", "REJECT"}:
            items.append(ExplanationItem("quality_concern", f"{name} has input-quality status {execution.evidence.quality_status.value}."))
    if executions["metadata_context"].evidence.available:
        items.append(ExplanationItem("contextual_metadata", "Metadata is contextual/background evidence only, not acute stroke evidence."))
    items.append(ExplanationItem("non_diagnostic", "This research screening result is not a clinical diagnosis or calibrated probability."))
    return tuple(items)
