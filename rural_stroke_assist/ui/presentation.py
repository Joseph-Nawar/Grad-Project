"""Pure presentation view models and humanization helpers.

This module intentionally knows nothing about Streamlit, persistence, or model
execution. It converts stored case/assessment dictionaries into immutable views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DisplayRow:
    label: str
    value: str


@dataclass(frozen=True)
class ModalitySummaryView:
    name: str
    availability: str
    quality_state: str
    interpretation: str
    failure_reason: str | None = None


@dataclass(frozen=True)
class WarningGroups:
    assessment_limitations: tuple[str, ...]
    modality_limitations: tuple[str, ...]
    input_quality_findings: tuple[str, ...]
    technical_warnings: tuple[str, ...]


_KEY_LABELS = {
    "avg_glucose_level": "Average glucose",
    "metadata_context": "Background risk",
    "acute_symptoms": "Acute symptoms",
    "face_drooping": "Face drooping",
    "arm_weakness": "Arm weakness",
    "speech_difficulty": "Speech difficulty",
    "balance_or_coordination_loss": "Balance or coordination loss",
    "vision_disturbance": "Vision disturbance",
    "sudden_severe_headache": "Sudden severe headache",
    "confusion_or_understanding_difficulty": "Confusion or understanding difficulty",
    "symptom_onset_minutes": "Symptom onset minutes",
    "symptoms_resolved": "Symptoms resolved",
    "ever_married": "Ever married",
    "heart_disease": "Heart disease",
    "hypertension": "Hypertension",
    "work_type": "Work type",
    "residence_type": "Residence type",
    "Residence_type": "Residence type",
    "smoking_status": "Smoking status",
    "gender": "Gender",
    "bmi": "BMI",
    "age": "Age",
}

_VALUE_LABELS = {
    "Govt_job": "Government employment",
    "Never_worked": "Never worked",
    "Self-employed": "Self-employed",
    "Private": "Private employment",
    "children": "Child",
    "formerly smoked": "Former smoker",
    "never smoked": "Never smoked",
    "no_stroke": "No stroke class",
    "dysarthric": "Dysarthria proxy evidence",
    "control": "Control speech evidence",
}

_SEMANTIC_LABELS = {
    "visual_proxy_evidence": "visual proxy evidence",
    "dysarthria_proxy_evidence": "dysarthria proxy evidence",
    "contextual_risk_evidence": "background contextual risk evidence",
    "deterministic_acute_symptom_evidence": "deterministic acute symptom evidence",
}

_MODALITY_LABELS = {
    "face": "Face",
    "speech": "Speech",
    "metadata_context": "Background risk",
    "acute_symptoms": "Acute symptoms",
}


def humanize_key(value: object) -> str:
    raw = str(value)
    if raw in _KEY_LABELS:
        return _KEY_LABELS[raw]
    words = raw.replace("_", " ").replace("-", " ").strip().split()
    return (words[0].capitalize() + (" " + " ".join(words[1:]) if len(words) > 1 else "")) if words else "Unknown field"


def humanize_value(value: object) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    raw = str(value)
    if raw in _VALUE_LABELS:
        return _VALUE_LABELS[raw]
    if raw in {"True", "False"}:
        return "Yes" if raw == "True" else "No"
    return raw.replace("_", " ").replace("-", " ").strip().title() or "Unknown"


def short_case_reference(case_id: str) -> str:
    return f"Case {str(case_id).replace('-', '').upper()[:8]}"


def band_guidance(band: object) -> str:
    normalized = str(band or "INSUFFICIENT_EVIDENCE").upper()
    guidance = {
        "LOW": "Low model evidence does not rule out stroke. Follow symptom-based emergency procedures and seek specialist review whenever clinical concern remains.",
        "MODERATE": "Moderate model evidence supports prompt specialist review and symptom-based escalation when clinically indicated; it is not a diagnosis.",
        "HIGH": "High model evidence supports urgent specialist review alongside clinical assessment; it is not a diagnosis or clinical probability.",
        "URGENT": "Urgent symptom evidence supports immediate emergency procedures and specialist review; it is not a diagnosis.",
        "INSUFFICIENT_EVIDENCE": "There is not enough usable multimodal evidence. Follow symptom-based emergency procedures whenever clinical concern remains.",
    }
    return guidance.get(normalized, "Review symptoms and seek specialist guidance when clinical concern remains; this result is not a diagnosis.")


def _dedupe(items: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _modality_label(name: object) -> str:
    return _MODALITY_LABELS.get(str(name), humanize_key(name))


def group_warnings(result: Mapping[str, Any]) -> WarningGroups:
    technical = _dedupe(tuple(str(item) for item in result.get("warnings", ())))
    assessment: list[str] = []
    modality: list[str] = []
    quality: list[str] = []
    for warning in technical:
        lowered = warning.lower()
        if "diagnos" in lowered or "calibrated clinical" in lowered or "screening estimate" in lowered:
            assessment.append("This is a research screening result, not a clinical diagnosis or calibrated clinical probability.")
    for name, execution in result.get("modality_executions", {}).items():
        label = _modality_label(name)
        failure = execution.get("failure")
        if failure:
            modality.append(f"{label}: the {label.lower()} adapter failed ({failure.get('error_type', 'unknown error')}).")
        semantics = execution.get("score_semantics")
        if execution.get("available") and semantics:
            modality.append(f"{label}: { _SEMANTIC_LABELS.get(semantics, humanize_key(semantics))}; not a clinical probability.")
        for finding in execution.get("quality_findings", ()):
            message = finding.get("message") if isinstance(finding, Mapping) else str(finding)
            quality.append(f"{label}: {message}")
    for warning in technical:
        lowered = warning.lower()
        if any(token in lowered for token in ("blur", "lighting", "clipping", "silence", "quality", "decode", "missing", "face image", "speech audio")):
            quality.append(warning)
    if not assessment:
        assessment.append("This is a research screening result, not a clinical diagnosis or calibrated clinical probability.")
    return WarningGroups(_dedupe(assessment), _dedupe(modality), _dedupe(quality), technical)


def build_context_rows(payload: Mapping[str, Any]) -> tuple[tuple[DisplayRow, ...], tuple[DisplayRow, ...]]:
    context: list[DisplayRow] = []
    symptoms: list[DisplayRow] = []
    for key, value in (payload.get("metadata") or {}).items():
        context.append(DisplayRow(humanize_key(key), humanize_value(value)))
    for key, value in (payload.get("acute_symptoms") or {}).items():
        symptoms.append(DisplayRow(humanize_key(key), humanize_value(value)))
    return tuple(context), tuple(symptoms)


def _interpretation(name: str, execution: Mapping[str, Any]) -> str:
    if not execution.get("available"):
        if execution.get("failure"):
            return "The adapter failed before producing usable evidence."
        return "No usable evidence was contributed."
    semantic = _SEMANTIC_LABELS.get(execution.get("score_semantics"), humanize_key(execution.get("score_semantics", "evidence")))
    return f"{_modality_label(name)}: available {semantic}; this describes branch evidence, not a clinical probability."


def build_modality_summaries(result: Mapping[str, Any]) -> tuple[ModalitySummaryView, ...]:
    summaries: list[ModalitySummaryView] = []
    for name, execution in result.get("modality_executions", {}).items():
        failure = execution.get("failure")
        failure_reason = None if not failure else failure.get("message") or failure.get("error_type")
        summaries.append(ModalitySummaryView(
            name=_modality_label(name),
            availability="Available" if execution.get("available") else "Unavailable",
            quality_state=humanize_value(execution.get("quality_status")),
            interpretation=_interpretation(name, execution),
            failure_reason=failure_reason,
        ))
    return tuple(summaries)


def humanize_explanation(item: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    code = item.get("code")
    if code == "strongest_contributions":
        contributions = (result.get("fusion") or {}).get("modality_contributions", {})
        names = sorted(contributions, key=lambda key: (-contributions[key], key))[:2]
        readable = []
        for name in names:
            semantic = _SEMANTIC_LABELS.get(next((execution.get("score_semantics") for modality, execution in result.get("modality_executions", {}).items() if modality == name), None), "branch evidence")
            readable.append(f"{_modality_label(name)} ({semantic})")
        return "Strongest contributions: " + ", ".join(readable) + "."
    message = str(item.get("message", ""))
    for internal, readable in _MODALITY_LABELS.items():
        message = message.replace(internal, readable)
    message = message.replace("evidence band", "evidence level")
    if code == "renormalized_weights":
        return "Available branch weights were renormalized after missing or failed inputs."
    if code == "urgent_override":
        return "The deterministic acute symptom safeguard triggered the urgent override."
    if code == "corroboration_floor":
        return "The acute evidence corroboration floor was applied."
    if code == "insufficient_evidence":
        return "No branch produced usable evidence, so no fused score or risk band was produced."
    return message


def runtime_profile_for_result(
    result: Mapping[str, Any], runtime_provenance: Mapping[str, Any] | None = None,
) -> str:
    """Return only the profile recorded with this assessment snapshot."""
    if runtime_provenance and runtime_provenance.get("profile"):
        return str(runtime_provenance["profile"])
    provenance = result.get("provenance", ())
    values = (provenance,) if isinstance(provenance, str) else provenance
    if isinstance(values, (list, tuple)):
        for item in values:
            value = str(item)
            if value.startswith("profile="):
                return value.partition("=")[2].split(";", 1)[0]
    return "unspecified"


def profile_label(profile: str) -> str:
    labels = {
        "pretrained_reference": "Research reference profile",
        "original": "Deployment-oriented profile · original runtime",
        "optimized": "Deployment-oriented profile · optimized runtime",
    }
    return labels.get(profile, f"Runtime profile · {humanize_value(profile)}")


def model_label(modality: str, execution: Mapping[str, Any], profile: str) -> str:
    """Summarize the model actually named by per-case provenance."""
    details = execution.get("details") or {}
    source = " ".join((
        str(execution.get("provenance", "")),
        str(details),
    )).lower()
    if modality == "acute_symptoms":
        return "Deterministic symptom rules"
    if modality == "face":
        if "mobilenetv2" in source or "stage0-face-trial-003" in source:
            return "ImageNet-pretrained MobileNetV2" if profile == "pretrained_reference" else "MobileNetV2"
        if profile == "optimized":
            return "Optimized edge visual artifact"
        if profile == "original":
            return "MobileNetV2"
        return "Visual model"
    if modality == "speech":
        if "distilhubert" in source or details.get("upstream_model_id"):
            return "DistilHuBERT"
        if "random_forest" in source or profile in {"original", "optimized"}:
            return "ONNX speech artifact" if profile == "optimized" else "MFCC + Random Forest"
        return "Speech model"
    if modality == "metadata_context":
        if "tabpfn" in source or details.get("model_version"):
            return "TabPFN v2"
        if "logistic" in source or profile in {"original", "optimized"}:
            return "Logistic Regression"
        return "Structured-data model"
    return "Model not identified"


def format_evidence_score(score: object) -> str | None:
    """Format a branch evidence score without presenting it as a probability."""
    if score is None:
        return None
    try:
        number = float(score)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return f"{number:.3f}"


def acute_escalation_triggered(result: Mapping[str, Any]) -> bool | None:
    execution = (result.get("modality_executions") or {}).get("acute_symptoms") or {}
    if not execution.get("available"):
        return None
    details = execution.get("details") or {}
    return bool(details.get("hard_escalation"))


def result_explanation_lines(result: Mapping[str, Any]) -> tuple[str, ...]:
    """Return concise, user-facing explanations already recorded by the service."""
    lines: list[str] = []
    for item in result.get("explanations", ()):
        if item.get("code") == "non_diagnostic":
            continue
        line = humanize_explanation(item, result)
        if line and line not in lines:
            lines.append(line)
    return tuple(lines)
