"""Shared clinical-research presentation components for Streamlit surfaces."""

from __future__ import annotations

from html import escape
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from rural_stroke_assist.quality.audio_quality import DefaultAudioQualityAssessor
from rural_stroke_assist.quality.face_quality import OpenCVFaceQualityAssessor
from rural_stroke_assist.ui.presentation import (
    acute_escalation_triggered,
    band_guidance,
    build_context_rows,
    build_modality_summaries,
    format_evidence_score,
    group_warnings,
    humanize_explanation,
    humanize_value,
    model_label,
    profile_label,
    result_explanation_lines,
    runtime_profile_for_result,
    short_case_reference,
)


def apply_theme(st: Any) -> None:
    """Apply the shared RuralStroke-Assist visual language."""
    st.markdown(
        """
        <style>
        :root {
          --rsa-navy:#142e46; --rsa-navy-2:#203e58; --rsa-teal:#087f83;
          --rsa-teal-soft:#e8f5f4; --rsa-amber:#a9560b; --rsa-amber-soft:#fff3df;
          --rsa-red:#a8202b; --rsa-red-soft:#fff0ef; --rsa-gray:#657383;
          --rsa-gray-soft:#eef1f4; --rsa-line:#dbe3e9; --rsa-page:#f4f7f9;
          --rsa-white:#ffffff;
        }
        html, body, [class*="stApp"] { font-family: Inter, "Segoe UI", Arial, sans-serif; color:var(--rsa-navy); }
        .stApp { background:var(--rsa-page); }
        .block-container { max-width:1720px; padding:1.15rem 2.1rem 3rem; }
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility:hidden; height:0; }
        header[data-testid="stHeader"] { height:1.8rem; background:transparent; }
        [data-testid="stVerticalBlock"] { gap:.72rem; }
        h1, h2, h3, h4, p, label { color:var(--rsa-navy); }
        h1 { letter-spacing:-.035em; }
        h2, h3 { letter-spacing:-.02em; }
        .stButton > button, .stFormSubmitButton > button {
          min-height:2.65rem; border-radius:.7rem; border:1px solid #b9c9d3;
          font-weight:650; padding:.45rem 1rem; color:var(--rsa-navy); background:#fff;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover { border-color:var(--rsa-teal); color:var(--rsa-teal); }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
          background:var(--rsa-teal); color:#fff; border-color:var(--rsa-teal);
        }
        button[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] button,
        button[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primaryFormSubmit"] button {
          background:var(--rsa-teal) !important; color:#fff !important; border-color:var(--rsa-teal) !important;
        }
        button[data-testid^="stBaseButton-primary"] * { color:#fff !important; }
        button[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primary"] button:hover,
        button[data-testid="stBaseButton-primaryFormSubmit"]:hover, [data-testid="stBaseButton-primaryFormSubmit"] button:hover {
          background:#076c70 !important; border-color:#076c70 !important;
        }
        .stButton > button:disabled, .stFormSubmitButton > button:disabled,
        button[data-testid^="stBaseButton-primary"]:disabled {
          background:#e8edf0 !important; color:#657383 !important; border-color:#d2dce3 !important;
          opacity:1 !important; cursor:not-allowed;
        }
        button:disabled * { color:#657383 !important; }
        [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked) > div:first-child {
          background-color:var(--rsa-teal) !important; border-color:var(--rsa-teal) !important;
        }
        [data-baseweb="select"] > div:focus-within {
          border-color:var(--rsa-teal) !important; box-shadow:0 0 0 .14rem rgba(8,127,131,.18) !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:var(--rsa-teal) !important; }
        [data-testid="stForm"] { background:#fff; border:1px solid var(--rsa-line); border-radius:.9rem; padding:.85rem; }
        [data-testid="stFileUploader"] button:focus { border-color:var(--rsa-teal) !important; box-shadow:0 0 0 .14rem rgba(8,127,131,.18) !important; }
        input, textarea, [data-baseweb="select"] > div { border-radius:.55rem !important; }
        [data-testid="stAlert"] { border-radius:.7rem; }
        .rsa-header { background:var(--rsa-white); border:1px solid var(--rsa-line); border-radius:1rem; padding:1.05rem 1.35rem .9rem; box-shadow:0 3px 14px rgba(24,49,68,.045); margin:.25rem 0 1rem; }
        .rsa-head-row { display:flex; justify-content:space-between; align-items:flex-start; gap:1.25rem; }
        .rsa-brand { font-size:1.65rem; line-height:1.15; font-weight:760; color:var(--rsa-navy); letter-spacing:-.035em; }
        .rsa-subtitle { margin:.35rem 0 0; color:#526577; font-size:.98rem; }
        .rsa-badges { display:flex; flex-wrap:wrap; gap:.45rem; justify-content:flex-end; }
        .rsa-badge { display:inline-flex; align-items:center; border-radius:999px; padding:.35rem .68rem; font-size:.76rem; font-weight:700; border:1px solid var(--rsa-line); background:#f6f8fa; color:var(--rsa-navy); white-space:nowrap; }
        .rsa-badge.research { color:#37506a; background:#eef3f7; }
        .rsa-badge.safety { color:#37506a; background:#f8fafb; }
        .rsa-badge.online { color:#056667; background:var(--rsa-teal-soft); border-color:#b8dedb; }
        .rsa-badge.offline { color:#8b4909; background:var(--rsa-amber-soft); border-color:#f0d2a5; }
        .rsa-workflow { display:flex; gap:.6rem; align-items:center; margin-top:.9rem; padding-top:.75rem; border-top:1px solid #edf1f4; }
        .rsa-step { display:flex; gap:.4rem; align-items:center; color:#7a8793; font-size:.82rem; font-weight:620; }
        .rsa-step-dot { display:grid; place-items:center; width:1.45rem; height:1.45rem; border-radius:50%; border:1px solid #cdd6dd; background:#f4f6f8; font-size:.72rem; }
        .rsa-step.active { color:var(--rsa-navy); }
        .rsa-step.active .rsa-step-dot { color:#fff; background:var(--rsa-teal); border-color:var(--rsa-teal); }
        .rsa-step.done { color:#087477; }
        .rsa-step.done .rsa-step-dot { color:#fff; background:#5ca6a1; border-color:#5ca6a1; }
        .rsa-step-arrow { color:#a4afb8; }
        .rsa-section-kicker { color:#607283; font-weight:730; font-size:.76rem; letter-spacing:.08em; text-transform:uppercase; margin:.1rem 0 .35rem; }
        .rsa-panel { border:1px solid var(--rsa-line); background:#fff; border-radius:.9rem; padding:1rem 1.1rem; box-shadow:0 2px 9px rgba(24,49,68,.035); }
        .rsa-result-hero { display:flex; justify-content:space-between; align-items:center; gap:1.5rem; border:1px solid #d7e0e6; border-left:8px solid var(--rsa-teal); border-radius:1rem; background:#fff; padding:1rem 1.35rem; box-shadow:0 4px 16px rgba(24,49,68,.06); }
        .rsa-result-hero.low, .rsa-result-hero.moderate, .rsa-result-hero.high { border-left-color:var(--rsa-teal); }
        .rsa-result-hero.urgent { border:2px solid #d73a43; border-left-width:10px; border-left-color:#a8202b; background:var(--rsa-red-soft); }
        .rsa-result-hero.insufficient { border-left-color:#798592; background:#f8f9fa; }
        .rsa-result-title { margin:.1rem 0 .35rem; font-size:1.62rem; line-height:1.2; font-weight:780; color:var(--rsa-navy); }
        .rsa-result-hero.urgent .rsa-result-title { color:#921b25; font-size:1.72rem; }
        .rsa-result-copy { margin:0; color:#43586a; font-size:.96rem; }
        .rsa-result-tags { display:flex; flex-wrap:wrap; align-items:center; justify-content:flex-end; gap:.5rem; min-width:200px; }
        .rsa-result-hero .rsa-badge { max-width:100%; white-space:normal; text-align:center; line-height:1.2; }
        .rsa-status-pill { display:inline-flex; padding:.38rem .72rem; border-radius:999px; font-size:.75rem; font-weight:780; letter-spacing:.045em; background:var(--rsa-teal-soft); color:#086c6e; border:1px solid #b8dedb; }
        .rsa-status-pill.urgent { background:#a8202b; border-color:#a8202b; color:white; }
        .rsa-status-pill.insufficient { background:var(--rsa-gray-soft); border-color:#cbd2d9; color:#465362; }
        .rsa-status-pill.partial { background:var(--rsa-amber-soft); border-color:#edce9c; color:#8b4909; }
        .rsa-status-pill.complete { background:var(--rsa-teal-soft); border-color:#b8dedb; color:#086c6e; }
        .rsa-evidence-card { min-height:9.7rem; height:100%; border:1px solid var(--rsa-line); border-radius:.85rem; background:#fff; padding:.85rem .95rem; box-shadow:0 2px 8px rgba(24,49,68,.035); }
        .rsa-evidence-top { display:flex; align-items:flex-start; justify-content:space-between; gap:.45rem; }
        .rsa-evidence-name { font-weight:740; font-size:1rem; color:var(--rsa-navy); }
        .rsa-score { text-align:right; color:#273e53; font-size:.82rem; font-weight:720; white-space:nowrap; }
        .rsa-score small { display:block; color:#697887; font-size:.66rem; font-weight:600; }
        .rsa-state { display:inline-flex; margin:.5rem 0 .38rem; padding:.22rem .48rem; border-radius:999px; font-size:.68rem; font-weight:750; background:var(--rsa-teal-soft); color:#096d6f; }
        .rsa-state.warn { color:#874609; background:var(--rsa-amber-soft); }
        .rsa-state.unavailable { color:#4f5b67; background:var(--rsa-gray-soft); }
        .rsa-evidence-meaning { min-height:2.15rem; margin:.12rem 0 .38rem; color:#4a5e6f; font-size:.83rem; line-height:1.34; }
        .rsa-evidence-model { color:#647485; font-size:.75rem; font-weight:620; }
        .rsa-fusion { border:1px solid var(--rsa-line); border-radius:.9rem; background:#fff; padding:.9rem 1rem; box-shadow:0 2px 8px rgba(24,49,68,.03); }
        .rsa-fusion-row { display:grid; grid-template-columns:minmax(130px,1fr) minmax(140px,2.5fr) minmax(185px,1.4fr); align-items:center; gap:.7rem; margin:.44rem 0; }
        .rsa-fusion-label { color:#334b60; font-weight:650; font-size:.81rem; }
        .rsa-track { height:.58rem; background:#e8edf0; border-radius:999px; overflow:hidden; }
        .rsa-fill { height:100%; background:#167f83; border-radius:999px; }
        .rsa-fusion-number { color:#56697a; text-align:right; font-size:.74rem; white-space:nowrap; }
        .rsa-fusion-summary { border-left:1px solid #e6ebef; padding-left:1rem; color:#42576a; font-size:.88rem; }
        .rsa-fusion-score { font-size:1.62rem; font-weight:780; color:var(--rsa-navy); margin:.1rem 0; }
        .rsa-why { border:1px solid #dbe6e9; border-radius:.9rem; background:#f9fcfc; padding:.8rem 1rem; }
        .rsa-why-title { margin:0 0 .35rem; font-size:.95rem; font-weight:730; }
        .rsa-why-line { margin:.22rem 0; color:#465a6c; font-size:.82rem; line-height:1.35; }
        .rsa-profile { border:1px solid var(--rsa-line); border-radius:.9rem; background:#fff; padding:.85rem 1rem; }
        .rsa-profile-label { color:#536879; font-size:.73rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
        .rsa-profile-title { margin:.18rem 0 .45rem; color:var(--rsa-navy); font-weight:750; font-size:1rem; }
        .rsa-profile-model { margin:.22rem 0; font-size:.82rem; color:#42586b; }
        .rsa-profile-note { margin:.4rem 0 0; color:#627282; font-size:.76rem; line-height:1.35; }
        .rsa-urgency-note { border:1px solid #ecc6c5; border-radius:.7rem; padding:.48rem .62rem; color:#8f2028; background:#fff6f5; font-size:.78rem; font-weight:700; }
        [data-testid="stExpander"] { border-color:#dbe3e9; border-radius:.7rem; background:#fff; }
        @media (max-width:900px) {
          .block-container { padding:.8rem 1rem 2rem; }
          .rsa-head-row { flex-direction:column; }
          .rsa-badges { justify-content:flex-start; }
          .rsa-fusion-row { grid-template-columns:1fr; gap:.25rem; }
          .rsa-fusion-number { text-align:left; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_product_header(
    st: Any,
    *,
    role: str,
    active_step: int,
    connectivity: tuple[str, str] | None = None,
) -> None:
    """Render the recognizable product identity and three-stage workflow."""
    roles = {
        "collector": "Multimodal prehospital stroke-triage decision support",
        "clinician": "Clinician review of submitted research evidence",
        "edge": "Local-first collection and assessment",
    }
    status_badge = ""
    if connectivity:
        label, state = connectivity
        status_badge = f"<span class='rsa-badge {escape(state)}'>{escape(label)}</span>"
    steps = ("Capture evidence", "Assess", "Clinician review")
    step_html = []
    for index, label in enumerate(steps, start=1):
        state = "done" if index < active_step else "active" if index == active_step else ""
        step_html.append(
            f"<span class='rsa-step {state}'><span class='rsa-step-dot'>{'✓' if index < active_step else index}</span>{label}</span>"
        )
    workflow = "<span class='rsa-step-arrow'>›</span>".join(step_html)
    title = roles.get(role, roles["collector"])
    st.markdown(
        "<div class='rsa-header'><div class='rsa-head-row'>"
        f"<div><div class='rsa-brand'>RuralStroke-Assist</div><p class='rsa-subtitle'>{escape(title)}</p></div>"
        "<div class='rsa-badges'><span class='rsa-badge research'>Research prototype</span>"
        "<span class='rsa-badge safety'>Decision support — not diagnosis</span>"
        f"{status_badge}</div></div><div class='rsa-workflow'>{workflow}</div></div>",
        unsafe_allow_html=True,
    )


def _status_label(status: object) -> tuple[str, str]:
    normalized = str(status or "").upper()
    if normalized == "INSUFFICIENT_EVIDENCE":
        return "INSUFFICIENT EVIDENCE", "insufficient"
    if normalized == "PARTIAL":
        return "PARTIAL", "partial"
    if normalized in {"COMPLETE", "ASSESSED"}:
        return "COMPLETE", "complete"
    return normalized.replace("_", " ") or "ASSESSMENT", "complete"


def _safe_text(value: object) -> str:
    return escape(str(value))


def _quality_display(value: object, available: bool) -> tuple[str, str]:
    if not available or str(value or "").upper() == "UNAVAILABLE":
        return "Unavailable", "unavailable"
    normalized = str(value).upper()
    if normalized in {"WARN", "REJECT"}:
        return ("Quality needs review" if normalized == "WARN" else "Quality rejected"), "warn"
    if normalized == "NOT_ASSESSED":
        return "Quality not assessed", "warn"
    return "Quality passed", "available"


def _unavailable_copy(name: str, *, failed: bool, renormalized: bool) -> str:
    if name == "speech":
        message = "Speech file could not be analyzed. Upload or record another sample." if failed else "No usable speech recording was provided."
    elif name == "face":
        message = "Visual evidence could not be analyzed. Upload or capture another image." if failed else "No usable face image was provided."
    elif name == "metadata_context":
        message = "Contextual risk evidence was unavailable; check the recorded background fields."
    else:
        message = "Acute symptom evidence was unavailable; review the symptom inputs."
    if renormalized:
        message += " Remaining fusion weights were renormalized."
    return message


def _render_evidence_card(
    st: Any,
    name: str,
    title: str,
    execution: Mapping[str, Any],
    *,
    profile: str,
    renormalized: bool,
) -> None:
    available = bool(execution.get("available")) and execution.get("score") is not None
    quality, quality_class = _quality_display(execution.get("quality_status"), available)
    score = format_evidence_score(execution.get("score")) if available else None
    score_html = (
        f"<div class='rsa-score'>{score}<small>Evidence score</small></div>"
        if score is not None
        else ""
    )
    semantic = {
        "face": "Visual proxy evidence; not a clinical probability.",
        "speech": "Dysarthria-related speech proxy evidence; not stroke-specific evidence.",
        "acute_symptoms": "Deterministic acute symptom evidence and safety safeguards.",
        "metadata_context": "Contextual/background risk evidence, not acute stroke evidence.",
    }.get(name, "Research branch evidence; not a clinical probability.")
    if not available:
        semantic = _unavailable_copy(
            name,
            failed=execution.get("failure") is not None,
            renormalized=renormalized,
        )
    badge_class = "" if available and quality_class == "available" else quality_class
    model = model_label(name, execution, profile)
    model_prefix = "Model" if available else "Configured branch"
    card = (
        "<div class='rsa-evidence-card'><div class='rsa-evidence-top'>"
        f"<div class='rsa-evidence-name'>{escape(title)}</div>{score_html}</div>"
        f"<span class='rsa-state {badge_class}'>{escape(quality)}</span>"
        f"<p class='rsa-evidence-meaning'>{escape(semantic)}</p>"
        f"<div class='rsa-evidence-model'>{model_prefix}: {escape(model)}</div></div>"
    )
    st.markdown(card, unsafe_allow_html=True)


def _render_late_fusion(st: Any, result: Mapping[str, Any], band: str) -> None:
    fusion = result.get("fusion")
    st.markdown("### Late-fusion summary")
    if not fusion:
        st.markdown(
            "<div class='rsa-fusion'><strong>INSUFFICIENT EVIDENCE</strong>"
            "<p class='rsa-result-copy'>No fused risk band was produced. No fused score is shown.</p></div>",
            unsafe_allow_html=True,
        )
        return
    contributions = fusion.get("modality_contributions") or {}
    weights = fusion.get("normalized_weights_used") or {}
    total = sum(max(0.0, float(value)) for value in contributions.values())
    rows = []
    display_names = {
        "face": "Visual evidence",
        "speech": "Speech evidence",
        "acute_symptoms": "Acute symptoms",
        "metadata_context": "Contextual risk",
    }
    for name in ("face", "speech", "acute_symptoms", "metadata_context"):
        if name not in weights:
            continue
        contribution = max(0.0, float(contributions.get(name, 0.0)))
        share = min(100.0, max(0.0, contribution / total * 100.0)) if total > 0 else 0.0
        rows.append(
            "<div class='rsa-fusion-row'>"
            f"<div class='rsa-fusion-label'>{display_names[name]}</div>"
            f"<div class='rsa-track'><div class='rsa-fill' style='width:{share:.1f}%'></div></div>"
            f"<div class='rsa-fusion-number'>Weighted contribution {contribution:.3f} · {share:.0f}% of total</div>"
            "</div>"
        )
    score = format_evidence_score(fusion.get("evidence_score"))
    score_copy = score if score is not None else "Not produced"
    escalation = acute_escalation_triggered(result)
    if escalation is True:
        escalation_copy = "Acute symptom safeguard triggered the URGENT override."
        escalation_class = "rsa-urgency-note"
    elif escalation is False:
        escalation_copy = "No deterministic urgent escalation was triggered by the recorded symptoms."
        escalation_class = "rsa-profile-note"
    else:
        escalation_copy = "Acute symptom status is unavailable."
        escalation_class = "rsa-profile-note"
    st.markdown(
        "<div class='rsa-fusion'><div class='rsa-fusion-row' style='grid-template-columns:1fr 1fr 1fr'>"
        "<div class='rsa-fusion-label'>Weighted contribution share</div>"
        "<div class='rsa-fusion-label' style='grid-column:span 2;color:#647485;font-weight:500'>Bars show each branch’s share of summed weighted contributions.</div></div>"
        + "".join(rows)
        + "<div class='rsa-fusion-summary' style='margin-top:.65rem;border-left:0;padding-left:0'>"
        f"<div>Fused evidence score</div><div class='rsa-fusion-score'>{score_copy}</div>"
        f"<div>Final evidence band: <strong>{escape(band.replace('_', ' '))}</strong></div></div>"
        f"<div class='{escalation_class}' style='margin-top:.55rem'>{escape(escalation_copy)}</div></div>",
        unsafe_allow_html=True,
    )


def _provenance_detail(
    execution: Mapping[str, Any],
    profile: str,
    modality: str,
    runtime_provenance: Mapping[str, Any] | None,
) -> str:
    details = execution.get("details") or {}
    provenance = str(execution.get("provenance", ""))
    if profile == "pretrained_reference":
        upstream = details.get("upstream_model_id")
        revision = details.get("upstream_revision")
        if upstream:
            return f"{upstream} · revision {str(revision or 'recorded')[:8]}"
        if details.get("package_version"):
            return f"tabpfn=={details['package_version']} · V2 checkpoint/context recorded"
        if "artifact_sha256=" in provenance:
            artifact_hash = provenance.partition("artifact_sha256=")[2].split(";", 1)[0]
            return f"Artifact SHA-256 {artifact_hash[:10]}…"
    runtime_artifacts = (runtime_provenance or {}).get("runtime_artifacts") or {}
    artifact = runtime_artifacts.get(modality) or {}
    if artifact:
        name = Path(str(artifact.get("artifact", ""))).name
        checksum = str(artifact.get("artifact_sha256", ""))
        backend = artifact.get("backend") or artifact.get("runtime")
        suffix = f" · {backend}" if backend else ""
        if checksum:
            suffix += f" · SHA-256 {checksum[:10]}…"
        return f"{name or 'Runtime artifact'}{suffix}"
    basename = Path(provenance.split(";", 1)[0]).name
    return f"Recorded provenance · {basename}" if basename else "Per-assessment provenance recorded"


def _render_model_provenance(
    st: Any,
    result: Mapping[str, Any],
    *,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> None:
    profile = runtime_profile_for_result(result, runtime_provenance)
    executions = result.get("modality_executions") or {}
    order = ("face", "speech", "metadata_context", "acute_symptoms")
    titles = {"face": "Visual", "speech": "Speech", "metadata_context": "Contextual risk", "acute_symptoms": "Acute symptoms"}
    current_models = []
    for name in order:
        execution = executions.get(name) or {}
        if execution.get("available"):
            current_models.append(
                f"<p class='rsa-profile-model'><strong>{titles[name]}:</strong> "
                f"{escape(model_label(name, execution, profile))} ·<br>"
                f"<span class='rsa-profile-note'>{escape(_provenance_detail(execution, profile, name, runtime_provenance))}</span></p>"
            )
        else:
            current_models.append(
                f"<p class='rsa-profile-model'><strong>{titles[name]}:</strong> Unavailable in this assessment</p>"
            )
    st.markdown("### Model & provenance")
    st.markdown(
        "<div class='rsa-profile'><div class='rsa-profile-label'>Current assessment profile</div>"
        f"<div class='rsa-profile-title'>{escape(profile_label(profile))}</div>"
        + "".join(current_models)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Project profile composition")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "<div class='rsa-profile'><div class='rsa-profile-label'>Research reference profile</div>"
            "<div class='rsa-profile-title'>Three selected pretrained models</div>"
            "<p class='rsa-profile-model'>ImageNet-pretrained MobileNetV2</p>"
            "<p class='rsa-profile-model'>DistilHuBERT speech representation</p>"
            "<p class='rsa-profile-model'>TabPFN v2 structured-data model</p>"
            "<p class='rsa-profile-model'>Deterministic acute symptom safeguards</p>"
            "<p class='rsa-profile-note'>The complete reference AssessmentService path has been executed with per-branch provenance.</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            "<div class='rsa-profile'><div class='rsa-profile-label'>Deployment-oriented profile</div>"
            "<div class='rsa-profile-title'>Lighter alternatives for constrained local execution</div>"
            "<p class='rsa-profile-model'>Visual: MobileNetV2 / edge artifact, by runtime</p>"
            "<p class='rsa-profile-model'>Speech: MFCC + Random Forest</p>"
            "<p class='rsa-profile-model'>Contextual risk: Logistic Regression</p>"
            "<p class='rsa-profile-model'>Acute symptoms: deterministic safeguards</p></div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "The reference profile demonstrates three selected pretrained models; the deployment-oriented profile retains lighter alternatives for constrained local execution."
    )


def render_assessment_result(
    st: Any,
    result: dict[str, Any],
    *,
    include_limitations: bool = True,
    runtime_provenance: Mapping[str, Any] | None = None,
    compact: bool = False,
) -> None:
    """Render the stored result as an evidence-led, non-diagnostic summary."""
    fusion = result.get("fusion") or {}
    band = str(fusion.get("risk_band") or "INSUFFICIENT_EVIDENCE").upper()
    status_text, status_class = _status_label(result.get("status"))
    assessment_status_text, assessment_status_class = status_text, status_class
    if band == "URGENT":
        title = "URGENT — acute symptom evidence indicates immediate escalation"
        hero_class = "urgent"
        status_text, status_class = "URGENT", "urgent"
    elif band == "INSUFFICIENT_EVIDENCE":
        title = "INSUFFICIENT EVIDENCE — no fused risk band was produced"
        hero_class = "insufficient"
        status_text, status_class = "INSUFFICIENT EVIDENCE", "insufficient"
    else:
        title = f"Multimodal evidence level: {band}"
        hero_class = band.lower()
    guidance = band_guidance(band)
    profile = runtime_profile_for_result(result, runtime_provenance)
    status_pills = f"<span class='rsa-status-pill {status_class}'>{escape(status_text)}</span>"
    if band == "URGENT":
        status_pills += (
            f"<span class='rsa-status-pill {assessment_status_class}'>"
            f"{escape(assessment_status_text)}</span>"
        )
    st.markdown(
        f"<div class='rsa-result-hero {hero_class}'>"
        f"<div><div class='rsa-result-title'>{escape(title)}</div>"
        f"<p class='rsa-result-copy'>{escape(guidance)}</p>"
        "<p class='rsa-result-copy' style='font-weight:700;margin-top:.34rem'>Decision support — not diagnosis.</p></div>"
        f"<div class='rsa-result-tags'>{status_pills}"
        f"<span class='rsa-badge research'>{escape(profile_label(profile))}</span></div></div>",
        unsafe_allow_html=True,
    )

    executions = result.get("modality_executions") or {}
    weights = fusion.get("normalized_weights_used") or {}
    cards = (
        ("face", "Visual evidence"),
        ("speech", "Speech evidence"),
        ("acute_symptoms", "Acute symptoms"),
        ("metadata_context", "Contextual risk"),
    )
    st.markdown("<div class='rsa-section-kicker' style='margin-top:.8rem'>Evidence breakdown · branch scores are evidence values, not clinical probabilities</div>", unsafe_allow_html=True)
    columns = st.columns(2 if compact else 4, gap="small")
    for index, (name, title) in enumerate(cards):
        execution = executions.get(name) or {"available": False, "score": None, "quality_status": "UNAVAILABLE"}
        renormalized = bool(fusion) and execution.get("score") is None and len(weights) < 4
        with columns[index % len(columns)]:
            _render_evidence_card(st, name, title, execution, profile=profile, renormalized=renormalized)

    _render_late_fusion(st, result, band)
    explanation = result_explanation_lines(result)
    if explanation:
        st.markdown(
            "<div class='rsa-why'><p class='rsa-why-title'>Why this result?</p>"
            + "".join(f"<p class='rsa-why-line'>• {escape(item)}</p>" for item in explanation[:4])
            + "</div>",
            unsafe_allow_html=True,
        )
    _render_model_provenance(st, result, runtime_provenance=runtime_provenance)
    if include_limitations:
        render_limitations(st, result)


def render_limitations(st: Any, result: dict[str, Any]) -> None:
    fusion = result.get("fusion") or {}
    groups = group_warnings(result)
    with st.expander("Quality, limitations & technical details", expanded=False):
        st.markdown("**Assessment limitations**")
        for value in groups.assessment_limitations:
            st.write(value)
        if groups.modality_limitations:
            st.markdown("**Branch limitations**")
            for value in groups.modality_limitations:
                st.write(value)
        if groups.input_quality_findings:
            st.markdown("**Input-quality findings**")
            for value in groups.input_quality_findings:
                st.warning(value)
        else:
            st.caption("No additional input-quality findings were recorded.")
    with st.expander("Advanced provenance and numeric details", expanded=False):
        if fusion.get("evidence_score") is not None:
            st.write("Fused evidence score", fusion["evidence_score"])
        st.write("Weighted contributions", fusion.get("modality_contributions", {}))
        st.write("Normalized weights", fusion.get("normalized_weights_used", {}))
        st.write("Per-branch provenance", result.get("provenance", []))
        st.write("Warnings", groups.technical_warnings)


def render_case_header(st: Any, case: Any) -> None:
    st.write(f"{short_case_reference(case.case_id)} · Status: `{humanize_value(case.status.value)}`")
    with st.expander("Technical case details", expanded=False):
        st.write("Full case ID", case.case_id)
        st.write("Created", case.created_at)


def render_case_context(st: Any, case: Any) -> None:
    context_rows, symptom_rows = build_context_rows(case.assessment_input)
    st.subheader("Collection context")
    for label, value in (("Collector", case.collector_identity), ("Facility", case.facility), ("Pseudonymous patient code", case.patient_code)):
        st.write(f"**{label}:** {humanize_value(value)}")
    st.subheader("Background metadata")
    if context_rows:
        for row in context_rows:
            st.write(f"**{row.label}:** {row.value}")
    else:
        st.caption("No background metadata recorded.")
    st.subheader("Acute symptoms")
    if symptom_rows:
        for row in symptom_rows:
            st.write(f"**{row.label}:** {row.value}")
    else:
        st.caption("No acute symptoms recorded.")
    with st.expander("Technical case data", expanded=False):
        st.json(case.assessment_input)


def render_submitted_media(st: Any, workflow: Any, case: Any) -> None:
    st.subheader("Submitted image and audio")
    found = {attachment.kind.value: attachment for attachment in case.attachments}
    face = found.get("face")
    audio = found.get("audio")
    if face:
        try:
            st.markdown("**Facial image evidence**")
            st.image(str(workflow.attachment_store.resolve(face)), caption="Submitted facial image")
        except Exception:
            st.warning("The submitted facial image is unavailable. Reopen or replace it if needed.")
    else:
        st.caption("No facial image was submitted.")
    if audio:
        try:
            st.markdown("**Speech recording evidence**")
            st.audio(str(workflow.attachment_store.resolve(audio)), format=audio.media_type)
            st.caption("Submitted speech recording")
        except Exception:
            st.warning("The submitted speech recording is unavailable. Upload or record it again if needed.")
    else:
        st.caption("No speech recording was submitted.")


def render_preassessment_quality(st: Any, face_bytes: bytes | None, audio_bytes: bytes | None) -> None:
    """Run lightweight quality previews and show concise status plus recovery."""
    if face_bytes:
        try:
            image = Image.open(BytesIO(face_bytes)).convert("RGB")
            result = OpenCVFaceQualityAssessor()(image)
            if result.status.value == "PASS":
                st.success("Visual input · Quality passed")
            else:
                st.warning(f"Visual input · {result.status.value.title()}")
            for finding in result.findings:
                if finding.status.value in {"WARN", "REJECT"}:
                    st.warning(f"Visual input: {finding.message}")
                elif finding.status.value == "NOT_ASSESSED":
                    st.caption(f"Visual input: {finding.message}")
        except Exception:
            st.error("Visual image could not be decoded. Upload or capture another image.")
    if audio_bytes:
        try:
            import soundfile as sf
            signal, sample_rate = sf.read(BytesIO(audio_bytes), always_2d=False)
            signal = np.asarray(signal, dtype=np.float32)
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            result = DefaultAudioQualityAssessor()(signal, sample_rate)
            if result.status.value == "PASS":
                st.success("Speech input · Quality passed")
            else:
                st.warning(f"Speech input · {result.status.value.title()}")
            for finding in result.findings:
                if finding.status.value in {"WARN", "REJECT"}:
                    st.warning(f"Speech input: {finding.message} Upload or record another sample if needed.")
        except Exception:
            st.error("Speech file could not be analyzed. Upload or record another sample.")
