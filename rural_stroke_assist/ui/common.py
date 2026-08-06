"""Shared calm clinical rendering helpers."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from rural_stroke_assist.quality.audio_quality import DefaultAudioQualityAssessor
from rural_stroke_assist.quality.face_quality import OpenCVFaceQualityAssessor
from rural_stroke_assist.ui.presentation import (
    band_guidance,
    build_context_rows,
    build_modality_summaries,
    group_warnings,
    humanize_explanation,
    humanize_value,
    short_case_reference,
)


def apply_theme(st: Any) -> None:
    st.markdown("""
    <style>
    :root { --navy:#12304a; --teal:#167c80; --surface:#f5f8fa; }
    .block-container { max-width: 1100px; padding-top: 2rem; }
    .stButton > button { border-radius: .45rem; border: 1px solid #167c80; }
    .clinical-card { background: var(--surface); border-left: 5px solid var(--teal); padding: 1rem; border-radius: .4rem; }
    .safety-note { color: #40515d; font-size: .92rem; }
    </style>
    """, unsafe_allow_html=True)


def render_assessment_result(st: Any, result: dict[str, Any], *, include_limitations: bool = True) -> None:
    fusion = result.get("fusion") or {}
    band = fusion.get("risk_band", "INSUFFICIENT_EVIDENCE")
    st.markdown(f"<div class='clinical-card'><h3>Current multimodal evidence level: {band}</h3><p>Review symptoms, individual modality quality, and specialist guidance together.</p></div>", unsafe_allow_html=True)
    st.info(band_guidance(band))
    st.subheader("Modality summaries")
    for summary in build_modality_summaries(result):
        st.markdown(f"**{summary.name}** · {summary.availability} · Quality: {summary.quality_state}")
        st.write(summary.interpretation)
        if summary.failure_reason:
            st.caption(f"Unavailable reason: {summary.failure_reason}")
    st.subheader("Explanations")
    for item in result.get("explanations", []):
        if item.get("code") != "non_diagnostic":
            st.write(f"• {humanize_explanation(item, result)}")
    if include_limitations:
        render_limitations(st, result)


def render_limitations(st: Any, result: dict[str, Any]) -> None:
    fusion = result.get("fusion") or {}
    groups = group_warnings(result)
    st.subheader("Limitations and quality")
    for title, values in (("Assessment limitations", groups.assessment_limitations), ("Modality limitations", groups.modality_limitations), ("Input-quality findings", groups.input_quality_findings)):
        st.markdown(f"**{title}**")
        if values:
            for value in values:
                st.write(f"• {value}")
        else:
            st.caption("None recorded.")
    with st.expander("Technical details", expanded=False):
        if fusion.get("evidence_score") is not None:
            st.write("Evidence score", fusion["evidence_score"])
        st.write("Contributions", fusion.get("modality_contributions", {}))
        st.write("Provenance", result.get("provenance", []))
    with st.expander("Technical warnings", expanded=False):
        for warning in groups.technical_warnings:
            st.write(f"• {warning}")


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
    with st.expander("Developer data", expanded=False):
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
            st.error("The submitted facial image is unavailable or could not be loaded.")
    else:
        st.info("No facial image was submitted.")
    if audio:
        try:
            st.markdown("**Speech recording evidence**")
            st.audio(str(workflow.attachment_store.resolve(audio)), format=audio.media_type)
            st.caption("Submitted speech recording")
        except Exception:
            st.error("The submitted speech recording is unavailable or could not be loaded.")
    else:
        st.info("No speech recording was submitted.")


def render_preassessment_quality(st: Any, face_bytes: bytes | None, audio_bytes: bytes | None) -> None:
    """Run lightweight existing quality checks before the learned assessment."""
    if face_bytes:
        try:
            image = Image.open(BytesIO(face_bytes)).convert("RGB")
            result = OpenCVFaceQualityAssessor()(image)
            st.caption(f"Face quality preview: {result.status.value}")
            for finding in result.findings:
                (st.warning if finding.status.value in {"WARN", "REJECT"} else st.info)(finding.message)
        except Exception:
            st.warning("Face quality preview could not decode this input; retake or replace it.")
    if audio_bytes:
        try:
            import soundfile as sf
            signal, sample_rate = sf.read(BytesIO(audio_bytes), always_2d=False)
            signal = np.asarray(signal, dtype=np.float32)
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            result = DefaultAudioQualityAssessor()(signal, sample_rate)
            st.caption(f"Audio quality preview: {result.status.value}")
            for finding in result.findings:
                (st.warning if finding.status.value in {"WARN", "REJECT"} else st.info)(finding.message)
        except Exception:
            st.warning("Audio quality preview could not decode this input; retake or replace it.")
