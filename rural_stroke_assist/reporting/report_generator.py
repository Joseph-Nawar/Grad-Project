from rural_stroke_assist.fusion.fusion_engine import FusionResult


def generate_text_report(fusion_result: FusionResult) -> str:
    """Generate a human-readable triage report."""
    evidence_text = "\n".join(f"- {item}" for item in fusion_result.evidence) or "- No specific evidence available."
    warnings_text = "\n".join(f"- {item}" for item in fusion_result.warnings) or "- No warnings."

    return f"""
RuralStroke-Triage Report

Triage Level:
{fusion_result.triage_level}

Final Risk Score:
{fusion_result.final_risk_score}

System Confidence:
{fusion_result.confidence}

Evidence:
{evidence_text}

Warnings:
{warnings_text}

Important Safety Note:
This system does not diagnose stroke. It provides decision-support guidance for early triage and escalation.
""".strip()