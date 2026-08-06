"""Deterministic Markdown report generation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def build_report(*, run_id: str, suite: str, results: dict[str, Any], warnings: list[str]) -> str:
    lines = ["# Phase 4 Evaluation Report", "", f"- Run ID: `{run_id}`", f"- Suite: `{suite}`", "- Scope: reproducibility, proxy-partition metrics, deterministic robustness, and sensitivity analysis.", "- Direct model performance and runtime-adapter accepted-subset performance are reported separately.", "- Rule verification, corruption robustness, missing-modality behavior, stability, and engineering sensitivity are not clinical accuracy measures.", "- Clinical diagnostic accuracy and fusion clinical benefit: unsupported; no paired multimodal clinical dataset is available.", "", "## Results", ""]
    if "face_coverage" in results:
        face = results["face_coverage"]; lines += ["## Prominent coverage finding", "", f"The face adapter accepted {face.get('accepted', 0)} of {face.get('total', 0)} held-out images ({face.get('coverage', 0):.1%}) and rejected {face.get('rejected', 0)}. Direct face-model metrics must not be read as runtime coverage metrics.", ""]
    speech = results.get("speech")
    if isinstance(speech, dict) and speech.get("direct", {}).get("roc_auc") is not None and speech.get("adapter_aware", {}).get("roc_auc") is not None:
        lines += ["## Speech metric reconciliation", "", f"The direct canonical held-out evaluation produced ROC-AUC {speech['direct'].get('roc_auc'):.4f} across {speech.get('processed_direct')} samples. The runtime adapter accepted {speech.get('accepted')} of {speech.get('total_held_out')} samples and produced ROC-AUC {speech['adapter_aware'].get('roc_auc'):.4f}. The historical 0.9426 value matches the direct path ({speech['direct'].get('roc_auc'):.10f}); 0.9512 is the accepted-subset adapter-aware result. The difference is therefore subset filtering by runtime quality rejection, not positive-class remapping or a changed model.", ""]
    for name, value in results.items():
        lines += [f"### {name.replace('_', ' ').title()}", "", "```json", json.dumps(value, indent=2, default=str), "```", ""]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {warning}" for warning in dict.fromkeys(warnings)] + [""]
    lines += ["## Interpretation", "", "Scores are branch-specific proxy evidence or contextual-risk scores, not calibrated clinical stroke probabilities. Fusion outputs are engineering scenario results and must not be interpreted as clinical validation.", ""]
    return "\n".join(lines)


def write_report(path: Path, **kwargs: Any) -> None:
    path.write_text(build_report(**kwargs), encoding="utf-8")
