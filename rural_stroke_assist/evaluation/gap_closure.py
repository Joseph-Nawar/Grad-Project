"""Execution-backed audits added to close the Phase 4 evaluation gaps."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from rural_stroke_assist.assessment.factory import create_default_assessment_service
from rural_stroke_assist.assessment.service import AssessmentService, ORDER
from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
from rural_stroke_assist.capture.schemas import AssessmentInput
from rural_stroke_assist.inference.contracts import ModalityEvidence, QualityStatus
from rural_stroke_assist.inference.face_adapter import FaceAdapter
from rural_stroke_assist.inference.metadata_adapter import MetadataInput
from rural_stroke_assist.inference.registry import BaselineRegistry
from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
from rural_stroke_assist.quality.face_quality import OpenCVFaceQualityAssessor
from rural_stroke_assist.evaluation.artifact_writer import write_csv, write_json
from rural_stroke_assist.evaluation.metrics import classification_metrics, metric_dict


def _clean(value: Any) -> Any:
    return None if pd.isna(value) else value


def _metadata_from_row(row: Any) -> MetadataInput:
    return MetadataInput.model_validate({
        "age": _clean(row.age), "hypertension": _clean(row.hypertension),
        "heart_disease": _clean(row.heart_disease), "avg_glucose_level": _clean(row.avg_glucose_level),
        "bmi": _clean(row.bmi), "gender": _clean(row.gender), "ever_married": _clean(row.ever_married),
        "work_type": _clean(row.work_type), "residence_type": _clean(row.Residence_type),
        "smoking_status": _clean(row.smoking_status),
    })


def _write_contact_sheet(records: list[dict[str, Any]], output: Path, title: str, columns: int = 4) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    thumbs: list[Image.Image] = []
    for record in records[:16]:
        try:
            with Image.open(record["path"]) as image:
                thumb = image.convert("RGB"); thumb.thumbnail((180, 140)); canvas = Image.new("RGB", (200, 180), "white"); canvas.paste(thumb, ((200 - thumb.width) // 2, 5)); ImageDraw.Draw(canvas).text((5, 150), f"{record['label']} {record['quality_state']}", fill="black"); thumbs.append(canvas)
        except (OSError, ValueError):
            continue
    if not thumbs:
        Image.new("RGB", (200, 40), "white").save(output); return
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 200, rows * 180), "white")
    for index, thumb in enumerate(thumbs): sheet.paste(thumb, ((index % columns) * 200, (index // columns) * 180))
    sheet.save(output)


def face_coverage_audit(manifest: str | Path, registry: BaselineRegistry, output_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(manifest).query("split == 'test'")
    adapter, assessor = FaceAdapter(registry=registry), OpenCVFaceQualityAssessor()
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        path = Path(row.path); quality_state, findings, score = "REJECT", [], None; face_count = None; face_size = None
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB"); assessment = assessor(rgb); findings = [item.code for item in assessment.findings]; quality_state = assessment.status.value
                gray = np.asarray(rgb.convert("L")); import cv2
                faces = assessor._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5); face_count = len(faces); face_size = max((min(int(item[2]), int(item[3])) for item in faces), default=None)
            evidence = adapter.infer(path); accepted = bool(evidence.available and evidence.score is not None); score = evidence.score
        except Exception as exc:
            accepted = False; findings.append(type(exc).__name__)
        records.append({"path": str(path), "label": str(row.class_label), "true_positive": int(str(row.class_label).lower() == "stroke"), "accepted": accepted, "quality_state": quality_state, "quality_findings": ";".join(dict.fromkeys(findings)), "width": int(row.width), "height": int(row.height), "face_count": face_count, "face_size_px": face_size, "score": score})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "face_adapter_coverage.json", {"total": len(records), "processed": len(records), "accepted": sum(item["accepted"] for item in records), "rejected": sum(not item["accepted"] for item in records), "coverage": sum(item["accepted"] for item in records) / len(records), "positive_class": "Stroke", "class_conditional_rejection_rate": {label: sum(not item["accepted"] for item in records if item["label"] == label) / max(1, sum(item["label"] == label for item in records)) for label in sorted({item["label"] for item in records})}})
    write_csv(output_dir / "face_rejections.csv", (item for item in records if not item["accepted"]))
    for filename, key in (("face_rejection_by_reason.csv", "quality_findings"), ("face_rejection_by_label.csv", "label"), ("face_acceptance_by_quality.csv", "quality_state")):
        counts = Counter(item[key] for item in records if (filename != "face_acceptance_by_quality.csv" and not item["accepted"]) or filename == "face_acceptance_by_quality.csv" and item["accepted"])
        write_csv(output_dir / filename, ({"value": value, "count": count} for value, count in sorted(counts.items())), ["value", "count"])
    groups = {"accepted_pass": lambda x: x["accepted"] and x["quality_state"] == "PASS", "accepted_warn": lambda x: x["accepted"] and x["quality_state"] == "WARN", "no_face": lambda x: "no_face" in x["quality_findings"], "face_too_small": lambda x: "face_too_small" in x["quality_findings"], "dimension_failure": lambda x: "dimensions" in x["quality_findings"], "multiple_face": lambda x: "multiple_faces" in x["quality_findings"], "false_positive": lambda x: x["accepted"] and x["true_positive"] == 0 and float(x["score"] or 0) >= .5, "false_negative": lambda x: x["accepted"] and x["true_positive"] == 1 and float(x["score"] or 0) < .5}
    for name, predicate in groups.items(): _write_contact_sheet([item for item in records if predicate(item)], output_dir / "contact_sheets" / f"{name}.png", name)
    return {"total": len(records), "processed": len(records), "accepted": sum(item["accepted"] for item in records), "rejected": sum(not item["accepted"] for item in records), "coverage": sum(item["accepted"] for item in records) / len(records), "positive_class": "Stroke"}


def speech_speaker_audit(manifest: str | Path, registry: BaselineRegistry, output_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(manifest).query("split == 'test'"); adapter = SpeechAdapter(registry=registry); rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        target = int(str(row.label).lower() == "dysarthric")
        try:
            evidence = adapter.infer(row.path); accepted = bool(evidence.available and evidence.score is not None); score = evidence.score; quality = evidence.quality_status.value; failure = "" if accepted else ";".join(item.code for item in evidence.quality_findings) or "quality_rejected"
        except Exception as exc:
            accepted, score, quality, failure = False, None, "REJECT", type(exc).__name__
        rows.append({"speaker_id": str(row.speaker_id), "class": str(row.label), "target": target, "path": str(row.path), "accepted": accepted, "score": score, "predicted": int(score >= .5) if score is not None else None, "quality_state": quality, "failure": failure})
    write_csv(output_dir / "speech_per_speaker.csv", rows)
    summary_rows = []
    for speaker, group in pd.DataFrame(rows).groupby("speaker_id"):
        accepted = group[group.accepted]; scores = accepted.score.dropna().astype(float); correct = int((accepted.predicted == accepted.target).sum())
        summary_rows.append({"speaker_id": speaker, "class": group.iloc[0]["class"], "sample_count": len(group), "accepted_count": len(accepted), "coverage": len(accepted) / len(group), "accuracy": correct / len(accepted) if len(accepted) else None, "correct_count": correct, "mean_score": float(scores.mean()) if len(scores) else None, "median_score": float(scores.median()) if len(scores) else None, "min_score": float(scores.min()) if len(scores) else None, "max_score": float(scores.max()) if len(scores) else None, "quality_failures": int((~group.accepted).sum())})
    summary_rows.sort(key=lambda item: (item["accuracy"] is None, item["accuracy"] if item["accuracy"] is not None else 0))
    write_json(output_dir / "speech_speaker_summary.json", {"speaker_count": len(summary_rows), "speakers": summary_rows, "single_class_per_speaker_roc_auc": "not calculated"})
    write_csv(output_dir / "speech_worst_speakers.csv", summary_rows[:10])
    return {"speaker_count": len(summary_rows), "accepted": sum(item["accepted"] for item in rows)}


def _fixed_input() -> AssessmentInput:
    face = str(pd.read_csv("data/processed/face_split_manifest.csv").query("split == 'test'").iloc[0].path); speech = str(pd.read_csv("data/processed/speech_split_manifest.csv").query("split == 'test'").iloc[0].path); row = pd.read_csv("data/processed/metadata_split_manifest.csv").query("split == 'test'").iloc[0]
    return AssessmentInput(session_id="phase4-fixed-input", face_image_path=face, speech_audio_path=speech, metadata=_metadata_from_row(row), acute_symptoms=AcuteStrokeSymptoms(face_drooping=True, symptom_onset_minutes=20))


def corruption_audit(output_dir: Path) -> dict[str, Any]:
    import soundfile as sf
    with tempfile.TemporaryDirectory(prefix="phase4-corrupt-") as temp:
        root = Path(temp); valid = _fixed_input(); image = root / "image.png"; Image.new("RGB", (160, 160), (30, 30, 30)).save(image); audio = root / "audio.wav"; sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
        small = root / "small.png"; Image.new("RGB", (20, 20), (100, 100, 100)).save(small)
        blurred = root / "blurred.png"
        with Image.open(valid.face_image_path) as original: original.convert("RGB").filter(ImageFilter.GaussianBlur(20)).save(blurred)
        clipped = root / "clipped.wav"; sf.write(clipped, np.ones(80000, dtype=np.float32), 16000)
        short = root / "short.wav"; sf.write(short, np.ones(400, dtype=np.float32) * .01, 16000)
        cases = [("empty_random_image", root / "random.jpg", valid.speech_audio_path, valid.metadata, valid.acute_symptoms), ("no_face_image", image, valid.speech_audio_path, valid.metadata, valid.acute_symptoms), ("very_small_image", small, valid.speech_audio_path, valid.metadata, valid.acute_symptoms), ("blurred_image", blurred, valid.speech_audio_path, valid.metadata, valid.acute_symptoms), ("empty_random_audio", valid.face_image_path, root / "random.wav", valid.metadata, valid.acute_symptoms), ("silent_audio", valid.face_image_path, audio, valid.metadata, valid.acute_symptoms), ("short_audio", valid.face_image_path, short, valid.metadata, valid.acute_symptoms), ("clipped_audio", valid.face_image_path, clipped, valid.metadata, valid.acute_symptoms), ("missing_metadata", valid.face_image_path, valid.speech_audio_path, None, valid.acute_symptoms), ("invalid_symptoms", valid.face_image_path, valid.speech_audio_path, valid.metadata, {"symptom_onset_minutes": -1})]
        (root / "random.jpg").write_bytes(os.urandom(128)); (root / "random.wav").write_bytes(os.urandom(128)); service = create_default_assessment_service(); rows = []
        for name, face, speech, metadata, symptoms in cases:
            try:
                result = service.assess({"session_id": name, "face_image_path": str(face), "speech_audio_path": str(speech), "metadata": metadata, "acute_symptoms": symptoms}); actual = "ok"; failure = ";".join(item.failure.error_type for item in result.modality_executions.values() if item.failure); usable = sum(item.evidence.available for item in result.modality_executions.values()); fusion = result.fusion is not None
            except Exception as exc:
                result, actual, failure, usable, fusion = None, type(exc).__name__, type(exc).__name__, 0, False
            if result is not None:
                quality_failures = [item.evidence for item in result.modality_executions.values() if not item.evidence.available]
                failure = ";".join(finding.code for findings in quality_failures for finding in findings.quality_findings) or failure
            rows.append({"case": name, "expected": "isolated_or_schema_rejection", "actual_status": actual, "usable_modalities": usable, "fusion_available": fusion, "failure_code": failure, "validation_or_isolation": "schema_validation" if actual != "ok" else "adapter_isolation_or_success", "pass": actual in {"ok", "AssessmentInputError", "ValidationError"}})
    write_csv(output_dir / "system_corruption_results.csv", rows); write_json(output_dir / "system_corruption_summary.json", {"count": len(rows), "passed": sum(item["pass"] for item in rows), "results": rows}); return {"count": len(rows), "passed": sum(item["pass"] for item in rows)}


class _FixedAdapter:
    def __init__(self, modality: str, score: float): self.modality, self.score = modality, score
    def infer(self, value: object) -> ModalityEvidence: return ModalityEvidence(modality=self.modality, available=True, score=self.score, score_semantics=f"{self.modality}_evidence", label="test", provenance=f"evaluation:{self.modality}")


def missing_modality_audit(output_dir: Path, registry: BaselineRegistry) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    from itertools import combinations
    from rural_stroke_assist.assessment.fusion_strategy import CanonicalLateFusionStrategy
    rows = []; metadata = MetadataInput(age=60, hypertension=0, heart_disease=0, avg_glucose_level=100, bmi=25, gender="Female", ever_married="Yes", work_type="Private", residence_type="Urban", smoking_status="never smoked"); symptoms = AcuteStrokeSymptoms()
    for size in range(1, 5):
        for combo in combinations(ORDER, size):
            adapters = {name: _FixedAdapter(name, .8) for name in combo}; result = AssessmentService(adapters=adapters, fusion_strategy=CanonicalLateFusionStrategy(registry=registry)).assess({"session_id": "combination", "face_image_path": "x" if "face" in combo else None, "speech_audio_path": "x" if "speech" in combo else None, "metadata": metadata if "metadata_context" in combo else None, "acute_symptoms": symptoms if "acute_symptoms" in combo else None}); rows.append({"combination": "+".join(combo), "status": result.status.value, "fusion": result.fusion is not None, "score": result.fusion.evidence_score if result.fusion else None, "band": result.fusion.risk_band if result.fusion else None, "weight_sum": sum(result.fusion.normalized_weights_used.values()) if result.fusion else 0, "normalized_weights": json.dumps(result.fusion.normalized_weights_used if result.fusion else {})})
    none_result = AssessmentService(adapters={}, fusion_strategy=CanonicalLateFusionStrategy(registry=registry)).assess({"session_id": "none"}); rows.append({"combination": "none", "status": none_result.status.value, "fusion": False, "score": None, "band": None, "weight_sum": 0, "normalized_weights": "{}"})
    service = create_default_assessment_service(); fixed = _fixed_input(); real_rows = []
    for missing in ORDER:
        value = fixed.model_dump(); value["session_id"] = f"missing-{missing}"; value["face_image_path"] = None if missing == "face" else value["face_image_path"]; value["speech_audio_path"] = None if missing == "speech" else value["speech_audio_path"]; value["metadata"] = None if missing == "metadata_context" else value["metadata"]; value["acute_symptoms"] = None if missing == "acute_symptoms" else value["acute_symptoms"]; result = service.assess(value); real_rows.append({"missing_branch": missing, "status": result.status.value, "fusion": result.fusion is not None, "score": result.fusion.evidence_score if result.fusion else None, "band": result.fusion.risk_band if result.fusion else None})
    write_csv(output_dir / "missing_modality_matrix.csv", rows); write_csv(output_dir / "missing_modality_real_artifact_checks.csv", real_rows); return {"combination_count": len(rows), "real_check_count": len(real_rows)}


def stability_audit(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    service = create_default_assessment_service(); value = _fixed_input(); rows = []
    for name in (*ORDER, "assessment_service"):
        scores, labels, qualities, warnings, bands = [], [], [], [], []
        for index in range(20):
            if name == "assessment_service": result = service.assess(value); evidence = result.modality_executions["acute_symptoms"].evidence; score = result.fusion.evidence_score if result.fusion else None; label = result.fusion.risk_band if result.fusion else None; quality = evidence.quality_status.value; warning = ";".join(result.warnings); band = label
            else: evidence = service.adapters[name].infer(value.face_image_path if name == "face" else value.speech_audio_path if name == "speech" else value.metadata if name == "metadata_context" else value.acute_symptoms); score, label, quality, warning, band = evidence.score, evidence.label, evidence.quality_status.value, ";".join(evidence.warnings), None
            scores.append(score); labels.append(label); qualities.append(quality); warnings.append(warning); bands.append(band); rows.append({"component": name, "run": index + 1, "score": score, "label": label, "quality": quality, "warnings": warning, "band": band})
        numeric = [float(x) for x in scores if x is not None]; write_json(output_dir / "stability_results.json", {}) if False else None
    summary = {}
    for name, group in pd.DataFrame(rows).groupby("component"):
        values = group.score.dropna().astype(float); summary[name] = {"runs": len(group), "score_min": float(values.min()) if len(values) else None, "score_max": float(values.max()) if len(values) else None, "max_absolute_difference": float(values.max() - values.min()) if len(values) else None, "label_consistent": group.label.nunique(dropna=False) <= 1, "quality_consistent": group.quality.nunique(dropna=False) <= 1, "band_consistent": group.band.nunique(dropna=False) <= 1}
    write_csv(output_dir / "stability_runs.csv", rows); write_json(output_dir / "stability_results.json", summary); return summary


def cold_start_audit(output_dir: Path, runs: int = 3) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(runs):
        import psutil
        started = time.perf_counter_ns(); process = subprocess.Popen([sys.executable, "scripts/phase4_cold_start_worker.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True); child = psutil.Process(process.pid); peak_rss = 0
        while process.poll() is None:
            try: peak_rss = max(peak_rss, child.memory_info().rss)
            except psutil.NoSuchProcess: pass
            time.sleep(.05)
        stdout, _stderr = process.communicate(); elapsed = (time.perf_counter_ns() - started) / 1e6
        try: payload = json.loads(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError): payload = {}
        rows.append({"run": index + 1, "process_wall_ms": elapsed, "exit_status": process.returncode, "successful_fusion": payload.get("successful_fusion"), "construction_ms": payload.get("construction_ms"), "assessment_ms": payload.get("assessment_ms"), "per_modality_ms": json.dumps(payload.get("per_modality_ms", {})), "peak_child_rss_bytes": max(peak_rss, int(payload.get("rss_bytes_at_completion", 0)))})
    write_csv(output_dir / "cold_start_subprocess_runs.csv", rows); write_json(output_dir / "cold_start_subprocess_summary.json", {"runs": rows, "successful_runs": sum(row["exit_status"] == 0 and row["successful_fusion"] for row in rows)}); return {"runs": len(rows), "successful": sum(row["exit_status"] == 0 for row in rows)}


def metadata_version_comparison(output_dir: Path, external_python: str | None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "SKIPPED", "reason": "No --sklearn-142-python path was supplied."}
    if external_python:
        executable = Path(external_python)
        if not executable.exists():
            result = {"status": "SKIPPED", "reason": "The supplied external Python path does not exist.", "python": external_python}
        else:
            frame = pd.read_csv("data/processed/metadata_split_manifest.csv").query("split == 'test'").copy(); columns = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
            import joblib
            model = joblib.load("models/experiments/metadata/mvp_metadata_risk_model.pkl"); current = model.predict_proba(frame[columns])[:, 1]
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle: output_path = Path(handle.name)
            completed = subprocess.run([str(executable), "scripts/phase4_metadata_version_worker.py", "--output", str(output_path)], capture_output=True, text=True, timeout=180)
            if completed.returncode != 0:
                result = {"status": "SKIPPED", "reason": "External comparison worker failed; environment was not modified.", "python": external_python, "stderr_tail": completed.stderr[-500:]}
            else:
                external = np.asarray(json.loads(output_path.read_text(encoding="utf-8"))["scores"], dtype=float); y = frame.stroke.astype(int).to_numpy(); current_metrics = classification_metrics(y, current); external_metrics = classification_metrics(y, external); result = {"status": "COMPARED", "python": external_python, "max_probability_difference": float(np.max(np.abs(current - external))), "mean_probability_difference": float(np.mean(np.abs(current - external))), "label_agreement": float(np.mean((current >= .5) == (external >= .5))), "runtime_1_5_2": metric_dict(current_metrics), "external": metric_dict(external_metrics)}
    write_json(output_dir / "metadata_version_comparison.json", result); return result
