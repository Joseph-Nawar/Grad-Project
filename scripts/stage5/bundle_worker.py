"""Representative full-assessment bundle benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", choices=["baseline", "mixed-litert", "mixed-onnx"], required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from rural_stroke_assist.assessment.service import AssessmentService
    from rural_stroke_assist.assessment.fusion_strategy import CanonicalLateFusionStrategy
    from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
    from rural_stroke_assist.capture.schemas import AssessmentInput
    from rural_stroke_assist.inference.face_adapter import FaceAdapter
    from rural_stroke_assist.inference.metadata_adapter import MetadataAdapter, MetadataInput
    from rural_stroke_assist.inference.registry import BaselineRegistry
    from rural_stroke_assist.inference.runners import LiteRTRunner, ONNXRunner, OriginalFaceRunner, OriginalTabularRunner
    from rural_stroke_assist.inference.speech_adapter import SpeechAdapter

    root = args.root
    registry = BaselineRegistry.from_file(root / "config/baseline_registry.json")
    face = pd.read_csv(registry.manifest_path("face_split")); face_path = Path(face.loc[face["split"] == "test", "path"].iloc[0])
    speech = pd.read_csv(registry.manifest_path("speech_split")); speech_path = Path(speech.loc[speech["split"] == "test", "path"].iloc[0])
    meta = pd.read_csv(registry.manifest_path("metadata_split")); row = meta.loc[meta["split"] == "test"].iloc[0]
    metadata = MetadataInput(age=row.age, hypertension=row.hypertension, heart_disease=row.heart_disease, avg_glucose_level=row.avg_glucose_level, bmi=row.bmi, gender=row.gender, ever_married=row.ever_married, work_type=row.work_type, Residence_type=row.Residence_type, smoking_status=row.smoking_status)
    if args.bundle == "baseline":
        face_runner = OriginalFaceRunner(registry.path_for("face")); speech_runner = OriginalTabularRunner(registry.path_for("speech"))
    elif args.bundle == "mixed-litert":
        face_runner = LiteRTRunner(root / ".stage5/candidates/face/face-litert-fp32.tflite"); speech_runner = ONNXRunner(root / ".stage5/candidates/speech/speech-onnx-random-forest-pipeline.onnx")
    else:
        face_runner = ONNXRunner(root / ".stage5/candidates/face/face-onnx-fp32.onnx"); speech_runner = ONNXRunner(root / ".stage5/candidates/speech/speech-onnx-random-forest-pipeline.onnx")
    service = AssessmentService(adapters={"face": FaceAdapter(registry=registry, runner=face_runner), "speech": SpeechAdapter(registry=registry, runner=speech_runner), "metadata_context": MetadataAdapter(registry=registry), "acute_symptoms": __import__("rural_stroke_assist.inference.symptom_adapter", fromlist=["SymptomAdapter"]).SymptomAdapter(registry=registry)}, fusion_strategy=CanonicalLateFusionStrategy(), runtime_profile=args.bundle)
    value = AssessmentInput(session_id=f"stage5-{args.bundle}", face_image_path=face_path, speech_audio_path=speech_path, metadata=metadata, acute_symptoms=AcuteStrokeSymptoms(face_drooping=True, arm_weakness=True))
    process = psutil.Process()
    started = time.perf_counter(); service.assess(value); cold_ms = (time.perf_counter() - started) * 1000
    warm = []
    for _ in range(10):
        started = time.perf_counter(); result = service.assess(value); warm.append((time.perf_counter() - started) * 1000)
    sizes = [registry.path_for("face").stat().st_size, registry.path_for("speech").stat().st_size, registry.path_for("metadata_context").stat().st_size]
    if args.bundle == "mixed-litert":
        sizes = [(root / ".stage5/candidates/face/face-litert-fp32.tflite").stat().st_size, (root / ".stage5/candidates/speech/speech-onnx-random-forest-pipeline.onnx").stat().st_size, sizes[2]]
    elif args.bundle == "mixed-onnx":
        sizes = [(root / ".stage5/candidates/face/face-onnx-fp32.onnx").stat().st_size, (root / ".stage5/candidates/speech/speech-onnx-random-forest-pipeline.onnx").stat().st_size, sizes[2]]
    print(json.dumps({"status": "SUCCEEDED", "cold_start_to_first_assessment_ms": cold_ms, "warm_p50_assessment_ms": float(np.percentile(warm, 50)), "warm_p95_assessment_ms": float(np.percentile(warm, 95)), "rss_bytes": process.memory_info().rss, "artifact_bytes": sum(sizes), "risk_band": result.fusion.risk_band if result.fusion else None, "warm_samples": len(warm)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
