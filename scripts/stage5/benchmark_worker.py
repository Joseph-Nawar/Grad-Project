"""Single clean-process Stage 5 benchmark worker."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from rural_stroke_assist.inference.registry import BaselineRegistry
    from rural_stroke_assist.inference.runners import LiteRTRunner, ONNXRunner, OriginalFaceRunner, OriginalTabularRunner
    from rural_stroke_assist.inference.face_adapter import FaceAdapter
    from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
    from rural_stroke_assist.inference.metadata_adapter import MetadataAdapter, MetadataInput
    from rural_stroke_assist.modeling.face_inference import preprocess_face_image

    registry = BaselineRegistry.from_file(args.root / "config" / "baseline_registry.json")
    root = args.root
    if args.target == "face-original":
        runner = OriginalFaceRunner(registry.path_for("face")); value = preprocess_face_image(pd.read_csv(registry.manifest_path("face_split")).loc[lambda f: f["split"] == "test", "path"].iloc[0])
        invoke = lambda: runner.run(value)
        artifact = registry.path_for("face")
    elif args.target == "face-litert-fp32":
        runner = LiteRTRunner(root / ".stage5/candidates/face/face-litert-fp32.tflite"); value = preprocess_face_image(pd.read_csv(registry.manifest_path("face_split")).loc[lambda f: f["split"] == "test", "path"].iloc[0]); invoke = lambda: runner.run(value); artifact = runner.path
    elif args.target == "face-onnx-fp32":
        runner = ONNXRunner(root / ".stage5/candidates/face/face-onnx-fp32.onnx"); value = preprocess_face_image(pd.read_csv(registry.manifest_path("face_split")).loc[lambda f: f["split"] == "test", "path"].iloc[0]); invoke = lambda: runner.run(value); artifact = runner.path
    elif args.target == "speech-original":
        runner = OriginalTabularRunner(registry.path_for("speech")); columns = registry.component("speech").feature_columns; value = pd.read_csv(root / "data/processed/speech_features.csv").loc[lambda f: f["split"] == "test", columns].iloc[[0]]; invoke = lambda: runner.run(value); artifact = runner.path
    elif args.target == "speech-onnx":
        runner = ONNXRunner(root / ".stage5/candidates/speech/speech-onnx-random-forest-pipeline.onnx"); columns = registry.component("speech").feature_columns; value = pd.read_csv(root / "data/processed/speech_features.csv").loc[lambda f: f["split"] == "test", columns].iloc[[0]]; invoke = lambda: runner.run(value); artifact = runner.path
    elif args.target == "metadata-original":
        row = pd.read_csv(registry.manifest_path("metadata_split")).loc[lambda f: f["split"] == "test"].iloc[0]
        value = MetadataInput(age=row.age, hypertension=row.hypertension, heart_disease=row.heart_disease, avg_glucose_level=row.avg_glucose_level, bmi=row.bmi, gender=row.gender, ever_married=row.ever_married, work_type=row.work_type, Residence_type=row.Residence_type, smoking_status=row.smoking_status)
        adapter = MetadataAdapter(registry=registry)
        invoke = lambda: adapter.infer(value)
        artifact = registry.path_for("metadata_context")
    else:
        raise ValueError(f"unknown benchmark target {args.target}")
    process = psutil.Process()
    cold_started = time.perf_counter()
    invoke()
    cold_ms = (time.perf_counter() - cold_started) * 1000
    warm = []
    for _ in range(30):
        started = time.perf_counter(); invoke(); warm.append((time.perf_counter() - started) * 1000)
    print(json.dumps({"status": "SUCCEEDED", "artifact_bytes": artifact.stat().st_size, "cold_start_to_first_prediction_ms": cold_ms, "warm_p50_ms": statistics.median(warm), "warm_p95_ms": float(np.percentile(warm, 95)), "rss_bytes": process.memory_info().rss, "sample_count": 30}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
