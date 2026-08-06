"""Reusable lightweight latency and RSS profiling helpers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import statistics
import time
from collections.abc import Callable

import psutil


@dataclass(frozen=True)
class ProfileResult:
    cold_runs: int
    warm_runs: int
    warm_median_ms: float
    warm_p95_ms: float
    peak_rss_bytes: int
    memory_increase_bytes: int


def profile_callable(function: Callable[[], object], *, warm_runs: int = 20, cold_runs: int = 0) -> ProfileResult:
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss
    for _ in range(cold_runs):
        function()
    durations = []
    peak = before
    for _ in range(warm_runs):
        start = time.perf_counter_ns(); function(); durations.append((time.perf_counter_ns() - start) / 1e6); peak = max(peak, process.memory_info().rss)
    return ProfileResult(cold_runs, warm_runs, statistics.median(durations) if durations else 0.0, float(sorted(durations)[max(0, int(len(durations) * .95) - 1)]) if durations else 0.0, peak, max(0, peak - before))


def profile_default_assessment(*, warm_runs: int = 20, cold_runs: int = 3) -> dict[str, object]:
    """Profile a fixed, local assessment input without exposing patient data."""
    import pandas as pd
    from rural_stroke_assist.assessment.factory import create_default_assessment_service
    from rural_stroke_assist.capture.schemas import AssessmentInput
    from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
    face = str(pd.read_csv("data/processed/face_split_manifest.csv").query("split == 'test'").iloc[0].path)
    speech = str(pd.read_csv("data/processed/speech_split_manifest.csv").query("split == 'test'").iloc[0].path)
    row = pd.read_csv("data/processed/metadata_split_manifest.csv").query("split == 'test'").iloc[0]
    from rural_stroke_assist.inference.metadata_adapter import MetadataInput
    clean = lambda value: None if pd.isna(value) else value
    metadata = MetadataInput.model_validate({"age": clean(row.age), "hypertension": clean(row.hypertension), "heart_disease": clean(row.heart_disease), "avg_glucose_level": clean(row.avg_glucose_level), "bmi": clean(row.bmi), "gender": clean(row.gender), "ever_married": clean(row.ever_married), "work_type": clean(row.work_type), "residence_type": clean(row.Residence_type), "smoking_status": clean(row.smoking_status)})
    value = AssessmentInput(session_id="phase4-fixed-input", face_image_path=face, speech_audio_path=speech, metadata=metadata, acute_symptoms=AcuteStrokeSymptoms(face_drooping=True, symptom_onset_minutes=20))
    service = create_default_assessment_service()
    profile = profile_callable(lambda: service.assess(value), warm_runs=warm_runs, cold_runs=cold_runs)
    adapter_profiles = {}
    for name, adapter, input_value in (("face", service.adapters["face"], face), ("speech", service.adapters["speech"], speech), ("metadata_context", service.adapters["metadata_context"], metadata)):
        item = profile_callable(lambda adapter=adapter, input_value=input_value: adapter.infer(input_value), warm_runs=warm_runs, cold_runs=cold_runs)
        adapter_profiles[name] = item.__dict__
    return {"assessment_service": profile.__dict__, "adapters": adapter_profiles, "model_sizes_bytes": {name: Path(path).stat().st_size for name, path in (("face", "models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras"), ("speech", "models/experiments/speech/trial_001_mfcc_random_forest/model.pkl"), ("metadata_context", "models/experiments/metadata/mvp_metadata_risk_model.pkl"))}}
