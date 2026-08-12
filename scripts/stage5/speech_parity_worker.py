"""Speech parity worker executed inside isolated Stage 5 environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from rural_stroke_assist.features.speech_features import extract_basic_audio_features, extract_mfcc_summary_features, load_audio
    from rural_stroke_assist.inference.registry import BaselineRegistry
    from rural_stroke_assist.inference.runners import ONNXRunner, OriginalTabularRunner
    from rural_stroke_assist.inference.speech_adapter import SpeechAdapter
    from scripts.stage5.parity import build_parity_metrics, parity_gate

    registry = BaselineRegistry.from_file(args.root / "config" / "baseline_registry.json")
    columns = registry.component("speech").feature_columns
    original_runner = OriginalTabularRunner(registry.path_for("speech"))
    candidate_runner = ONNXRunner(args.candidate)
    feature_table = pd.read_csv(args.root / "data" / "processed" / "speech_features.csv")
    rows = feature_table.loc[feature_table["split"] == "test"].reset_index(drop=True)
    original_scores: list[float] = []
    candidate_scores: list[float] = []
    original_labels: list[int] = []
    candidate_labels: list[int] = []
    for index in range(len(rows)):
        frame = rows.loc[[index], columns]
        original_probabilities, classes = original_runner.run(frame)
        candidate_probabilities, candidate_classes = candidate_runner.run(frame)
        original_score = float(original_probabilities[0, list(classes).index(1)])
        candidate_score = float(candidate_probabilities[0, list(candidate_classes).index(1)])
        original_scores.append(original_score)
        candidate_scores.append(candidate_score)
        original_labels.append(int(original_score >= 0.5))
        candidate_labels.append(int(candidate_score >= 0.5))
    micro = build_parity_metrics(
        original_scores=original_scores, candidate_scores=candidate_scores,
        original_labels=original_labels, candidate_labels=candidate_labels,
        original_auc=roc_auc_score(rows["label_encoded"], original_scores), candidate_auc=roc_auc_score(rows["label_encoded"], candidate_scores),
        original_acceptance=[True] * len(rows), candidate_acceptance=[True] * len(rows),
        original_failure_semantics=["ok"] * len(rows), candidate_failure_semantics=["ok"] * len(rows),
        fused_original_bands=["LOW"], fused_candidate_bands=["LOW"],
    )
    audio_rows = pd.read_csv(registry.manifest_path("speech_split"))
    audio_rows = audio_rows.loc[audio_rows["split"] == "test"].reset_index(drop=True)
    original_adapter = SpeechAdapter(registry=registry, runner=original_runner)
    candidate_adapter = SpeechAdapter(registry=registry, runner=candidate_runner)
    adapter_original_scores: list[float] = []
    adapter_candidate_scores: list[float] = []
    adapter_original_labels: list[str] = []
    adapter_candidate_labels: list[str] = []
    adapter_original_acceptance: list[bool] = []
    adapter_candidate_acceptance: list[bool] = []
    adapter_original_failure: list[str] = []
    adapter_candidate_failure: list[str] = []
    for row in audio_rows.itertuples(index=False):
        path = Path(row.path)
        original = original_adapter.infer(path)
        candidate = candidate_adapter.infer(path)
        adapter_original_scores.append(float(original.score or 0.0))
        adapter_candidate_scores.append(float(candidate.score or 0.0))
        adapter_original_labels.append(original.label or "unavailable")
        adapter_candidate_labels.append(candidate.label or "unavailable")
        adapter_original_acceptance.append(original.available)
        adapter_candidate_acceptance.append(candidate.available)
        adapter_original_failure.append("available" if original.available else original.quality_status.value)
        adapter_candidate_failure.append("available" if candidate.available else candidate.quality_status.value)
    adapter = build_parity_metrics(
        original_scores=adapter_original_scores, candidate_scores=adapter_candidate_scores,
        original_labels=adapter_original_labels, candidate_labels=adapter_candidate_labels,
        original_auc=roc_auc_score((audio_rows["label"] == "dysarthric").astype(int), adapter_original_scores), candidate_auc=roc_auc_score((audio_rows["label"] == "dysarthric").astype(int), adapter_candidate_scores),
        original_acceptance=adapter_original_acceptance, candidate_acceptance=adapter_candidate_acceptance,
        original_failure_semantics=adapter_original_failure, candidate_failure_semantics=adapter_candidate_failure,
        fused_original_bands=["LOW"], fused_candidate_bands=["LOW"],
    )
    print(json.dumps({"status": "SUCCEEDED", "micro": {"metrics": micro.__dict__, "gate": parity_gate(micro)}, "adapter": {"metrics": adapter.__dict__, "gate": parity_gate(adapter)}, "micro_sample_count": len(rows), "adapter_sample_count": len(audio_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
