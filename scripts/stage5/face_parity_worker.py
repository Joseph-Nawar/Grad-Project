"""Face parity worker executed by the isolated Stage 5 environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _metrics(original_scores, candidate_scores, original_labels, candidate_labels, original_acceptance, candidate_acceptance, original_failure, candidate_failure, *, auc_labels=None):
    from scripts.stage5.parity import build_parity_metrics, parity_gate

    metrics = build_parity_metrics(
        original_scores=original_scores,
        candidate_scores=candidate_scores,
        original_labels=original_labels,
        candidate_labels=candidate_labels,
        original_auc=roc_auc_score(auc_labels if auc_labels is not None else original_labels, original_scores),
        candidate_auc=roc_auc_score(auc_labels if auc_labels is not None else original_labels, candidate_scores),
        original_acceptance=original_acceptance,
        candidate_acceptance=candidate_acceptance,
        original_failure_semantics=original_failure,
        candidate_failure_semantics=candidate_failure,
        fused_original_bands=["LOW"],
        fused_candidate_bands=["LOW"],
    )
    return {"metrics": metrics.__dict__, "gate": parity_gate(metrics)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from rural_stroke_assist.inference.face_adapter import FaceAdapter
    from rural_stroke_assist.inference.registry import BaselineRegistry
    from rural_stroke_assist.inference.runners import LiteRTRunner, ONNXRunner, OriginalFaceRunner
    from rural_stroke_assist.modeling.face_inference import preprocess_face_image

    registry = BaselineRegistry.from_file(args.root / "config" / "baseline_registry.json")
    source = registry.path_for("face")
    manifest = pd.read_csv(registry.manifest_path("face_split"))
    rows = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    candidate_runner = LiteRTRunner(args.candidate) if "litert" in args.candidate_id else ONNXRunner(args.candidate)
    original_runner = OriginalFaceRunner(source)
    direct_original: list[float] = []
    direct_candidate: list[float] = []
    direct_labels: list[int] = []
    direct_original_predictions: list[int] = []
    original_adapter_scores: list[float] = []
    candidate_adapter_scores: list[float] = []
    original_adapter_labels: list[str] = []
    candidate_adapter_labels: list[str] = []
    original_acceptance: list[bool] = []
    candidate_acceptance: list[bool] = []
    original_failure: list[str] = []
    candidate_failure: list[str] = []
    adapter_truth: list[int] = []
    original_adapter = FaceAdapter(registry=registry, runner=original_runner)
    candidate_adapter = FaceAdapter(registry=registry, runner=candidate_runner)
    for row in rows.itertuples(index=False):
        path = Path(row.path)
        tensor = preprocess_face_image(path, image_size=(160, 160))
        original_score = float(np.asarray(original_runner.run(tensor)).reshape(-1)[0])
        candidate_score = float(np.asarray(candidate_runner.run(tensor)).reshape(-1)[0])
        direct_original.append(original_score)
        direct_candidate.append(candidate_score)
        direct_labels.append(int(row.class_label == "Stroke"))
        direct_original_predictions.append(int(original_score >= 0.5))
        adapter_truth.append(int(row.class_label == "Stroke"))
        original_evidence = original_adapter.infer(path)
        candidate_evidence = candidate_adapter.infer(path)
        original_adapter_scores.append(float(original_evidence.score or 0.0))
        candidate_adapter_scores.append(float(candidate_evidence.score or 0.0))
        original_adapter_labels.append(original_evidence.label or "unavailable")
        candidate_adapter_labels.append(candidate_evidence.label or "unavailable")
        original_acceptance.append(original_evidence.available)
        candidate_acceptance.append(candidate_evidence.available)
        original_failure.append("available" if original_evidence.available else original_evidence.quality_status.value)
        candidate_failure.append("available" if candidate_evidence.available else candidate_evidence.quality_status.value)
    direct = _metrics(
        direct_original, direct_candidate, direct_original_predictions,
        [int(score >= 0.5) for score in direct_candidate],
        [True] * len(rows), [True] * len(rows), ["ok"] * len(rows), ["ok"] * len(rows),
    )
    adapter = _metrics(
        original_adapter_scores, candidate_adapter_scores, original_adapter_labels, candidate_adapter_labels,
        original_acceptance, candidate_acceptance, original_failure, candidate_failure,
        auc_labels=adapter_truth,
    )
    print(json.dumps({
        "status": "SUCCEEDED",
        "candidate_id": args.candidate_id,
        "sample_count": len(rows),
        "micro": direct,
        "adapter": adapter,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
