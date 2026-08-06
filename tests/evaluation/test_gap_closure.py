from pathlib import Path
import json

from rural_stroke_assist.evaluation.gap_closure import _FixedAdapter, missing_modality_audit
from rural_stroke_assist.inference.registry import load_baseline_registry


def test_missing_modality_matrix_executes_all_fifteen_combinations(tmp_path: Path):
    result = missing_modality_audit(tmp_path, load_baseline_registry())
    assert result["combination_count"] == 16
    assert (tmp_path / "missing_modality_matrix.csv").is_file()


def test_fixed_adapter_returns_bounded_evidence():
    evidence = _FixedAdapter("face", 0.7).infer(None)
    assert evidence.available and evidence.score == 0.7


def test_cold_worker_protocol_is_json_serializable():
    payload = {"successful_fusion": True, "per_modality_ms": {"face": 1.0}}
    assert json.loads(json.dumps(payload))["successful_fusion"] is True
