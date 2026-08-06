"""YAML-backed evaluation configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by clean-environment validation
    yaml = None


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 42
    bootstrap_iterations: int = 1000
    smoke_bootstrap_iterations: int = 25
    warm_runs: int = 20
    cold_runs: int = 3
    test_split: str = "test"
    output_root: str = "reports/evaluation/phase4"
    face_manifest: str = "data/processed/face_split_manifest.csv"
    speech_manifest: str = "data/processed/speech_split_manifest.csv"
    metadata_manifest: str = "data/processed/metadata_split_manifest.csv"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "EvaluationConfig":
        known = {key: values[key] for key in cls.__dataclass_fields__ if key in values and key != "extra"}
        return cls(**known, extra={key: value for key, value in values.items() if key not in known})


def load_evaluation_config(path: str | Path = "config/evaluation/phase4.yaml") -> EvaluationConfig:
    path = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the Phase 4 configuration.")
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("Phase 4 configuration must contain a mapping.")
    return EvaluationConfig.from_mapping(values)
