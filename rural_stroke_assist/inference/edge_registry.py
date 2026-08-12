"""Promotion-registry access for the explicit optimized profile."""

from __future__ import annotations

from pathlib import Path

from rural_stroke_assist.inference.exceptions import ArtifactConfigurationError
from rural_stroke_assist.inference.runners import LiteRTRunner, ONNXRunner, OriginalFaceRunner, OriginalTabularRunner
from rural_stroke_assist.stage5.contracts import Stage5Registry


def load_edge_registry(path: str | Path | None = None) -> Stage5Registry:
    registry_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "config" / "edge_runtime_registry.json"
    if not registry_path.is_file():
        raise ArtifactConfigurationError(
            f"Optimized runtime promotion registry is not available: {registry_path}"
        )
    try:
        return Stage5Registry.read(registry_path)
    except Exception as exc:
        raise ArtifactConfigurationError("Optimized runtime promotion registry is invalid.") from exc


def build_optimized_runners():
    registry = load_edge_registry()
    if registry.runtime_profile != "optimized":
        raise ArtifactConfigurationError("Edge promotion registry is not approved for optimized runtime.")
    root = Path(__file__).resolve().parents[2]
    baseline = __import__("rural_stroke_assist.inference.registry", fromlist=["load_baseline_registry"]).load_baseline_registry()
    runners = {}
    for modality, entry in registry.modalities.items():
        artifact = root / str(entry["artifact"])
        if not artifact.is_file():
            raise ArtifactConfigurationError(f"Promoted edge artifact is missing: {artifact}")
        if entry["decision"] == "ADOPTED":
            if entry["backend"] == "LiteRT":
                runners[modality] = LiteRTRunner(artifact)
            elif entry["backend"] == "ONNX":
                runners[modality] = ONNXRunner(artifact)
            else:
                raise ArtifactConfigurationError(f"Unsupported promoted backend: {entry['backend']}")
        elif entry["decision"] == "ORIGINAL_RETAINED":
            original = baseline.path_for(modality)
            runners[modality] = OriginalFaceRunner(original) if modality == "face" else OriginalTabularRunner(original)
        else:
            raise ArtifactConfigurationError(f"Optimized profile contains non-promoted decision: {entry['decision']}")
    required = {"face", "speech", "metadata_context"}
    if set(runners) != required:
        raise ArtifactConfigurationError("Optimized promotion registry must declare all learned modalities.")
    return runners
