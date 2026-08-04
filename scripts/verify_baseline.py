"""Read-only verification for the Phase 0 RuralStroke-Assist baseline."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "baseline_registry.json"

CANONICAL_COMPONENTS = (
    "face",
    "speech",
    "metadata_context",
    "acute_symptoms",
    "fusion",
    "legacy_fusion",
    "fer2013_auxiliary",
)
REQUIRED_COMPONENTS = (
    "face",
    "speech",
    "metadata_context",
    "acute_symptoms",
    "fusion",
)
REQUIRED_MANIFESTS = ("face_split", "speech_split", "metadata_split")


@dataclass
class VerificationResult:
    """Collected verification messages and failure count."""

    messages: list[str] = field(default_factory=list)
    failed: int = 0
    required_checks: int = 0

    def check(self, label: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        suffix = f": {detail}" if detail else ""
        self.messages.append(f"[{status}] {label}{suffix}")
        self.required_checks += 1
        if not passed:
            self.failed += 1


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load the JSON registry."""
    with path.open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    validate_registry_structure(registry)
    return registry


def validate_registry_structure(registry: dict[str, Any]) -> None:
    """Validate required registry sections and component fields."""
    required_sections = {"schema_version", "project", "environment", "components", "manifests", "fusion"}
    missing = required_sections - set(registry)
    if missing:
        raise ValueError(f"Missing registry section(s): {sorted(missing)}")

    components = registry["components"]
    if set(components) != set(CANONICAL_COMPONENTS):
        raise ValueError(
            "Registry components must match canonical component set: "
            f"expected {sorted(CANONICAL_COMPONENTS)}, got {sorted(components)}"
        )

    for name, component in components.items():
        required_fields = {
            "path",
            "status",
            "framework",
            "artifact_type",
            "sha256",
            "size_bytes",
            "input_contract",
            "preprocessing_contract",
            "output_contract",
            "class_mapping",
            "thresholds",
            "dataset_provenance",
            "scientific_limitations",
            "integration_status",
        }
        missing_fields = required_fields - set(component)
        if missing_fields:
            raise ValueError(f"Component {name} missing field(s): {sorted(missing_fields)}")
        if len(component["sha256"]) != 64:
            raise ValueError(f"Component {name} has an invalid SHA256 value")
        if not isinstance(component["size_bytes"], int) or component["size_bytes"] < 0:
            raise ValueError(f"Component {name} has an invalid size_bytes value")

    for name in REQUIRED_MANIFESTS:
        manifest = registry["manifests"].get(name)
        if not manifest:
            raise ValueError(f"Missing registry manifest: {name}")
        for field_name in ("path", "sha256", "size_bytes"):
            if field_name not in manifest:
                raise ValueError(f"Manifest {name} missing field: {field_name}")

    fusion = registry["fusion"]
    for field_name in ("weights", "moderate_threshold", "high_threshold", "hard_escalation_score_floor"):
        if field_name not in fusion:
            raise ValueError(f"Missing fusion registry field: {field_name}")
    if set(fusion["weights"]) != {"face", "speech", "acute_symptoms", "metadata_context"}:
        raise ValueError("Fusion registry weights do not cover the canonical four inputs")


def _component_path(root: Path, registry: dict[str, Any], name: str) -> Path:
    return root / registry["components"][name]["path"]


def _verify_files(root: Path, registry: dict[str, Any], result: VerificationResult) -> None:
    for name, component in registry["components"].items():
        path = _component_path(root, registry, name)
        exists = path.is_file()
        result.check(f"required component file: {name}", exists, str(path))
        if not exists:
            continue
        result.check(f"component size: {name}", path.stat().st_size == component["size_bytes"])
        result.check(
            f"component SHA256: {name}",
            sha256_file(path) == component["sha256"],
            path.as_posix(),
        )

    for name, manifest in registry["manifests"].items():
        path = root / manifest["path"]
        exists = path.is_file()
        result.check(f"required manifest file: {name}", exists, str(path))
        if not exists:
            continue
        result.check(f"manifest size: {name}", path.stat().st_size == manifest["size_bytes"])
        result.check(f"manifest SHA256: {name}", sha256_file(path) == manifest["sha256"])


def _load_models(root: Path, registry: dict[str, Any], result: VerificationResult) -> None:
    try:
        import tensorflow as tf

        face = tf.keras.models.load_model(_component_path(root, registry, "face"))
        result.check("face model loads", True, face.name)
        result.check("face input structure", face.input_shape == (None, 160, 160, 3), str(face.input_shape))
        result.check("face output structure", face.output_shape == (None, 1), str(face.output_shape))
    except Exception as exc:  # pragma: no cover - exercised by environment failures
        result.check("face model loads", False, f"{type(exc).__name__}: {exc}")

    try:
        import joblib

        speech = joblib.load(_component_path(root, registry, "speech"))
        expected_speech = registry["components"]["speech"]["feature_columns"]
        actual_speech = list(speech.feature_names_in_)
        result.check("speech model loads", True, type(speech).__name__)
        result.check("speech feature contract", actual_speech == expected_speech, f"{len(actual_speech)} features")
        result.check("speech feature count", getattr(speech, "n_features_in_", None) == len(expected_speech))
    except Exception as exc:  # pragma: no cover - exercised by environment failures
        result.check("speech model loads", False, f"{type(exc).__name__}: {exc}")

    try:
        import joblib

        metadata = joblib.load(_component_path(root, registry, "metadata_context"))
        expected_metadata = registry["components"]["metadata_context"]["feature_columns"]
        actual_metadata = list(metadata.feature_names_in_)
        result.check("metadata model loads", True, type(metadata).__name__)
        result.check("metadata feature contract", actual_metadata == expected_metadata, f"{len(actual_metadata)} features")
        result.check("metadata pipeline steps", list(metadata.named_steps) == ["preprocessor", "model"])
    except Exception as exc:  # pragma: no cover - exercised by environment failures
        result.check("metadata model loads", False, f"{type(exc).__name__}: {exc}")


def _verify_runtime_contracts(registry: dict[str, Any], result: VerificationResult) -> None:
    try:
        from rural_stroke_assist.capture.acute_symptom_schema import AcuteStrokeSymptoms
        from rural_stroke_assist.modules.acute_symptom_module import assess_acute_stroke_symptoms

        acute = assess_acute_stroke_symptoms(AcuteStrokeSymptoms(face_drooping=True, arm_weakness=True))
        result.check("acute symptom smoke test", acute.hard_escalation and acute.risk_band == "URGENT")
    except Exception as exc:
        result.check("acute symptom smoke test", False, f"{type(exc).__name__}: {exc}")

    try:
        from rural_stroke_assist.modules.fusion_module import (
            DEFAULT_FUSION_WEIGHTS,
            DEFAULT_HIGH_THRESHOLD,
            DEFAULT_MODERATE_THRESHOLD,
            FusionInput,
            fuse_multimodal_scores,
        )

        expected = registry["fusion"]
        result.check("fusion weights match registry", DEFAULT_FUSION_WEIGHTS == expected["weights"])
        result.check("fusion moderate threshold matches registry", DEFAULT_MODERATE_THRESHOLD == expected["moderate_threshold"])
        result.check("fusion high threshold matches registry", DEFAULT_HIGH_THRESHOLD == expected["high_threshold"])
        fusion = fuse_multimodal_scores(FusionInput(acute_symptom_score=0.85, acute_symptom_hard_escalation=True))
        result.check("canonical fusion smoke test", fusion.risk_band == "URGENT" and fusion.fused_score >= 0.85)
    except Exception as exc:
        result.check("canonical fusion smoke test", False, f"{type(exc).__name__}: {exc}")

    result.check(
        "legacy fusion is classified as legacy",
        registry["components"]["legacy_fusion"]["status"] == "legacy",
    )


def _version_text() -> str:
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("tensorflow", "keras", "joblib", "sklearn", "pandas", "numpy", "pydantic", "pytest"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[package] = f"unavailable ({type(exc).__name__})"
    return ", ".join(f"{name}={version}" for name, version in versions.items())


def verify_all(registry_path: Path = REGISTRY_PATH) -> VerificationResult:
    """Run all required read-only checks and return the result."""
    result = VerificationResult()
    try:
        registry = load_registry(registry_path)
        result.check("registry structure", True)
    except Exception as exc:
        result.check("registry structure", False, f"{type(exc).__name__}: {exc}")
        return result

    root = registry_path.resolve().parents[1]
    result.messages.append(f"[INFO] environment: {_version_text()}")
    _verify_files(root, registry, result)
    _load_models(root, registry, result)
    _verify_runtime_contracts(registry, result)
    return result


def main() -> int:
    result = verify_all()
    print("RuralStroke-Assist Phase 0 baseline verification")
    print("=" * 52)
    print("\n".join(result.messages))
    print("=" * 52)
    print(f"Checks: {result.required_checks}; failures: {result.failed}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
