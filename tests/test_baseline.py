import json
from pathlib import Path

import pytest

from scripts.verify_baseline import (
    CANONICAL_COMPONENTS,
    load_registry,
    sha256_file,
    validate_registry_structure,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "baseline_registry.json"


def test_registry_contains_all_canonical_components() -> None:
    registry = load_registry(REGISTRY_PATH)

    assert set(registry["components"]) == set(CANONICAL_COMPONENTS)
    assert registry["fusion"]["weights"] == {
        "face": 0.35,
        "speech": 0.30,
        "acute_symptoms": 0.25,
        "metadata_context": 0.10,
    }


def test_registry_structure_rejects_missing_required_sections(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps({"components": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="registry section"):
        validate_registry_structure(json.loads(invalid_path.read_text(encoding="utf-8")))


def test_registry_hash_matches_canonical_face_manifest() -> None:
    registry = load_registry(REGISTRY_PATH)
    manifest = ROOT / registry["manifests"]["face_split"]["path"]

    assert sha256_file(manifest) == registry["manifests"]["face_split"]["sha256"]


def test_registry_classifies_new_fusion_as_canonical_and_legacy_as_legacy() -> None:
    registry = load_registry(REGISTRY_PATH)

    assert registry["components"]["fusion"]["status"] == "canonical"
    assert registry["components"]["legacy_fusion"]["status"] == "legacy"


@pytest.mark.integration
def test_selected_artifacts_load_and_match_registry() -> None:
    from scripts.verify_baseline import verify_all

    result = verify_all(ROOT / "config" / "baseline_registry.json")

    assert result.failed == 0, "\n".join(result.messages)
    assert result.required_checks > 0
