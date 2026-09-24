"""Prepare or verify the external resources for pretrained_reference.

This explicit setup command may download the pinned upstream resources. Normal
assessment construction uses local-only validation through the reference
registry and never invokes this module's preparation path.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rural_stroke_assist.inference.reference_registry import ReferenceProfileRegistry


def _tabpfn_target(registry: ReferenceProfileRegistry) -> Path:
    filename = str(registry.component("metadata_context").data["checkpoint_filename"])
    explicit = os.environ.get("RURALSTROKE_TABPFN_CHECKPOINT")
    if explicit:
        return Path(explicit)
    cache = os.environ.get("RURALSTROKE_TABPFN_CACHE_DIR")
    if cache:
        return Path(cache) / filename
    return Path.home() / "AppData" / "Roaming" / "tabpfn" / filename


def prepare_external_resources(registry: ReferenceProfileRegistry) -> dict[str, Any]:
    registry.validate_project_artifacts()
    speech_snapshot = registry.resolve_speech_snapshot(local_only=False)

    metadata = registry.component("metadata_context").data
    target = _tabpfn_target(registry)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import download_model
    except Exception as exc:
        raise RuntimeError("Unable to prepare the pinned TabPFN V2 checkpoint.") from exc
    package_version = importlib.metadata.version("tabpfn")
    if package_version != str(metadata["package_version"]):
        raise RuntimeError(
            f"TabPFN package mismatch: expected {metadata['package_version']}, got {package_version}"
        )
    result = download_model(target, version=ModelVersion.V2, which="classifier")
    if result != "ok":
        raise RuntimeError(f"TabPFN V2 download failed: {result!r}")

    os.environ["RURALSTROKE_TABPFN_CHECKPOINT"] = str(target)
    resources = registry.validate_external_resources(local_only=True)
    return {
        **resources,
        "tabpfn_package": importlib.metadata.version("tabpfn"),
        "tabpfn_model_version": str(metadata["model_version"]),
        "tabpfn_checkpoint_sha256": str(metadata["checkpoint_sha256"]),
        "license_notice": metadata["model_weight_license"],
        "speech_upstream_license": registry.component("speech").data["upstream_license"],
    }


def verify_external_resources(registry: ReferenceProfileRegistry) -> dict[str, Any]:
    registry.validate_project_artifacts()
    resources = registry.validate_external_resources(local_only=True)
    metadata = registry.component("metadata_context").data
    return {
        **resources,
        "tabpfn_package": importlib.metadata.version("tabpfn"),
        "tabpfn_model_version": str(metadata["model_version"]),
        "tabpfn_checkpoint_sha256": str(metadata["checkpoint_sha256"]),
        "license_notice": metadata["model_weight_license"],
        "speech_upstream_license": registry.component("speech").data["upstream_license"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only validate already prepared local resources; never download.",
    )
    args = parser.parse_args()
    registry = ReferenceProfileRegistry.from_file()
    result = (
        verify_external_resources(registry)
        if args.verify_only
        else prepare_external_resources(registry)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
