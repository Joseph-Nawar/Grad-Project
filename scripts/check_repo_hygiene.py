"""Check the filesystem source boundary without consulting Git metadata."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MODEL_PATHS = frozenset(
    {
        "models/experiments/face/trial_003_mobilenetv2_balanced_160/model.keras",
        "models/experiments/speech/trial_001_mfcc_random_forest/model.pkl",
        "models/experiments/metadata/mvp_metadata_risk_model.pkl",
        "models/experiments/fer2013/trial_001_mobilenetv2_160/model.keras",
    }
)
ALLOWED_NON_MODEL_PATHS = frozenset({"models/.gitkeep"})
LOCAL_ARTIFACT_ROOTS = frozenset(
    {
        ".phase5-test-venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        "venv",
        "env",
        "autokeras_env",
        "pycaret_env",
        "data",
        "runtime_data",
        ".runtime-secrets",
        "build",
        "dist",
        "downloads",
    }
)
LOCAL_ARTIFACT_SEGMENTS = frozenset({".terraform", "__pycache__", ".ipynb_checkpoints"})
MODEL_SUFFIXES = (".keras", ".h5", ".pkl", ".joblib")
FORBIDDEN_SUFFIXES = (
    ".tfstate",
    ".tfstate.backup",
    ".sqlite",
    ".sqlite3",
    ".sqlite3-shm",
    ".sqlite3-wal",
    ".db",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".img",
    ".qcow2",
    ".log",
)
FORBIDDEN_NAMES = frozenset(
    {
        "checkpoint.weights.h5",
        "crash.log",
        "terraform.tfstate",
        "terraform.tfstate.backup",
    }
)


@dataclass(frozen=True)
class HygieneReport:
    """Filesystem-only source-boundary findings."""

    missing_canonical_models: tuple[str, ...]
    noncanonical_model_files: tuple[str, ...]
    source_boundary_violations: tuple[str, ...]
    local_artifacts: tuple[str, ...]
    local_artifact_roots: tuple[str, ...]
    suspicious_large_files: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether the source boundary is valid."""
        return not self.missing_canonical_models and not self.source_boundary_violations


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield files while pruning `.git` before it is traversed."""
    for current, directories, filenames in os.walk(root):
        directories[:] = [name for name in directories if name != ".git"]
        current_path = Path(current)
        for filename in filenames:
            yield current_path / filename


def _is_terraform_local_artifact(parts: tuple[str, ...], name: str) -> bool:
    return (
        len(parts) >= 2
        and parts[0] == "infra"
        and parts[1] == "terraform"
        and (
            ".terraform" in parts or name == "crash.log" or ".tfstate" in name or ".tfvars" in name
        )
    )


def _is_known_local_artifact(parts: tuple[str, ...], relative: str, name: str) -> bool:
    if parts and parts[0] in LOCAL_ARTIFACT_ROOTS:
        return True
    if any(segment in LOCAL_ARTIFACT_SEGMENTS for segment in parts):
        return True
    if _is_terraform_local_artifact(parts, name):
        return True
    if name.lower().endswith((".log", ".pyc", ".pyo")):
        return True
    if (
        parts
        and parts[0] == "models"
        and relative not in CANONICAL_MODEL_PATHS | ALLOWED_NON_MODEL_PATHS
    ):
        return True
    return False


def _is_forbidden(relative: str, parts: tuple[str, ...], name: str) -> bool:
    lower_name = name.lower()
    lower_relative = relative.lower()
    if any(segment in LOCAL_ARTIFACT_SEGMENTS for segment in parts):
        return True
    if lower_name in FORBIDDEN_NAMES:
        return True
    if any(lower_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        return True
    if (
        any(segment == ".terraform" for segment in Path(relative).parts)
        or "checkpoint.weights.h5" in lower_relative
    ):
        return True
    if (
        parts
        and parts[0] == "models"
        and relative not in CANONICAL_MODEL_PATHS | ALLOWED_NON_MODEL_PATHS
    ):
        return True
    if lower_name.endswith(MODEL_SUFFIXES) and relative not in CANONICAL_MODEL_PATHS:
        return True
    return False


def scan_repository(
    root: Path = ROOT, *, large_threshold_bytes: int = 50 * 1024 * 1024
) -> HygieneReport:
    """Scan normal filesystem files and never inspect `.git`."""
    root = root.resolve()
    if root.name == ".git":
        raise ValueError("refusing to inspect .git")
    present = {_relative_path(root, path) for path in _iter_files(root)}
    missing = tuple(sorted(CANONICAL_MODEL_PATHS - present))
    noncanonical_models: list[str] = []
    source_violations: list[str] = []
    local_artifacts: list[str] = []
    local_artifact_roots: set[str] = set()
    suspicious_large: list[str] = []

    for path in _iter_files(root):
        relative = _relative_path(root, path)
        parts = tuple(Path(relative).parts)
        name = path.name
        if (
            parts
            and parts[0] == "models"
            and relative not in CANONICAL_MODEL_PATHS | ALLOWED_NON_MODEL_PATHS
        ):
            noncanonical_models.append(relative)
        if path.stat().st_size > large_threshold_bytes and relative not in CANONICAL_MODEL_PATHS:
            suspicious_large.append(relative)
        known_local_artifact = _is_known_local_artifact(parts, relative, name)
        if known_local_artifact:
            if parts and parts[0] in LOCAL_ARTIFACT_ROOTS:
                local_artifact_roots.add(parts[0])
            elif ".terraform" in parts:
                local_artifact_roots.add(".terraform")
            elif any(segment in LOCAL_ARTIFACT_SEGMENTS for segment in parts):
                local_artifact_roots.update(
                    segment for segment in parts if segment in LOCAL_ARTIFACT_SEGMENTS
                )
            elif parts and parts[0] == "models":
                local_artifact_roots.add("models")
        if not _is_forbidden(relative, parts, name):
            continue
        if known_local_artifact:
            if parts and parts[0] == "models":
                local_artifacts.append(relative)
            elif (
                any(relative.lower().endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
                or name.lower() in FORBIDDEN_NAMES
            ):
                local_artifacts.append(relative)
        else:
            source_violations.append(relative)

    return HygieneReport(
        missing_canonical_models=tuple(sorted(missing)),
        noncanonical_model_files=tuple(sorted(noncanonical_models)),
        source_boundary_violations=tuple(sorted(source_violations)),
        local_artifacts=tuple(sorted(local_artifacts)),
        local_artifact_roots=tuple(sorted(local_artifact_roots)),
        suspicious_large_files=tuple(sorted(suspicious_large)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when known local generated/training/runtime artifacts are present",
    )
    args = parser.parse_args(argv)
    report = scan_repository(args.root)
    payload = asdict(report) | {
        "ok": report.ok and (not args.strict or not report.local_artifact_roots)
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
