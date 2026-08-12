from pathlib import Path

import pytest
from scripts.check_repo_hygiene import CANONICAL_MODEL_PATHS, scan_repository


def _write_canonical_models(root: Path) -> None:
    for relative_path in CANONICAL_MODEL_PATHS:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"canonical")


def test_clean_source_boundary_passes_without_git_metadata(tmp_path: Path) -> None:
    _write_canonical_models(tmp_path)
    (tmp_path / "rural_stroke_assist" / "source.py").parent.mkdir()
    (tmp_path / "rural_stroke_assist" / "source.py").write_text("source\n", encoding="utf-8")

    report = scan_repository(tmp_path)

    assert report.ok
    assert report.missing_canonical_models == ()
    assert report.source_boundary_violations == ()


def test_forbidden_source_artifacts_fail_the_source_boundary(tmp_path: Path) -> None:
    _write_canonical_models(tmp_path)
    (tmp_path / "rural_stroke_assist" / "checkpoint.weights.h5").parent.mkdir()
    (tmp_path / "rural_stroke_assist" / "checkpoint.weights.h5").write_bytes(b"bad")
    (tmp_path / "rural_stroke_assist" / "cases.sqlite3").write_bytes(b"bad")

    report = scan_repository(tmp_path)

    assert not report.ok
    assert "rural_stroke_assist/checkpoint.weights.h5" in report.source_boundary_violations
    assert "rural_stroke_assist/cases.sqlite3" in report.source_boundary_violations


def test_known_local_artifacts_are_reported_but_not_source_violations(tmp_path: Path) -> None:
    _write_canonical_models(tmp_path)
    local_model = (
        tmp_path / "models" / "experiments" / "face" / "trial_002" / "checkpoint.weights.h5"
    )
    local_model.parent.mkdir(parents=True, exist_ok=True)
    local_model.write_bytes(b"training")
    (tmp_path / "runtime_data" / "cases.sqlite3").parent.mkdir()
    (tmp_path / "runtime_data" / "cases.sqlite3").write_bytes(b"runtime")

    report = scan_repository(tmp_path, large_threshold_bytes=1)

    assert report.ok
    assert "models/experiments/face/trial_002/checkpoint.weights.h5" in report.local_artifacts
    assert "runtime_data/cases.sqlite3" in report.local_artifacts
    assert (
        "models/experiments/face/trial_002/checkpoint.weights.h5" in report.suspicious_large_files
    )


def test_git_metadata_root_is_rejected_without_inspection(tmp_path: Path) -> None:
    git_root = tmp_path / ".git"
    git_root.mkdir()
    (git_root / "sensitive-object").write_bytes(b"must not be read")

    with pytest.raises(ValueError, match="refusing to inspect .git"):
        scan_repository(git_root)
