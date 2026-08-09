from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_apps_use_api_client_and_not_central_or_sqlite_repositories() -> None:
    for name in ("collector_app.py", "clinician_app.py"):
        imports = _imports(ROOT / "apps" / name)
        assert not any(
            "sqlite_repository" in item
            or "cases.factory" in item
            or "cases.workflow_service" in item
            for item in imports
        )
        assert "rural_stroke_assist.client" in imports


def test_clinician_app_does_not_import_inference_or_assessment_service() -> None:
    imports = _imports(ROOT / "apps" / "clinician_app.py")
    assert not any(
        "assessment" in item or "inference" in item or "fusion" in item or "model" in item
        for item in imports
    )
