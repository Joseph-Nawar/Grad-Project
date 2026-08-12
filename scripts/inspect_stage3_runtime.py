"""Inventory the files allowed to enter the production API runtime image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_EXCLUDED_ROOTS = {
    ".git": "vcs",
    ".venv": "venv",
    "venv": "venv",
    "env": "venv",
    "data": "dataset",
    "notebooks": "notebook",
    "reports": "report",
    "runtime_data": "runtime-data",
    "codex_workspace": "codex-workspace",
    ".pytest_cache": "cache",
    ".ruff_cache": "cache",
    ".mypy_cache": "cache",
}


def _excluded_reason(relative: Path) -> str | None:
    if relative.name == ".env" or relative.name.startswith(".env."):
        return "secret"
    for part in relative.parts:
        if part in _EXCLUDED_ROOTS:
            return _EXCLUDED_ROOTS[part]
    if any(part.endswith("_env") for part in relative.parts):
        return "venv"
    if relative.suffix in {".pyc", ".pyo", ".ipynb_checkpoints"}:
        return "cache"
    if "__pycache__" in relative.parts:
        return "cache"
    return None


def build_runtime_inventory(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    source: list[str] = []
    models: list[str] = []
    excluded: dict[str, str] = {}
    for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        relative = path.relative_to(root_path)
        relative_text = relative.as_posix()
        reason = _excluded_reason(relative)
        if reason:
            excluded[relative_text] = reason
        elif relative.parts and relative.parts[0] == "models":
            models.append(relative_text)
        else:
            source.append(relative_text)
    return {"source": source, "models": models, "excluded": excluded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_runtime_inventory(args.root), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
