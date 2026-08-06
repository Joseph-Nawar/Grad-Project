"""Safe immutable evaluation output writing."""
from __future__ import annotations
from pathlib import Path
import csv
import json
from typing import Any, Iterable


def prepare_output(directory: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(directory)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    fields = fieldnames or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
