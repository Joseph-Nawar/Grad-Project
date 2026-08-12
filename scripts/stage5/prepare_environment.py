"""Provision and record the isolated Stage 5 conversion environment."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.stage5.conversion import ensure_isolated_conversion_environment, stage5_workspace


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    result = ensure_isolated_conversion_environment(
        root,
        base_python=Path(__file__).resolve().parents[2].parent / "autokeras_env" / "Scripts" / "python.exe",
    )
    output = root / "reports" / "production" / "stage5" / "conversion_environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
