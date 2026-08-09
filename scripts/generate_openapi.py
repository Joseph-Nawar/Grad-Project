"""Generate or check the deterministic Stage 2 OpenAPI snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rural_stroke_assist.server.app import create_app

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs" / "api" / "openapi-v1.json"


def rendered_snapshot() -> str:
    app = create_app(initialize_resources=False)
    document = app.openapi()
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = rendered_snapshot()
    if args.check:
        if not SNAPSHOT.is_file() or SNAPSHOT.read_text(encoding="utf-8") != rendered:
            print(f"OpenAPI snapshot is stale: {SNAPSHOT}")
            return 1
        print("OpenAPI snapshot is current")
        return 0
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {SNAPSHOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
