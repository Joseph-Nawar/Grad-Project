"""Safely inspect or clear only the repository runtime_data directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default = root / "runtime_data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", default=str(default))
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    target = Path(args.runtime_dir).expanduser().resolve()
    if (
        target != default.resolve()
        or target.name != "runtime_data"
        or root.resolve() not in target.parents
    ):
        raise SystemExit(
            "Refusing: runtime directory must be the repository-local runtime_data directory."
        )
    entries = list(target.iterdir()) if target.exists() else []
    print(f"Target: {target}")
    print(f"Entries: {len(entries)}")
    if not args.confirm:
        print("Dry run only. Re-run with --confirm to delete these runtime entries.")
        return 0
    for entry in entries:
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    print(f"Removed {len(entries)} runtime entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
