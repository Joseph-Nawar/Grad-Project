"""Idempotently provision a repository-local Python 3.11 environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--venv", default=".venv", help="Repository-relative or absolute virtual-environment path."
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = Path(args.venv)
    environment = environment if environment.is_absolute() else root / environment
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not args.verify_only and not python.exists():
        print(f"Creating {environment}")
        venv.EnvBuilder(with_pip=True).create(environment)
    if not python.exists():
        raise SystemExit(f"Python executable not found: {python}")
    if not args.verify_only and not args.skip_install:
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-r",
                str(root / "requirements-baseline.txt"),
                "-c",
                str(root / "constraints-baseline.txt"),
            ],
            check=True,
        )
        subprocess.run([str(python), "-m", "pip", "install", "-e", str(root)], check=True)
    verify_python = python
    probe = subprocess.run([str(python), "-c", "import yaml"], capture_output=True)
    if probe.returncode != 0:
        if args.skip_install:
            verify_python = Path(sys.executable)
            print(
                "Target environment lacks optional verification dependencies; using the current Python for release validation."
            )
        else:
            raise SystemExit(
                "The provisioned environment could not import PyYAML after installation."
            )
    subprocess.run(
        [str(verify_python), str(root / "scripts" / "check_release_readiness.py")], check=True
    )
    print("Local environment is ready.")
    print("Launch both apps with: python scripts/run_phase3_apps.py")
    print("Run baseline verification with: python scripts/verify_baseline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
