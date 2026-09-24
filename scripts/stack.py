"""Cross-platform local Stage 3 Compose lifecycle command.

This module intentionally uses only the Python standard library so a clean
checkout needs only Python and Docker installed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets as random_secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "compose.yaml"
SECRET_DIR = ROOT / ".runtime-secrets"
PROJECT_NAME = "ruralstroke-stage3"
STAGE4_VOLUMES = ("ruralstroke-stage4-collector-sqlite", "ruralstroke-stage4-collector-media")


def _urlsafe_json(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _demo_token(secret: str) -> str:
    now = int(time.time())
    header = _urlsafe_json({"alg": "HS256", "typ": "JWT"})
    payload = _urlsafe_json({
        "sub": "local-demo",
        "roles": ["collector", "clinician"],
        "facilities": ["*"],
        "iss": "ruralstroke-local",
        "aud": "ruralstroke-api",
        "iat": now,
        "exp": now + 3600,
    })
    unsigned = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), unsigned, hashlib.sha256).digest()
    return f"{header}.{payload}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"


def _write_secret(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        stream.write(value)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def ensure_demo_secrets(secret_dir: str | Path = SECRET_DIR) -> dict[str, Path]:
    directory = Path(secret_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    postgres_password_path = directory / "postgres_password"
    jwt_secret_path = directory / "jwt_secret"
    minio_access_path = directory / "minio_access_key"
    minio_secret_path = directory / "minio_secret_key"
    _write_secret(postgres_password_path, random_secrets.token_urlsafe(24))
    _write_secret(jwt_secret_path, random_secrets.token_urlsafe(32))
    _write_secret(minio_access_path, "ruralstroke-minio")
    # mc treats a credential beginning with '-' as a command-line flag.
    _write_secret(minio_secret_path, random_secrets.token_hex(32))
    password = postgres_password_path.read_text(encoding="utf-8").strip()
    jwt_secret = jwt_secret_path.read_text(encoding="utf-8").strip()
    database_url_path = directory / "database_url"
    _write_secret(database_url_path, f"postgresql+psycopg://ruralstroke:{password}@postgres:5432/ruralstroke")
    token_path = directory / "demo_token"
    _write_secret(token_path, _demo_token(jwt_secret))
    return {
        "postgres_password": postgres_password_path,
        "database_url": database_url_path,
        "jwt_secret": jwt_secret_path,
        "minio_access_key": minio_access_path,
        "minio_secret_key": minio_secret_path,
        "demo_token": token_path,
    }


def _compose_environment(secret_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["RURALSTROKE_DEMO_SECRET_DIR"] = str(secret_dir.resolve())
    return environment


def run_compose(arguments: list[str], *, secret_dir: Path = SECRET_DIR, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose", "--project-name", PROJECT_NAME, "--file", str(COMPOSE_FILE), *arguments]
    return subprocess.run(command, cwd=ROOT, env=_compose_environment(secret_dir), text=True, check=check)


def _image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def wait_for_url(url: str, *, timeout_seconds: float = 300.0) -> float:
    started = time.monotonic()
    deadline = started + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=3) as response:
                if 200 <= response.status < 400:
                    return time.monotonic() - started
        except (OSError, URLError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def smoke_check() -> dict[str, float]:
    checks = {
        "api_ready_seconds": "http://127.0.0.1:8000/health/ready",
        "api_docs_seconds": "http://127.0.0.1:8000/docs",
        "collector_seconds": "http://127.0.0.1:8501/_stcore/health",
        "collector_edge_seconds": "http://127.0.0.1:8503/_stcore/health",
        "clinician_seconds": "http://127.0.0.1:8502/_stcore/health",
    }
    timings = {name: wait_for_url(url) for name, url in checks.items()}
    print("Local stack is ready:")
    print("  collector: http://127.0.0.1:8501")
    print("  clinician: http://127.0.0.1:8502")
    print("  api:       http://127.0.0.1:8000")
    print("  api docs:  http://127.0.0.1:8000/docs")
    return timings


def up() -> int:
    ensure_demo_secrets()
    compose_up = ["up", "--detach"]
    if not (_image_exists("ruralstroke-api:stage3-local") and _image_exists("ruralstroke-ui:stage3-local") and _image_exists("ruralstroke-collector-edge:stage4-local")):
        compose_up.insert(1, "--build")
    run_compose(compose_up)
    timings = smoke_check()
    print(json.dumps({"readiness_seconds": timings}, indent=2, sort_keys=True))
    return 0


def status() -> int:
    run_compose(["ps"], check=False)
    return 0


def logs(services: list[str]) -> int:
    run_compose(["logs", "--tail", "100", *services], check=False)
    return 0


def down() -> int:
    run_compose(["down", "--remove-orphans"], check=False)
    return 0


def reset() -> int:
    run_compose(["down", "--volumes", "--remove-orphans"], check=False)
    target = SECRET_DIR.resolve()
    if target == (ROOT / ".runtime-secrets").resolve() and target.is_dir():
        shutil.rmtree(target)
    print("Removed Stage 3 demo containers, named volumes, and local demo secrets.")
    return 0


def reset_collector() -> int:
    run_compose(["down", "--remove-orphans"], check=False)
    for volume in STAGE4_VOLUMES:
        subprocess.run(["docker", "volume", "rm", volume], cwd=ROOT, check=False, text=True)
    print("Removed only Stage 4 collector named volumes; central Stage 3 data was preserved.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("up")
    subparsers.add_parser("status")
    logs_parser = subparsers.add_parser("logs")
    logs_parser.add_argument("services", nargs="*")
    subparsers.add_parser("down")
    subparsers.add_parser("reset")
    subparsers.add_parser("reset-collector")
    args = parser.parse_args(argv)
    if args.command == "up":
        return up()
    return {"status": status, "down": down, "reset": reset, "reset-collector": reset_collector}.get(args.command, lambda: logs(args.services))()


if __name__ == "__main__":
    sys.exit(main())
