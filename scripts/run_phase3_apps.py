"""Launch the local collector and clinician Streamlit applications."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

from rural_stroke_assist.cases.factory import create_default_workflow_service


ROOT = Path(__file__).resolve().parents[1]


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.15)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    create_default_workflow_service()
    processes: list[subprocess.Popen] = []
    commands = [("collector", 8501, ROOT / "apps" / "collector_app.py"), ("clinician", 8502, ROOT / "apps" / "clinician_app.py")]
    try:
        for name, port, app in commands:
            if port_open(port):
                print(f"{name.title()} already running: http://localhost:{port}")
                continue
            process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(port), "--server.headless", "true"], cwd=ROOT)
            processes.append(process)
            print(f"{name.title()} started: http://localhost:{port}")
        print("Press Ctrl+C to stop applications launched by this command.")
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("Stopping launched applications...")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
