#!/usr/bin/env python3
"""Single supervisor for the TP4 Q8+DFlash2 engine and OpenAI facade."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENGINE_HEALTH = "http://127.0.0.1:62001/health"
STARTUP_TIMEOUT = 240.0


def wait_engine(proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"engine exited during startup rc={proc.returncode}")
        try:
            with urllib.request.urlopen(ENGINE_HEALTH, timeout=2) as response:
                body = json.load(response)
            if body.get("status") == "ok":
                return
            last_error = RuntimeError(f"unexpected engine health: {body}")
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"engine health timeout: {last_error}")


def terminate(proc: subprocess.Popen, timeout: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    env = dict(os.environ)
    logs = ROOT / ".runtime"
    logs.mkdir(exist_ok=True)
    engine_log = open(logs / "serve_engine.log", "a", buffering=1)
    api_log = open(logs / "serve_api.log", "a", buffering=1)
    engine = api = None

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        # Pin the CUDA interpreter and physical GPU4-7 placement.
        engine_command = (
            f"source {ROOT / 'scripts' / 'env_tp4.sh'} && "
            f"exec $PYTHON_BIN {ROOT / 'scripts' / 'serve_engine.py'}")
        engine = subprocess.Popen(
            ["/bin/bash", "-lc", engine_command], cwd=ROOT, env=env,
            stdout=engine_log, stderr=subprocess.STDOUT)
        wait_engine(engine)
        api = subprocess.Popen(
            [sys.executable, "-m", "server.server"], cwd=ROOT, env=env,
            stdout=api_log, stderr=subprocess.STDOUT)
        while True:
            if engine.poll() is not None:
                raise RuntimeError(f"engine exited rc={engine.returncode}")
            if api.poll() is not None:
                raise RuntimeError(f"OpenAI facade exited rc={api.returncode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        if api is not None:
            terminate(api)
        if engine is not None:
            terminate(engine, timeout=110.0)
        engine_log.close()
        api_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
