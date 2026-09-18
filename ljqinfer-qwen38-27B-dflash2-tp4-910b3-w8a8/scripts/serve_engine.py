#!/usr/bin/env python3
"""Launch the persistent four-rank strategy engine on NPU 4-7."""
from __future__ import annotations
import argparse, os, shutil, signal, subprocess, sys, time, uuid
from model.config import CONFIG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def child(rank: int):
    os.environ["LJQ_SPMD_RANK"] = str(rank)
    if rank == 0:
        from server.engine_server import main
        main([])
    else:
        from strategy.strategy import follower_main
        follower_main()


def parent():
    run_id = uuid.uuid4().hex[:10]
    root_file = f"/dev/shm/qwen_tp4_server_{run_id}.bin"
    control_dir = f"/tmp/qwen_tp4_control_{run_id}"
    os.makedirs(control_dir, mode=0o700, exist_ok=False)
    procs = []
    for rank in range(CONFIG.tp):
        env = dict(os.environ)
        env["LJQ_HCCL_ROOT"] = root_file
        env["LJQ_CONTROL_DIR"] = control_dir
        env["LJQ_SPMD_RANK"] = str(rank)
        cmd = [sys.executable, os.path.abspath(__file__), "--rank", str(rank)]
        log = open(f"/tmp/qwen_tp4_serve_rank{rank}.log", "w")
        procs.append((subprocess.Popen(cmd, cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT), log))
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    rc = 0
    try:
        while True:
            exited = [(proc, proc.poll()) for proc, _ in procs
                      if proc.poll() is not None]
            if exited:
                rc = next((code for _, code in exited if code), 0)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        rc = 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        for proc, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 10.0
        for proc, _ in procs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    proc.kill()
        for proc, log in procs:
            proc.wait(); log.close()
        shutil.rmtree(control_dir, ignore_errors=True)
        try:
            os.unlink(root_file)
        except FileNotFoundError:
            pass
    return rc


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--rank", type=int, default=-1)
    args = parser.parse_args()
    if args.rank >= 0:
        child(args.rank)
    else:
        raise SystemExit(parent())


if __name__ == "__main__":
    main()
