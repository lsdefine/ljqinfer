#!/usr/bin/env python3
"""Launch the persistent four-rank strategy engine on physical CUDA GPU4-7."""
from __future__ import annotations
import argparse, os, shutil, signal, subprocess, sys, time, uuid
# Device visibility is fixed before any CUDA runtime import.
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
from model.config import CONFIG


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
        env["LJQ_NCCL_ROOT"] = root_file
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
    rank_failed = False
    try:
        while True:
            exited = [(proc, proc.poll()) for proc, _ in procs
                      if proc.poll() is not None]
            if exited:
                rank_failed = True
                rc = next((code for _, code in exited if code), 1)
                print("[supervisor] rank exited unexpectedly; aborting TP group", flush=True)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        # Rank0 owns admission/drain and broadcasts collective model release.
        rc = 0
    finally:
        signal.signal(signal.SIGTERM, previous)
        leader = procs[0][0]
        if leader.poll() is None:
            leader.terminate()
        # A missing rank cannot participate in collective graceful shutdown.
        if rank_failed:
            for proc, _ in procs:
                if proc.poll() is None:
                    proc.terminate()
        deadline = time.monotonic() + (5.0 if rank_failed else 90.0)
        forced = False
        while any(proc.poll() is None for proc, _ in procs):
            if time.monotonic() >= deadline:
                forced = True
                break
            time.sleep(0.1)
        if forced:
            print("[supervisor] graceful shutdown timed out; forced cleanup", flush=True)
            for proc, _ in procs:
                if proc.poll() is None:
                    proc.terminate()
            deadline = time.monotonic() + 5.0
            for proc, _ in procs:
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        proc.kill()
        codes = []
        for rank, (proc, log) in enumerate(procs):
            code = proc.wait()
            codes.append(code)
            log.close()
            print(f"[supervisor] rank={rank} exit_code={code}", flush=True)
        if forced:
            rc = 1
        elif rc == 0:
            rc = next((code for code in codes if code), 0)
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
