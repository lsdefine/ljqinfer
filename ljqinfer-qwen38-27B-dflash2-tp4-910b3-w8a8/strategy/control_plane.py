"""Local Unix-socket control plane for persistent SPMD strategy ranks.

HCCL is reserved for model collectives. Request commands travel over AF_UNIX so
followers never sit inside a model collective while rank0 waits for HTTP work.
"""
from __future__ import annotations
import inspect
import json
import os
import socket
import threading
import time
from typing import Callable


def _send(fp, value) -> None:
    payload = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    view = memoryview(payload)
    while view:
        written = fp.write(view)
        if written is None or written <= 0:
            raise ConnectionError("strategy control short write")
        view = view[written:]
    fp.flush()


def _recv(fp):
    line = fp.readline()
    if not line:
        raise ConnectionError("strategy control peer closed")
    return json.loads(line)


class Rank0Coordinator:
    def __init__(self, directory: str, world: int, timeout: float = 60.0):
        self.directory = directory
        self.world = int(world)
        self.timeout = float(timeout)
        self._peers = {}
        # Defense in depth: shared AF_UNIX file objects must never interleave
        # two generate commands even if an upper layer regresses.
        self._io_lock = threading.Lock()

    def _connect(self, rank: int):
        deadline = time.monotonic() + self.timeout
        path = os.path.join(self.directory, f"rank{rank}.sock")
        last = None
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect(path)
                fp = sock.makefile("rwb", buffering=0)
                self._peers[rank] = (sock, fp)
                return sock, fp
            except OSError as exc:
                last = exc
                try: sock.close()
                except Exception: pass
                time.sleep(0.05)
        raise TimeoutError(f"rank{rank} control socket unavailable: {last}")

    def _peer(self, rank: int):
        return self._peers.get(rank) or self._connect(rank)

    def close(self) -> None:
        peers, self._peers = self._peers, {}
        for sock, fp in peers.values():
            try:
                fp.close()
            finally:
                sock.close()

    def run(self, input_ids, max_new_tokens: int, local: Callable[[], dict],
            *, op: str = "generate") -> dict:
        with self._io_lock:
            return self._run_locked(input_ids, max_new_tokens, local, op=op)

    def _run_locked(self, input_ids, max_new_tokens: int,
                    local: Callable[[], dict], *, op: str = "generate") -> dict:
        if op not in ("generate", "generate_dynamic"):
            raise ValueError(f"unsupported control-plane op: {op}")
        command = {"op": op, "input_ids": [int(x) for x in input_ids],
                   "max_new_tokens": int(max_new_tokens)}
        try:
            peers = [(rank, self._peer(rank)) for rank in range(1, self.world)]
            for _, (_, fp) in peers:
                _send(fp, command)
            for rank, (_, fp) in peers:
                reply = _recv(fp)
                if reply.get("state") != "ready":
                    raise RuntimeError(f"rank{rank} failed before GO: {reply}")
            for _, (_, fp) in peers:
                _send(fp, {"op": "go"})
            local_result = local()
            for rank, (_, fp) in peers:
                reply = _recv(fp)
                if reply.get("state") != "done":
                    raise RuntimeError(f"rank{rank} execution failed: {reply}")
            return local_result
        except Exception:
            # A failed request can leave unread DONE/error frames. Drop every
            # connection so the next request starts a fresh protocol session.
            self.close()
            raise


class FollowerControlServer:
    def __init__(self, directory: str, rank: int):
        self.directory = directory
        self.rank = int(rank)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        self.path = os.path.join(directory, f"rank{rank}.sock")
        try: os.unlink(self.path)
        except FileNotFoundError: pass

    def serve(self, execute: Callable[..., None]) -> None:
        signature = inspect.signature(execute)
        try:
            signature.bind((), 0, op="generate")
            supports_op = True
        except TypeError:
            supports_op = False
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.path)
        os.chmod(self.path, 0o600)
        server.listen(1)
        while True:
            conn, _ = server.accept()
            fp = conn.makefile("rwb", buffering=0)
            try:
                while True:
                    command = _recv(fp)
                    if command.get("op") == "shutdown":
                        _send(fp, {"state": "done"})
                        return
                    op = command.get("op")
                    if op not in ("generate", "generate_dynamic"):
                        _send(fp, {"state": "error", "error": "bad command"})
                        continue
                    ids = tuple(int(x) for x in command["input_ids"])
                    n_new = int(command["max_new_tokens"])
                    _send(fp, {"state": "ready", "rank": self.rank})
                    go = _recv(fp)
                    if go.get("op") != "go":
                        raise RuntimeError(f"expected GO, got {go}")
                    try:
                        if supports_op:
                            execute(ids, n_new, op=op)
                        elif op == "generate":
                            execute(ids, n_new)
                        else:
                            raise TypeError(
                                "follower execute callback does not support "
                                "generate_dynamic")
                        _send(fp, {"state": "done", "rank": self.rank})
                    except Exception as exc:
                        _send(fp, {"state": "error", "rank": self.rank,
                                   "error": f"{type(exc).__name__}: {exc}"})
                        raise
            except (ConnectionError, BrokenPipeError):
                pass
            finally:
                try: fp.close()
                finally: conn.close()
