"""Resident TP8 service: commit-point boarding and epoch page leases."""
import os
import json
import threading
import time
import uuid
from queue import Queue, Empty
from collections import deque
from model.glm53_resident import ResidentBatch
import torch
import torch.distributed as dist
from model.glm53_generate import Generator
from model.config import DEFAULT_PREFILL_CHUNK_TOKENS

MAX_TOTAL_TOKENS = 144 * 1024
CAPACITY = ((MAX_TOTAL_TOKENS + 8 + 63) // 64) * 64
_jobs = Queue()
_generator = None
_resident = None
_worker = None
_control = None
MAX_BATCH = 4
BATCH_WAIT_SECONDS = 0.02


class Handle:
    def __init__(self):
        self.request_id = uuid.uuid4().hex
        self.event = threading.Event()
        self.state = 'pending'
        self.stop_reason = None

    def cancel(self):
        changed = not self.event.is_set()
        if self.stop_reason is None:self.stop_reason = 'cancelled'
        self.event.set()
        return changed

    def stop_at_semantic_eos(self):
        changed = not self.event.is_set()
        self.stop_reason = 'semantic_eos'
        self.event.set()
        return changed


def startup(devices=None, prefill_chunk_tokens=None):
    global _generator, _resident, _worker, _control
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group('nccl')
    from datetime import timedelta
    _control = dist.new_group(backend='gloo', timeout=timedelta(seconds=90))
    _generator = Generator.load(capacity=CAPACITY,
        prefill_chunk_tokens=(DEFAULT_PREFILL_CHUNK_TOKENS if prefill_chunk_tokens is None else prefill_chunk_tokens))
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8/tokenizer.json')
    ids = tokenizer.encode('The capital of France is').ids
    _generator.generate(ids, max_new_tokens=8, eos_token_ids=[])
    _generator.reset()
    _resident = ResidentBatch(_generator, MAX_BATCH)
    _resident.warm()
    # Do not advertise readiness after only a tiny prompt and graph capture.
    from strategy.glm53_capacity import check_capacity
    check_capacity(_resident, MAX_TOTAL_TOKENS, ids)
    dist.barrier(device_ids=[rank])
    print(f'[glm53] rank={rank} warmed Q8+DFlash capacity={CAPACITY}', flush=True)
    if rank == 0:
        _worker = threading.Thread(target=worker_loop, name='glm53-tp8', daemon=True)
        _worker.start()


def query(input_ids, max_new_tokens):
    ids = list(input_ids)
    n = int(max_new_tokens)
    if not ids or n < 1 or len(ids) + n > MAX_TOTAL_TOKENS:
        raise ValueError(f'nonempty prompt, output>=1 and prompt+output<={MAX_TOTAL_TOKENS} required')
    if any(type(i) is not int or not 0 <= i < 154880 for i in ids):
        raise ValueError('invalid token id')
    out = Queue()
    out.cancel_handle = handle = Handle()
    out.request_id = handle.request_id
    _jobs.put((ids, n, out, handle))
    return out


def _run_batch(payload, jobs, rank, deferred):
    # CPU control group avoids a device allocation, NCCL launch and D2H fence
    # every round. Last word advertises pending admissions to all ranks.
    flags = torch.zeros(len(payload) + 1, dtype=torch.int32)
    pending = False
    batch_at_prefill = {}
    timing = {}

    def cancelled():
        nonlocal flags, pending
        started = time.perf_counter()
        if flags.numel() != len(payload) + 1:
            flags = torch.zeros(len(payload) + 1, dtype=torch.int32)
        if rank == 0:
            for i, (_, _, _, h) in enumerate(jobs):
                flags[i] = 2 if h.stop_reason == 'semantic_eos' else int(h.event.is_set())
            flags[-1] = bool(deferred) or not _jobs.empty()
        dist.broadcast(flags, src=0, group=_control)
        pending = bool(flags[-1])
        result = flags[:-1].tolist()
        elapsed = time.perf_counter() - started
        for t in timing.values():
            t["control"] += elapsed
        return result

    def prefilled(i, metrics):
        batch_at_prefill[i] = metrics['batch_size']
        now = time.perf_counter()
        timing[i] = dict(start=now, model=0.0, control=0.0, stall=0.0)
        # Existing lanes waited while this request was being prefetched.
        for rid, t in timing.items():
            if rid != i:t['stall'] += metrics['strategy_seconds']
        if rank == 0:
            jobs[i][2].put({'type': 'prefill', 'metrics': {
                'model_prefill_seconds': metrics['seconds'],
                'input_tokens': len(payload[i][0]),
                'cache_hit_tokens': metrics['cache_hit_tokens'],
                'cache_source': metrics['cache_source'],
                'cache_load_seconds': metrics['cache_load_seconds'],
                'cache_store_seconds': metrics['cache_store_seconds'],
                'cache_store_enqueue_seconds': metrics['cache_store_enqueue_seconds'],
                'cache_store_drain_seconds': metrics['cache_store_drain_seconds'],
                'strategy_seconds': metrics['strategy_seconds'],
                'prefill_tokens': metrics['prefill_tokens'],
                'batch_size': metrics['batch_size']}})

    def tokens(i, items):
        if rank == 0:
            jobs[i][2].put({'type': 'token', 'token_ids': items})

    def round_done(rec):
        for i in rec["active"]:
            timing[i]["model"] += rec["round_ms"] / 1000

    def done(i, result):
        t = timing.pop(i, None)
        rounds = len(result.steps)
        scale = 1000 / rounds if rounds else 0.0
        stats = dict(
            wall_ms_per_step=(time.perf_counter()-t["start"])*scale if t else 0.0,
            control_ms_per_step=t["control"]*scale if t else 0.0,
            model_step_host_ms=t["model"]*scale if t else 0.0,
            prefill_stall_seconds=t["stall"] if t else 0.0)
        if rank == 0:
            print('[strategy] ' + json.dumps(dict(input_tokens=len(payload[i][0]),
                output_tokens=len(result.token_ids), **stats), sort_keys=True), flush=True)
            jobs[i][3].state = 'done'
            jobs[i][2].put({'type': 'end', 'finish_reason': result.finish_reason,
                **stats, 'decode_steps': len(result.steps),
                'accepted_tokens': sum(s['accepted_drafts'] for s in result.steps),
                'proposed_tokens': 7 * len(result.steps),
                'batch_size': batch_at_prefill.get(i, 0),
                'cancelled': result.finish_reason == 'cancelled'})

    def board(remaining, capacity):
        nonlocal pending
        if not pending:return None
        # All ranks enter only at a committed decode boundary. Rank 0 alone
        # owns the queue; admission and cancellation decisions are broadcast.
        started = time.perf_counter()
        command = [None]
        if rank == 0:
            while True:
                try:
                    job = deferred.popleft() if deferred else _jobs.get_nowait()
                except Empty:
                    break
                if job is None:
                    deferred.appendleft(job)
                    break
                ids, n, out, handle = job
                if handle.event.is_set():
                    handle.state = 'done'
                    out.put({'type': 'end', 'cancelled': True, 'finish_reason': 'cancelled'})
                    continue
                needed = ((len(ids) + n + 8 + 63) // 64) * 64
                if needed > remaining:
                    deferred.appendleft(job)
                    break
                jobs.append(job)
                handle.state = 'running'
                command[0] = (ids, n)
                break
        dist.broadcast_object_list(command, src=0, group=_control)
        elapsed = time.perf_counter() - started
        for t in timing.values():
            t["control"] += elapsed
        if command[0] is not None:
            payload.append(command[0])
        else:
            pending = False
        return command[0]

    def boarded(row, active):
        if rank == 0:
            print(f'[glm53] boarded request={jobs[row][3].request_id} row={row} active={active}', flush=True)

    before = _resident.capture_count
    _resident.generate([ids for ids, _ in payload], [n for _, n in payload],
        on_tokens=tokens, on_prefill=prefilled, should_stop=cancelled, on_done=done,
        board_request=board, on_boarded=boarded, on_round=round_done)
    if rank == 0:
        print(f'[glm53] epoch requests={len(payload)} new_verify_graphs={_resident.capture_count-before}', flush=True)


def worker_loop():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    flag = torch.zeros(1, device=rank, dtype=torch.int32)
    deferred = deque()
    try:
        while True:
            out = handle = None
            jobs = []
            command = [None]
            if rank == 0:
                try:
                    job = deferred.popleft() if deferred else _jobs.get(timeout=30)
                except Empty:
                    job = None
                    command[0] = "idle"
                if job is not None:
                    jobs = [job]
                    pages = ((len(job[0]) + job[1] + 8 + 63) // 64)
                    deadline = time.perf_counter() + BATCH_WAIT_SECONDS
                    while len(jobs) < MAX_BATCH:
                        try:
                            nxt = _jobs.get(timeout=max(0, deadline - time.perf_counter()))
                        except Empty:
                            break
                        if nxt is None:
                            deferred.append(nxt)
                            break
                        needed = ((len(nxt[0]) + nxt[1] + 8 + 63) // 64)
                        if pages + needed > _generator.engine.capacity // 64:
                            deferred.append(nxt)
                            break
                        jobs.append(nxt)
                        pages += needed
                    command[0] = [(ids, n) for ids, n, _, _ in jobs]
                    for _, _, _, h in jobs:
                        h.state = 'running'
                    ids, n, out, handle = jobs[0]
            dist.broadcast_object_list(command, src=0, group=_control)
            if command[0] is None:
                break
            if command[0] == "idle":
                continue
            _run_batch(command[0], jobs, rank, deferred)
    except Exception:
        import traceback
        traceback.print_exc()
        # A failed rank cannot safely continue collectives; torchrun tears down peers.
        os._exit(1)
    finally:
        if _resident is not None:
            _resident.close()
        _generator.close()
        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()


def shutdown():
    if _worker is not None:
        _jobs.put(None)
        _worker.join()
