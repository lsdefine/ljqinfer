"""Opt-in prefill spans; read events only after the caller's existing sync.
Create <PREFILL_TRACE_DIR>/enabled to trace subsequent requests, remove to stop.
No per-span synchronization, tensor retention, or arithmetic changes.
"""
import functools
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import torch

_active = None
_serial = itertools.count()


def memory():
    stats = torch.npu.memory_stats()
    return {k: stats.get(k, 0) for k in (
        'num_alloc_retries', 'num_ooms', 'allocated_bytes.all.current',
        'reserved_bytes.all.current', 'inactive_split_bytes.all.current')}


def begin(rank, tokens):
    global _active
    _active = None
    root = os.environ.get('PREFILL_TRACE_DIR')
    if root and (Path(root) / 'enabled').exists():
        _active = dict(root=root, rank=rank, seq=next(_serial), tokens=len(tokens),
                       prompt=hashlib.sha256(bytes(str(tokens), 'utf-8')).hexdigest()[:16],
                       spans=[], chunk=None)


def span(kind):
    def decorate(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            active = _active
            if active is None:
                return fn(*args, **kwargs)
            meta = dict(kind=kind)
            previous = active['chunk']
            if kind == 'chunk':
                start = kwargs['start']
                meta.update(start=start, end=start+len(args[1]), phase=kwargs['phase'])
                active['chunk'] = start
            else:
                meta.update(start=active['chunk'], rows=len(args[0]), keys=len(args[2]),
                            heads=args[0].shape[1], ratio=kwargs['ratio'])
            before = memory()
            a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            a.record()
            wall, cpu = time.perf_counter(), time.thread_time()
            try:
                return fn(*args, **kwargs)
            finally:
                meta.update(host_seconds=time.perf_counter()-wall,
                            thread_cpu_seconds=time.thread_time()-cpu)
                b.record()
                meta.update(memory_before=before, memory_after=memory())
                active['spans'].append((meta, a, b))
                active['chunk'] = previous
        return run
    return decorate


def finish():
    global _active
    active, _active = _active, None
    if active is None:
        return
    records = []
    for meta, a, b in active.pop('spans'):
        meta['stream_seconds'] = a.elapsed_time(b)/1000
        records.append(meta)
    root = Path(active.pop('root'))
    active.pop('chunk')
    active['spans'] = records
    (root / ('rank%d_request%03d.json' % (active['rank'], active['seq']))).write_text(
        json.dumps(active, indent=2))
