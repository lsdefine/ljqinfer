"""Prefill wall time, to check the gather did not cost what the scatter saved."""
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, '/mnt/data/kw/ljqinfer_dsv41f_tp8')

from strategy.decode_worker import bootstrap        # noqa: E402
from tests.batch_probe import _prefill              # noqa: E402
from tests.tie_probe import served_row              # noqa: E402

LENGTHS = (11, 512, 1024)
WARMUP, TIMED = 3, 12


def once(eng, toks):
    past = eng.past
    slot = past.alloc()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _prefill(eng, slot, toks)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0
    past.release(slot)
    return dt


def main():
    base, _ = served_row()
    with torch.inference_mode():
        eng = bootstrap()
        say = dist.get_rank() == 0
        for n in LENGTHS:
            toks = (list(base) * (n // len(base) + 1))[:n]
            for _ in range(WARMUP):
                once(eng, toks)
            ms = sorted(once(eng, toks) for _ in range(TIMED))
            if say:
                print('BP len=%d median=%.2f ms min=%.2f max=%.2f'
                      % (n, statistics.median(ms), ms[0], ms[-1]), flush=True)


if __name__ == '__main__':
    main()
