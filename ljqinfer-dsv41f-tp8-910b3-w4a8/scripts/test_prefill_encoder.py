"""TP8 encoder refactor acceptance; run from repo with PYTHONPATH=repo.
Loads weights once and checks the frozen Git baseline against the candidate.
PREFILL_ACCEPT_SERVE=1 continues serving after acceptance; default exits.
Run only with the existing TP8 backend stopped; requires eight free devices.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import time
import torch
import torch.distributed as dist
from strategy import decode_worker as worker

OUT = Path(os.environ.get('PREFILL_ACCEPT_OUT', '/tmp/prefill_mod_accept'))
BASELINE = 'aef455fc743983ce32fc189ed5b16f1b9239182c'


def load_reference():
    modules = {}
    for name in ('npu_routed', 'prefill_layer', 'prefill_block', 'prefill_attention'):
        path = ('ops/prefill/' if name == 'npu_routed' else 'model/') + name + '.py'
        source = subprocess.check_output(['git', 'show', BASELINE + ':' + path], text=True)
        spec = importlib.util.spec_from_loader('ref_' + name, loader=None)
        mod = importlib.util.module_from_spec(spec)
        exec(compile(source, BASELINE + ':' + path, 'exec'), mod.__dict__)
        modules[name] = mod
    return modules


def variants(model, refs):
    old = []
    for block in model.blocks:
        b = copy.copy(block)
        b.__class__ = refs['prefill_block'].PrefillBlock
        b.attention = copy.copy(block.attention)
        b.attention.__class__ = refs['prefill_layer'].PrefillAttention
        b.ffn = copy.copy(block.ffn)
        b.ffn.routed = copy.copy(block.ffn.routed)
        b.ffn.routed.__class__ = refs['npu_routed'].PackedRouted
        if block.engram is not None:
            b.engram = copy.copy(block.engram)
            b.engram.__class__ = refs['prefill_block'].PrefillEngram
        old.append(b)
    return old


class RecordBlock:
    def __init__(self, block, records):
        self.block, self.records = block, records

    def __getattr__(self, name):
        return getattr(self.block, name)

    def __call__(self, *args):
        h, pre = self.block(*args)
        # Small parity cases: retain every element on host, no extra NPU slab.
        self.records.append((self.layer, h.cpu().clone(), pre.cpu().clone()))
        return h, pre


def cpu(t):
    return t.detach().cpu().clone()


def snapshot(past, slot, end):
    result = {}
    for layer, source in past.sources.items():
        if layer > 20:
            continue
        count = end // source.ratio
        result[f'{layer}.kv'] = cpu(source.ckv_pool.read(slot, 0, count))
        result[f'{layer}.index'] = cpu(source.index_pool.read(slot, 0, count))
        if source.ratio == 2:
            result[f'{layer}.carry_kv'] = cpu(source.kv_state[slot])
            result[f'{layer}.carry_score'] = cpu(source.score_state[slot])
    for layer, window in past.windows.items():
        if layer < 20:
            result[f'{layer}.window'] = cpu(window.main_kv[slot])
    for name, value in past.prefill_tails[slot].items():
        if isinstance(value, torch.Tensor):
            result['tail.' + name] = cpu(value)
    return result


def run_case(eng, blocks, state_class, chunks, record):
    import model.prefill as pref
    model, past = eng.compute, eng.past
    original = (model.blocks, model.encoder, model.decoder, pref.AttentionState)
    records = []
    model.blocks = blocks
    model.encoder = [RecordBlock(b, records) for b in blocks[:20]] if record else blocks[:20]
    model.decoder = blocks[20:]
    pref.AttentionState = state_class
    slot = past.alloc()
    tokens = [100 + ((i * 7919 + 17) % 90000) for i in range(sum(chunks))]
    start = 0
    times = []
    states = []
    try:
        for length in chunks:
            end = start + length
            past.ensure(slot, end)
            torch.npu.synchronize()
            dist.barrier(group=eng.ctrl)
            before = torch.npu.memory_stats()
            ev0 = torch.npu.Event(enable_timing=True)
            ev1 = torch.npu.Event(enable_timing=True)
            t = time.perf_counter()
            ev0.record()
            model.forward(tokens[start:end], past=past, slot=slot, start=start,
                          history_tokens=tokens[max(0, start-3):start])
            ev1.record()
            torch.npu.synchronize()
            elapsed = time.perf_counter() - t
            maximum = torch.tensor([elapsed], dtype=torch.float64)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=eng.ctrl)
            after = torch.npu.memory_stats()
            times.append({'wall_max': maximum.item(), 'wall': elapsed,
                          'event': ev0.elapsed_time(ev1)/1000,
                          'retries': after.get('num_alloc_retries', 0)-before.get('num_alloc_retries', 0),
                          'allocated': after.get('allocated_bytes.all.current', 0),
                          'reserved': after.get('reserved_bytes.all.current', 0)})
            past.set_pos(slot, end)
            if record:
                states.append(snapshot(past, slot, end))
            start = end
        return records, states, times
    finally:
        past.release(slot)
        model.blocks, model.encoder, model.decoder, pref.AttentionState = original


def compare(reference, candidate):
    failures = []
    exact = total = 0
    for i, (a, b) in enumerate(zip(reference[0], candidate[0], strict=True)):
        assert a[0] == b[0]
        for name, x, y in [('h', a[1], b[1]), ('pre', a[2], b[2])]:
            total += 1
            if torch.equal(x, y):
                exact += 1
            else:
                d = (x.float() - y.float()).abs()
                failures.append({'where': f'block[{i}].{name}', 'max_abs': d.max().item(),
                                 'unequal': int((x != y).sum())})
    for i, (a, b) in enumerate(zip(reference[1], candidate[1], strict=True)):
        assert a.keys() == b.keys()
        for name, x in a.items():
            total += 1
            y = b[name]
            if torch.equal(x, y):
                exact += 1
            else:
                failures.append({'where': f'state[{i}].{name}',
                                 'unequal': int((x != y).sum())})
    return {'exact': exact, 'total': total, 'failures': failures}


@torch.inference_mode()
def acceptance(eng):
    from model.prefill_attention import AttentionState
    model = eng.compute
    current = model.blocks
    refs = load_reference()
    baseline = variants(model, refs)
    report = {'rank': eng.rank, 'cases': [], 'timing': []}
    for chunks in ([1], [127, 2, 129], [513]):
        ref = run_case(eng, baseline, refs['prefill_attention'].AttentionState, chunks, True)
        new = run_case(eng, current, AttentionState, chunks, True)
        item = {'chunks': chunks, **compare(ref, new)}
        report['cases'].append(item)
        (OUT / f'rank{eng.rank}.json').write_text(json.dumps(report, indent=2))
        print('MOD_PARITY', eng.rank, chunks, item['exact'], item['total'],
              item['failures'][:3], flush=True)
    failed = torch.tensor([int(any(c['failures'] for c in report['cases']))])
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=eng.ctrl)
    if failed.item():
        raise AssertionError('encoder/state parity failed; see per-rank reports')
    # Warm both paths; alternate order to avoid always favoring the later path.
    for i in range(34):
        order = [('baseline', baseline, refs['prefill_attention'].AttentionState),
                 ('candidate', current, AttentionState)]
        if i % 2:
            order.reverse()
        for label, blocks, state in order:
            result = run_case(eng, blocks, state, [8192], False)
            report['timing'].append({'iteration': i, 'variant': label,
                                    'measure': result[2][0], 'warmup': i < 4, 'tokens': 8192})
            print('MOD_TIMING', eng.rank, report['timing'][-1], flush=True)
    report['status'] = 'PASS'
    (OUT / f'rank{eng.rank}.json').write_text(json.dumps(report, indent=2))
    dist.monitored_barrier(group=eng.ctrl)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eng = worker.bootstrap()
    with worker.worker_scope([eng, eng.ctrl], 'MOD_ACCEPT', close=True):
        with torch.inference_mode():
            acceptance(eng)
            if os.environ.get('PREFILL_ACCEPT_SERVE') != '1':
                return
            jobs = queue.Queue()
            if eng.rank == 0:
                worker.serve_rank0(eng, jobs)
                print('ENGINE_READY MOD_ACCEPT_PASS', flush=True)
                worker.drive(eng, jobs, worker.MAX_BATCH)
                eng.command(worker.OP_STOP)
            else:
                while eng.command() is not False:
                    pass


if __name__ == '__main__':
    main()
