# -*- coding: utf-8 -*-
"""prefill-vs-decode equivalence probe (torchrun --nproc_per_node=8).
Sequence S = prompt + rep0_out[:K] (K=1799: prefill-path says EOS, decode-path said </think>).
A: full prefill of S (chunked, 128-aligned) -> last-token argmax + KV snapshot.
B: prefill S[:-m] then m Q-decode steps (teacher forced with real tokens) -> argmax + KV.
Compare per-layer KV/ckv/ickv rows at positions [N-m, N) and the final logits.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, os, time, traceback
import torch
import torch.distributed as dist

t0 = time.time()
RANK = int(os.environ['RANK'])
def log(*a):
    if RANK == 0:
        print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

K = int(os.environ.get('K', '1799'))
MS = [int(x) for x in os.environ.get('MS', '8,64').split(',')]
CHUNK = 1024

try:
    dist.init_process_group('nccl')
    torch.cuda.set_device(RANK)
    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    arch.init_distributed()

    prompt = json.load(open('/tmp/g5k/prompt_ids.json'))
    out = json.load(open('/tmp/g5k/rep0_out.json'))
    S = list(prompt) + list(out[:K])
    N = len(S)
    log(f'seq N={N} (prompt {len(prompt)} + out {K})')

    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
    model = Transformer(args)
    bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]
    assert not bad, bad
    dev = f'cuda:{RANK}'
    for m_ in model.modules():
        for k_, v_ in list(m_._buffers.items()):
            if v_ is not None and v_.device.type == 'cpu':
                m_._buffers[k_] = v_.to(dev)
    torch.set_default_device(dev)
    dist.barrier()
    log('model ready')

    
    SPEC_Q = 8
    DRAFT = os.environ.get('DRAFT', 'mtp')     # mtp: real MTP drafts, only row0 teacher-forced | true: all rows true tokens
    GRAPH = int(os.environ.get('GRAPH', '0'))

    def prefill(seq, slot, draft=True):
        pos = 0; lg = None; mh = None
        while pos < len(seq):
            n = min(CHUNK, len(seq) - pos)
            if pos + n < len(seq):
                n = (n // 128) * 128 or n
            toks = torch.tensor([seq[pos:pos + n]], device=dev)
            with torch.no_grad():
                oid, lg, mh = model(toks, start_pos=pos, full_logits=False, slot=slot)
                if draft:
                    model.write_draft_kv(mh, pos, slot)
            pos += n
        lg = (lg[0] if lg.dim() == 3 else lg)[-1].float()
        return lg, mh[:, -1:]

    def snap(slot, t0_, t1_):
        d = {}
        for lid, p in model.pool.layers.items():
            d[f'kv{lid}'] = p.kv(slot, t0_, t1_).float().clone()
            if hasattr(p, 'ckv'): d[f'ckv{lid}'] = p.ckv(slot, t1_).float().clone()
            if hasattr(p, 'ickv'): d[f'ickv{lid}'] = p.ickv(slot, t1_).float().clone()
        return d

    def kvreport(snapA, snapB, P):
        for lid in sorted(model.pool.layers.keys()):
            a, b = snapA[f'kv{lid}'], snapB[f'kv{lid}']
            d = (a - b).abs().amax(dim=tuple(range(1, a.dim())))
            nzp = torch.nonzero(d > 0).flatten()
            s_ = f'L{lid:2d} kv first_diff={int(nzp[0]) if nzp.numel() else -1} relA={float(a.abs().max()):.3g} tail_max={d[P:].max().item():.3g} tail_mean={d[P:].mean().item():.3g} prefix_max={d[:P].max().item():.3g}'
            for nm in ('ckv', 'ickv'):
                k_ = f'{nm}{lid}'
                if k_ in snapA:
                    a, b = snapA[k_], snapB[k_]
                    if a.shape != b.shape: s_ += f' {nm} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}'; continue
                    d2 = (a - b).abs().amax(dim=tuple(range(1, a.dim()))) if a.dim() > 1 else (a - b).abs()
                    nz2 = torch.nonzero(d2 > 0).flatten()
                    s_ += f' {nm} n={a.shape[0]} first_diff={int(nz2[0]) if nz2.numel() else -1} max={d2.max().item():.3g} mean={d2.mean().item():.3g}'
            log('   ' + s_)

    last_lg = {}
    _orig_fqg = model.forward_q_g
    def _hook_fqg(qin, pos_t, slot):
        lg, mh = _orig_fqg(qin, pos_t, slot)
        last_lg['lg'] = lg
        return lg, mh
    model.forward_q_g = _hook_fqg

    # ---- A: full prefill (service prefill path incl. draft kv)
    slotA = model.pool.alloc()
    lgA, _ = prefill(S, slotA)
    topA = int(lgA.argmax())
    log(f'A full-prefill argmax={topA} top5={lgA.topk(5).indices.tolist()}')
    torch.cuda.synchronize(); snapA = snap(slotA, 0, N)
    model.pool.release(slotA)

    for m in MS:
        slot = model.pool.alloc()
        qin = torch.zeros(1, SPEC_Q, dtype=torch.int64, device=dev)
        pos_t = torch.zeros(1, dtype=torch.int64, device=dev)
        temp_t = torch.zeros(1, dtype=torch.float32, device=dev)
        graph = None
        if GRAPH:
            # mirror ModelExecution._capture: capture on fresh slot at pos0=256 before prefill
            pos_t.fill_(256); model.pool.ensure(slot, 256 + 4 * SPEC_Q + 8)
            if dist.get_world_size() > 1:
                from ops import peer_ar; peer_ar.prewarm()
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2): model.step_g(qin, pos_t, slot, temp_t)
            torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize(); dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                g_out, n_out, _ = model.step_g(qin, pos_t, slot, temp_t)
            torch.cuda.synchronize(); dist.barrier()
        P = N - m
        lgP, mh = prefill(S[:P], slot)
        tok = S[P]   # teacher: real next token (service would use _pick(lgP))
        log(f'B m={m} prefill({P}) argmax={int(lgP.argmax())} teacher_tok={tok} agree={int(lgP.argmax())==tok}')
        with torch.no_grad():
            temp_t.fill_(0.0)
            d0, _, _ = model.forward_spec(torch.tensor([tok], device=dev), mh, P - 1, slot=slot)
            n0 = min(d0.size(1), SPEC_Q)
            qin[0, :n0] = d0[0, :n0]; qin[0, n0:] = d0[0, n0 - 1]
            pos_t.fill_(P)
        pos = P; steps = 0; acc_hist = []; pred = {}; final_lg = None
        while pos < N:
            with torch.no_grad():
                model.pool.ensure(slot, pos + SPEC_Q)
                qin[0, 0] = S[pos]                                   # teacher-force current token
                if DRAFT == 'true' or pos + SPEC_Q > N - 1:            # final window: true drafts so row N-1 is valid
                    tt = S[pos + 1:pos + SPEC_Q] + [S[N - 1]] * SPEC_Q
                    qin[0, 1:] = torch.tensor(tt[:SPEC_Q - 1], device=dev)
                qin_before = qin[0].tolist()
                if graph is not None:
                    graph.replay(); g = g_out; a1 = n_out
                else:
                    g, a1, _ = model.step_g(qin, pos_t, slot, temp_t)
                g = g.tolist(); a1 = int(a1.item())
                lgq = last_lg['lg'][0].float()                         # [Q, V]
            qin_l = qin_before
            for r in range(SPEC_Q):
                p = pos + r
                valid = qin_l[:r + 1] == S[pos:pos + r + 1]           # row r only meaningful if its prefix is the true tokens
                if valid and p < N and p not in pred:
                    pred[p] = g[r]
                    if p == N - 1:
                        final_lg = lgq[r].clone()
            # rows accepted in service semantics = a1; teacher forcing: advance the same way the service would
            acc_hist.append(a1)
            # TEACHER-FORCED advance: only rows whose token equals S are committed (service accepts by its own
            # greedy/sample; here the reference sequence S is the oracle). Rows beyond get rewritten next window.
            run = 0
            while run < SPEC_Q and pos + run < N and qin_l[run] == S[pos + run]: run += 1
            run = max(1, min(run, a1)) if os.environ.get('CAP_A1', '0') == '1' else max(1, run)
            pos = pos + run; pos_t.fill_(pos); steps += 1
            if pos > N - 1 and final_lg is not None:
                break
            if pos >= N:
                break
        mism = [p for p in range(P, N - 1) if p in pred and pred[p] != S[p + 1]]; nvalid = sum(1 for p in range(P, N - 1) if p in pred)
        topB = int(final_lg.argmax()) if final_lg is not None else -1
        log(f'B m={m} DRAFT={DRAFT} GRAPH={GRAPH} steps={steps} mean_acc={sum(acc_hist)/max(1,len(acc_hist)):.2f} '
            f'final argmax={topB} same_as_A={topB==topA} top5={final_lg.topk(5).indices.tolist() if final_lg is not None else None} '
            f'logit_maxdiff={(lgA-final_lg).abs().max().item() if final_lg is not None else -1:.4f} '
            f'greedy_mismatch_vs_teacher={len(mism)}/{nvalid}(valid rows of {N-1-P}) first_mism={mism[:5]}')
        torch.cuda.synchronize(); kvreport(snapA, snap(slot, 0, N), P)
        model.pool.release(slot)

    dist.barrier(); dist.destroy_process_group()
    log('SPEC_PROBE_DONE')
except Exception:
    traceback.print_exc()
    log('SPEC_PROBE_FAILED')
