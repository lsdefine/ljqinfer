"""B1Q8 full-step gate: Transformer.step_g eager (slot0) vs CUDA-graph (slot1), both vs gold."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os, sys, json, time, torch, torch.distributed as dist
RANK = int(os.environ.get('RANK', '0'))
def log(*a):
    if RANK == 0: print(*a, flush=True)
N_GEN = int(os.environ.get('N_GEN', '40')); Q = 8
os.environ['LJQ_DECODE_G'] = '1'
try:
    dist.init_process_group('nccl'); torch.cuda.set_device(RANK)
    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    from tokenizers import Tokenizer
    arch.init_distributed()
    tk = Tokenizer.from_file('/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json')
    ids = json.load(open('/tmp/c1_ids.json'))
    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    torch.set_default_dtype(torch.bfloat16)
    args = make_args(); model = Transformer(args); bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]; assert not bad, bad
    dev = f'cuda:{RANK}'
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu': m._buffers[k] = v.to(dev)
    torch.set_default_device(dev); dist.barrier()
    P = len(ids); toks = torch.tensor([ids], device=dev)
    gold = json.load(open('/tmp/gold_c1_out.json'))

    def run(slot, use_graph):
        model.pool.ensure(slot, P + N_GEN + 2 * Q + 8)
        out, _, mh = model(toks, start_pos=0, slot=slot)
        first = out.view(-1)[-1:].clone()
        model.forward_spec(first, mh, 0, slot=slot)                  # write prompt draft main_kv
        d0, _, _ = model.forward_spec(first, mh[:, -1:], P - 1, slot=slot)  # [1, 6]
        qin = torch.zeros(1, Q, dtype=torch.int64, device=dev)
        qin[0, :d0.size(1)] = d0[0]; qin[0, d0.size(1):] = d0[0, -1]
        pos_t = torch.full((1,), P, dtype=torch.int64, device=dev)
        if use_graph:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            q_bak, p_bak = qin.clone(), pos_t.clone()
            with torch.cuda.stream(s):
                for _ in range(2): model.step_g(qin, pos_t, slot)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                g_out, n_out, _ = model.step_g(qin, pos_t, slot)
            torch.cuda.synchronize(); dist.barrier()
            # warmup/capture advanced state: restore by redoing prefill + draft
            out, _, mh = model(toks, start_pos=0, slot=slot)
            model.forward_spec(first, mh, 0, slot=slot)
            d0, _, _ = model.forward_spec(first, mh[:, -1:], P - 1, slot=slot)
            qin[0, :d0.size(1)] = d0[0]; qin[0, d0.size(1):] = d0[0, -1]; pos_t.fill_(P)
        gen = [int(first)]; ts = []; acc = []
        while len(gen) < N_GEN + 1:
            torch.cuda.synchronize(); ta = time.time()
            if use_graph: graph.replay()
            else: g_out, n_out, _ = model.step_g(qin, pos_t, slot)
            torch.cuda.synchronize(); ts.append(time.time() - ta)
            n = int(n_out); acc.append(n - 1)
            gen += g_out[:n].tolist()
        return gen[:N_GEN + 1], ts, acc

    gen_e, t_e, acc_e = run(0, False)
    log('eager  gen', gen_e); log('eager  acc', acc_e)
    gen_g, t_g, acc_g = run(1, True)
    log('graph  gen', gen_g); log('graph  acc', acc_g)
    gd = gold[:N_GEN + 1]
    ok_e, ok_g = gen_e == gd, gen_g == gd
    if RANK == 0:
        log('gold ', gd)
        log('TEXT graph:', repr(tk.decode(gen_g, skip_special_tokens=False)))
        log(f'eager step {sum(t_e[2:])/len(t_e[2:])*1000:.1f}ms x{len(t_e)} steps  mean_acc {sum(acc_e)/len(acc_e):.2f} | '
            f'graph step {sum(t_g[2:])/len(t_g[2:])*1000:.1f}ms x{len(t_g)} steps  mean_acc {sum(acc_g)/len(acc_g):.2f}')
        log('eager==gold', ok_e, ' graph==gold', ok_g, ' eager==graph', gen_e == gen_g)
        log('GATE PASS' if (ok_e and ok_g) else 'GATE FAIL')
    dist.barrier(); dist.destroy_process_group()
except Exception:
    import traceback; traceback.print_exc(); sys.stdout.flush(); os._exit(1)
