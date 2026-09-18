import sys
import statistics
import torch

ROOT = "/mnt/data/kw/ljqinfer_tp8"
sys.path.insert(0, ROOT)
from torch.utils.cpp_extension import load
from model import wcache


def latency_ms(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize()
        vals.append(a.elapsed_time(b))
    return statistics.median(vals)


def main():
    torch.manual_seed(20260812)
    mod = load(name="special_moe_quant_decode_leaf_v2",
               sources=[ROOT + "/ops/special_moe_quant_decode.cu"],
               extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], verbose=False)
    weights = wcache.load("tp8", register=False, verbose=False)
    cases = [
        ("iq4xs_q5k", weights.layers[8].ffn, mod.special_iq4xs_q5k),
        ("iq3xxs_q6k", weights.layers[75].ffn, mod.special_iq3xxs_q6k),
        ("q3k_q4k", weights.mtp.block.ffn, mod.special_q3k_q4k),
    ]
    failed = False
    for name, w, op in cases:
        dev = w.gate_exps[0].device
        torch.cuda.set_device(dev)
        for q in (1, 2, 4):
            x = (torch.randn(q, 6144, device=dev, dtype=torch.float32) * .2).contiguous()
            eid = torch.stack([torch.randperm(256, device=dev)[:8] for _ in range(q)]).to(torch.int64).contiguous()
            ew = torch.softmax(torch.randn(q, 8, device=dev), -1).float().contiguous()
            out = torch.empty(q, 6144, device=dev, dtype=torch.float32)
            fn = lambda: op(w.gate_exps[0], w.up_exps[0], w.down_exps[0], x, eid, ew, out)
            fn(); torch.cuda.synchronize(dev)

            # No allocator traffic is allowed after caller-owned tensors exist.
            before = torch.cuda.memory_allocated(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            for _ in range(3): fn()
            torch.cuda.synchronize(dev)
            after = torch.cuda.memory_allocated(dev)
            peak = torch.cuda.max_memory_allocated(dev)
            alloc_ok = (after == before and peak == before)

            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(dev)
            with torch.cuda.graph(graph):
                fn()
            x.add_(0.03125)
            eager = torch.empty_like(out)
            op(w.gate_exps[0], w.up_exps[0], w.down_exps[0], x, eid, ew, eager)
            torch.cuda.synchronize(dev)
            graph.replay(); torch.cuda.synchronize(dev)
            diff = (out - eager).abs().max().item()
            graph_ok = torch.isfinite(out).all().item() and diff < 1e-5
            ms = latency_ms(graph.replay)
            ok = alloc_ok and graph_ok
            failed |= not ok
            print(name, "Q", q, "PASS" if ok else "FAIL",
                  {"graph_max_abs": diff, "alloc_delta": after-before,
                   "peak_delta": peak-before, "graph_median_ms": ms}, flush=True)
            del graph, eager, out, ew, eid, x
        torch.cuda.empty_cache()
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
