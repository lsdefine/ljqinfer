
import io
p = "bench/decode_graph.py"
s = io.open(p, encoding="utf-8").read()
anchor = "    if rank == 0:\n        gap = float("
assert anchor in s, "anchor missing"
prof = """    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
    if rank == 0:
        ka = list(prof.key_averages())
        tot = sum(e.self_device_time_total for e in ka)
        print("PROF_MS_PER_REPLAY", round(tot / 1000.0 / 5, 3), flush=True)
        print("PROF_KERNELS_PER_REPLAY", round(sum(e.count for e in ka) / 5.0, 1), flush=True)
        for e in sorted(ka, key=lambda x: -x.self_device_time_total)[:30]:
            print("K", round(e.self_device_time_total / 1000.0 / 5, 3),
                  round(e.count / 5.0, 1), str(e.key)[:70], flush=True)
"""
s = s.replace(anchor, prof + anchor, 1)
io.open("bench/decode_prof.py", "w", encoding="utf-8").write(s)
print("WROTE", len(s))
