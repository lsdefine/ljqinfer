"""Summarise per-request metrics from the service log."""
import json
import sys


def band(name, vals, fmt="%.2f"):
    if not vals:
        print("  %-24s n/a" % name)
        return
    vals = sorted(vals)
    n = len(vals)
    args = (vals[0], vals[n // 2], vals[-1], sum(vals) / n)
    print(("  %-24s n=%-3d min=" + fmt + " med=" + fmt + " max=" + fmt +
           " avg=" + fmt) % ((name, n) + args))


def col(rows, key):
    return [r[key] for r in rows if isinstance(r.get(key), (int, float))]


def main(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        i = line.find("metrics=")
        if i >= 0:
            rows.append(json.loads(line[i + 8:]))
    print("requests=%d" % len(rows))
    solo = [r for r in rows if (r.get("queue_wait_seconds") or 0) < 0.3]
    conc = [r for r in rows if (r.get("queue_wait_seconds") or 0) >= 0.3]

    print("[speed] all requests")
    band("model_step_host_ms", col(rows, "model_step_host_ms"))
    band("wall_ms_per_step", col(rows, "wall_ms_per_step"))
    band("control_ms_per_step", col(rows, "control_ms_per_step"))
    band("mtp_accepted_per_step", col(rows, "mtp_accepted_per_step"))
    band("decode_tps", col(rows, "decode_tps"))
    print("[speed] solo (queue_wait<0.3s) n=%d" % len(solo))
    band("model_step_host_ms", col(solo, "model_step_host_ms"))
    print("[speed] batched/queued n=%d" % len(conc))
    band("model_step_host_ms", col(conc, "model_step_host_ms"))
    band("queue_wait_seconds", col(conc, "queue_wait_seconds"))
    band("prefill_stall_seconds", col(conc, "prefill_stall_seconds"))

    print("[latency]")
    band("ttft_seconds", col(rows, "time_to_first_token_seconds"))
    band("total_seconds", col(rows, "total_seconds"))

    print("[cache]")
    hot = [r for r in rows if (r.get("cache_hit_tokens") or 0) > 0]
    cold = [r for r in rows if not (r.get("cache_hit_tokens") or 0)]
    for tag, grp in (("cold", cold), ("hit", hot)):
        if not grp:
            print("  %-6s none" % tag)
            continue
        pt = sum(col(grp, "prefill_tokens"))
        ps = sum(col(grp, "model_prefill_seconds"))
        hits = sum(col(grp, "cache_hit_tokens"))
        ins = sum(col(grp, "input_tokens"))
        print("  %-6s n=%-3d input=%d hit=%d (%.0f%%) prefill_tok=%d "
              "prefill_s=%.2f tok/s=%.0f"
              % (tag, len(grp), ins, hits, 100.0 * hits / max(ins, 1), pt, ps,
                 pt / max(ps, 1e-9)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/srv.log")
