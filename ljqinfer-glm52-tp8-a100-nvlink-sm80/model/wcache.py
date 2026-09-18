#!/usr/bin/env python3
"""wcache -- /dev/shm weight-snapshot daemon + client for ljqinfer.

Replaces the "resident engine + exec injection" dev loop (devd/devd_cp): every
experiment now runs in a *fresh* process; only the host-side weight bytes are
resident (tmpfs).  GPU state is rebuilt from scratch each time -> no pollution.

  build : load TP8 weights once (~5 min), snapshot the FINAL device-layout
          bytes of every tensor into /dev/shm/ljqw/tp8.blob + manifest.json.
          The CP variant (attn q_b/k_b/v_b replicated, 79 blocks) is stored as
          a small delta blob read straight from the GGUF (one copy, shared by
          all 8 ranks).
  hold  : resident CPU daemon: builds if missing, then heartbeats status.json.
          (tmpfs persists by itself; the daemon is liveness marker + rebuilder.)
  client: sys.path += [repo, repo/tools]; import wcache
              w = wcache.load("tp8")      # -> weights.Weights, all on GPU
              w = wcache.load("cp")       # -> CP-attention variant
              e = wcache.engine()               # fresh Engine, fixed-capacity snapshot H2D

Perf (8xA100, measured): register ~10 GiB/s, H2D 8-GPU aggregate ~70 GiB/s.
"""
from __future__ import annotations
import dataclasses, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from model import weights as W

BASE = "/dev/shm/ljqw"
MODEL = "/mnt/data/kw/models/huihui-ai/Huihui-GLM-5.2-abliterated-GGUF/UD-Q3_K_M"
ALIGN = 4096
CP_NAMES = ("attn_q_b", "attn_k_b", "attn_v_b")
_DC = {c.__name__: c for c in (W.Attn, W.DenseFFN, W.MoE, W.Layer, W.MTP, W.Weights)}
_T0 = time.perf_counter()


def _dt(name):
    return getattr(torch, name)


def _log(msg):
    print(f"[wcache +{time.perf_counter() - _T0:.1f}s] {msg}", flush=True)


# ================================================================== build ====
class _Alloc:
    def __init__(self):
        self.off = 0

    def take(self, nbytes):
        off = self.off
        self.off = (off + nbytes + ALIGN - 1) // ALIGN * ALIGN
        return off


def _encode(obj, alloc, writes, blob):
    """Weights tree -> JSON tree; tensor -> {"t":[blob,off,nbytes,shape,dtype,dev]}."""
    if isinstance(obj, torch.Tensor):
        t = obj.contiguous()
        nbytes = t.numel() * t.element_size()
        off = alloc.take(nbytes)
        writes.append((t, off, nbytes))
        return {"t": [blob, off, nbytes, list(t.shape),
                      str(t.dtype).split(".")[1], t.device.index]}
    if isinstance(obj, list):
        return {"l": [_encode(x, alloc, writes, blob) for x in obj]}
    if type(obj).__name__ in _DC:
        return {"d": type(obj).__name__,
                "f": {f.name: _encode(getattr(obj, f.name), alloc, writes, blob)
                      for f in dataclasses.fields(obj)}}
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return {"v": obj}
    raise TypeError(f"unhandled node type {type(obj)}")


def build():
    os.makedirs(BASE, exist_ok=True)
    _log(f"loading TP8 weights from {MODEL} ...")
    w = W.load_tp8(MODEL, progress=True)
    _log("TP8 loaded; snapshotting to tmpfs")

    alloc, writes = _Alloc(), []
    tree_tp8 = _encode(w, alloc, writes, "tp8")
    size = alloc.off
    blob = f"{BASE}/tp8.blob.tmp"
    with open(blob, "wb") as f:
        f.truncate(size)
    mm = np.memmap(blob, dtype=np.uint8, mode="r+")
    done = 0
    for t, off, n in writes:
        dst = torch.from_numpy(mm[off:off + n])
        dst.copy_(t.view(torch.uint8).reshape(-1))
        done += n
        if done % (32 << 30) < n:
            _log(f"  D2H {done >> 30} / {size >> 30} GiB")
    mm.flush()
    del mm
    _log(f"tp8.blob written: {size / 2**30:.1f} GiB, {len(writes)} tensors")
    del w, writes
    torch.cuda.empty_cache()

    # ---- CP delta: full (unsharded) q_b/k_b/v_b raw bytes, one copy each ----
    _log("building CP attention delta from GGUF ...")
    src = W.TensorSource(MODEL)
    alloc2 = _Alloc()
    cp_refs, raws = {}, []
    for L in list(range(W.N_LAYER)) + [W.MTP_LAYER]:
        for nm in CP_NAMES:
            raw = np.ascontiguousarray(np.asarray(src[f"blk.{L}.{nm}.weight"].data))
            off = alloc2.take(raw.nbytes)
            raws.append((raw, off))
            cp_refs[(L, nm)] = [off, raw.nbytes, list(raw.shape), str(raw.dtype)]
    size2 = alloc2.off
    blob2 = f"{BASE}/cp.blob.tmp"
    with open(blob2, "wb") as f:
        f.truncate(size2)
    mm2 = np.memmap(blob2, dtype=np.uint8, mode="r+")
    for raw, off in raws:
        mm2[off:off + raw.nbytes] = raw.reshape(-1).view(np.uint8)
    mm2.flush()
    del mm2, raws
    _log(f"cp.blob written: {size2 / 2**30:.2f} GiB")

    # ---- CP tree = tp8 tree with q_b/k_b/v_b swapped to replicated refs ----
    tree_cp = json.loads(json.dumps(tree_tp8))          # deep copy
    field = {"attn_q_b": "q_b", "attn_k_b": "k_b", "attn_v_b": "v_b"}

    def _patch_attn(attn_node, L):
        for nm, fld in field.items():
            off, nbytes, shape, dtype = cp_refs[(L, nm)]
            attn_node["f"][fld] = {"l": [
                {"t": ["cp", off, nbytes, shape, dtype, r]} for r in range(W.TP)]}

    for i, layer_node in enumerate(tree_cp["f"]["layers"]["l"]):
        _patch_attn(layer_node["f"]["attn"], i)
    mtp = tree_cp["f"]["mtp"]
    if "d" in mtp:
        _patch_attn(mtp["f"]["block"]["f"]["attn"], W.MTP_LAYER)

    man = {"meta": {"model": MODEL, "built": time.strftime("%F %T"),
                    "tp": W.TP, "n_layer": W.N_LAYER},
           "blobs": {"tp8": {"path": f"{BASE}/tp8.blob", "size": size},
                     "cp": {"path": f"{BASE}/cp.blob", "size": size2}},
           "trees": {"tp8": tree_tp8, "cp": tree_cp}}
    tmp = f"{BASE}/manifest.json.tmp"
    with open(tmp, "w") as f:
        json.dump(man, f)
    os.rename(blob, f"{BASE}/tp8.blob")
    os.rename(blob2, f"{BASE}/cp.blob")
    os.rename(tmp, f"{BASE}/manifest.json")
    _log("build complete")


# ================================================================== client ===
def _collect_refs(node, out):
    if "t" in node:
        out.append(node["t"])
    elif "l" in node:
        for x in node["l"]:
            _collect_refs(x, out)
    elif "d" in node:
        for x in node["f"].values():
            _collect_refs(x, out)


def _register(buf_addr, blob_offs, total, threads=8):
    """cudaHostRegister the blob in chunks cut at tensor boundaries (no tensor
    straddles two registrations, so every copy sees fully-pinned pages)."""
    cudart = torch.cuda.cudart()
    offs = sorted(set(blob_offs))
    cuts = [0]
    for i in range(1, threads):
        target = total * i // threads
        cuts.append(min(offs, key=lambda o: abs(o - target)))
    cuts = sorted(set(cuts)) + [total]
    spans = [(cuts[i], cuts[i + 1] - cuts[i])
             for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]

    def _reg(span):
        rc = cudart.cudaHostRegister(buf_addr + span[0], span[1], 0)
        if rc != 0:
            raise RuntimeError(f"cudaHostRegister rc={rc} span={span}")

    with ThreadPoolExecutor(len(spans)) as ex:
        list(ex.map(_reg, spans))
    return spans


def load(version="tp8", register=False, verbose=True):
    """Rebuild weights.Weights on the 8 GPUs from the tmpfs snapshot.

    register=False (default): plain pageable copies, 1 thread per GPU.
    torch's internal pinned staging pipeline hits ~32 GiB/s aggregate,
    while cudaHostRegister itself crawls at ~3 GiB/s (kernel page pinning)
    -- so registering is a net loss for one-shot loads."""
    t0 = time.perf_counter()
    with open(f"{BASE}/manifest.json") as f:
        man = json.load(f)
    tree = man["trees"][version]
    refs = []
    _collect_refs(tree, refs)
    blobs_needed = {r[0] for r in refs}

    for d in range(W.TP):                       # init contexts up front
        torch.cuda.set_device(d)
        torch.cuda.current_stream(d)
    if verbose:
        _log(f"[{version}] cuda ctx up: {time.perf_counter() - t0:.1f}s")

    host, spans, mms = {}, {}, {}
    for b in blobs_needed:
        info = man["blobs"][b]
        mm = np.memmap(info["path"], dtype=np.uint8, mode="r+")
        mms[b] = mm
        host[b] = torch.from_numpy(mm)
        if register:
            offs = [r[1] for r in refs if r[0] == b] + [info["size"]]
            spans[b] = _register(host[b].data_ptr(), offs[:-1], info["size"])
            if verbose:
                _log(f"[{version}] {b} registered "
                     f"{info['size'] / 2**30:.1f} GiB: {time.perf_counter() - t0:.1f}s")

    streams = [torch.cuda.Stream(device=d) for d in range(W.TP)]
    tasks = [[] for _ in range(W.TP)]           # per-device copy queues

    def _dec(node):
        if "t" in node:
            b, off, n, shape, dtype, dev = node["t"]
            hv = host[b][off:off + n].view(_dt(dtype)).view(shape)
            with torch.cuda.device(dev), torch.cuda.stream(streams[dev]):
                d = torch.empty(shape, dtype=_dt(dtype), device=f"cuda:{dev}")
                if register:
                    d.copy_(hv, non_blocking=True)
                else:
                    tasks[dev].append((hv, d))
            return d
        if "l" in node:
            return [_dec(x) for x in node["l"]]
        if "d" in node:
            return _DC[node["d"]](**{k: _dec(v) for k, v in node["f"].items()})
        return node["v"]

    w = _dec(tree)
    if not register:
        def _drain(dev):
            torch.cuda.set_device(dev)
            for hv, d in tasks[dev]:
                d.copy_(hv)                     # blocking pageable copy
        with ThreadPoolExecutor(W.TP) as ex:
            list(ex.map(_drain, range(W.TP)))
    for s in streams:
        s.synchronize()
    if verbose:
        _log(f"[{version}] H2D done: {time.perf_counter() - t0:.1f}s")
    if register:
        cudart = torch.cuda.cudart()
        for b, sp in spans.items():
            base = host[b].data_ptr()
            for off, _ in sp:
                cudart.cudaHostUnregister(base + off)
    if verbose:
        _log(f"[{version}] weights ready: {time.perf_counter() - t0:.1f}s "
             f"({sum(r[2] for r in refs) / 2**30:.1f} GiB, {len(refs)} tensors)")
    return w


def engine(version="tp8", register=False, **kw):
    """Fresh Engine in THIS process; weights H2D from snapshot (no GGUF load)."""
    from model import model as M
    w = load(version, register=register)
    orig = W.load_tp8
    W.load_tp8 = lambda *a, **k: w
    try:
        e = M.Engine.load(**kw)
    finally:
        W.load_tp8 = orig
    return e


# ================================================================== daemon ===
def hold(rebuild=False):
    if rebuild or not os.path.exists(f"{BASE}/manifest.json"):
        _log("no snapshot -> building in subprocess")
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "build"])
        if rc != 0:
            _log(f"BUILD FAILED rc={rc}")
            sys.exit(rc)
    _log(f"holding (pid {os.getpid()}); snapshot lives in {BASE}")
    while True:
        st = {"pid": os.getpid(), "alive": time.strftime("%F %T"),
              "files": {f: os.path.getsize(f"{BASE}/{f}")
                        for f in os.listdir(BASE) if not f.endswith(".tmp")}}
        with open(f"{BASE}/status.json", "w") as f:
            json.dump(st, f)
        time.sleep(30)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "hold"
    if cmd == "build":
        build()
    elif cmd == "hold":
        hold(rebuild="--rebuild" in sys.argv)
    else:
        sys.exit("usage: wcache.py [build|hold [--rebuild]]")
