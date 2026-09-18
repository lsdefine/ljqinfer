#!/usr/bin/env python3
"""Shared-memory packed-weight cache for DeepSeek-V4-Flash-0731 TP8.

The cache preserves checkpoint bytes exactly.  Replicated tensors are stored once;
TP tensors occupy exactly their original byte count, with rank shards made
contiguous.  No FP8/FP4 dequantization or requantization happens here.

Usage (from repository root):
    python -m model.wcache plan
    python -m model.wcache build
    python -m model.wcache build-final
    python -m model.wcache verify
    python -m model.wcache hold
    python -m model.wcache status
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import mmap
import os
import shutil
import signal
import struct
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from model import weights as W

from model.config import ALIGNMENT, CACHE_DIR, CACHE_FORMAT, MODEL_DIR, TP

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}


def _log(msg: str) -> None:
    print(f"[wcache] {msg}", flush=True)


def _align(x: int) -> int:
    return (x + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def _source_fingerprint(model_dir: Path) -> dict:
    files = [model_dir / "config.json", model_dir / "model.safetensors.index.json"]
    h = hashlib.sha256()
    out = []
    for p in files:
        st = p.stat()
        data = p.read_bytes()
        h.update(p.name.encode())
        h.update(data)
        out.append({"name": p.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return {"sha256": h.hexdigest(), "files": out}


@dataclass(frozen=True)
class SourceTensor:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    file_offset: int
    nbytes: int


def _read_safetensor_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"truncated safetensors header: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_len))
    return 8 + header_len, header


def scan_checkpoint(model_dir: Path = MODEL_DIR) -> list[SourceTensor]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        by_file.setdefault(filename, []).append(name)
    tensors: list[SourceTensor] = []
    for filename in sorted(by_file):
        data_base, header = _read_safetensor_header(model_dir / filename)
        for name in by_file[filename]:
            meta = header[name]
            dtype = meta["dtype"]
            shape = tuple(int(x) for x in meta["shape"])
            begin, end = (int(x) for x in meta["data_offsets"])
            nbytes = end - begin
            expected = int(np.prod(shape, dtype=np.int64)) * DTYPE_BYTES[dtype]
            if nbytes != expected:
                raise RuntimeError(f"byte-size mismatch for {name}: {nbytes} != {expected}")
            tensors.append(SourceTensor(name, filename, dtype, shape,
                                        data_base + begin, nbytes))
    tensors.sort(key=lambda x: x.name)
    if len(tensors) != len(weight_map):
        raise RuntimeError("checkpoint scan did not cover the full weight map")
    return tensors


def _component(name: str) -> str | None:
    parts = name.split(".")
    for key in ("embed", "head", "wq_b", "wo_a", "wo_b", "attn_sink",
                "weights_proj", "markov_w1", "markov_w2"):
        if key in parts:
            return key
    return None


def split_dim(name: str, shape: tuple[int, ...]) -> int | None:
    """Return the stored-byte tensor dimension sharded by TP8.

    FP4 expert matrices are byte-packed on their input dimension.  Their stored
    shapes already account for two logical values per byte, so ordinary slicing
    on the stored dimension preserves the official nibble layout and UE8M0 groups.
    """
    routed = ".ffn.experts." in name
    shared = ".ffn.shared_experts." in name
    if routed or shared:
        if ".w1." in name or ".w3." in name:
            return 0
        if ".w2." in name:
            return 1
    key = _component(name)
    if key in {"embed", "head", "wq_b", "wo_a", "attn_sink",
               "weights_proj", "markov_w1", "markov_w2"}:
        return 0
    if key == "wo_b":
        return 1
    return None


def make_plan(model_dir: Path = MODEL_DIR) -> dict:
    tensors = scan_checkpoint(model_dir)
    cursor = 0
    entries = {}
    split_count = 0
    replica_count = 0
    source_bytes = 0
    for t in tensors:
        source_bytes += t.nbytes
        dim = split_dim(t.name, t.shape)
        cursor = _align(cursor)
        if dim is None:
            entries[t.name] = {
                "dtype": t.dtype, "shape": list(t.shape), "layout": "replica",
                "offset": cursor, "nbytes": t.nbytes,
                "source": [t.file, t.file_offset],
            }
            cursor += t.nbytes
            replica_count += 1
            continue
        if dim >= len(t.shape) or t.shape[dim] % TP:
            raise RuntimeError(f"cannot TP{TP}-split {t.name} shape={t.shape} dim={dim}")
        shard_shape = list(t.shape)
        shard_shape[dim] //= TP
        shard_bytes = t.nbytes // TP
        shards = []
        for rank in range(TP):
            shards.append({"offset": cursor + rank * shard_bytes,
                           "nbytes": shard_bytes, "shape": shard_shape})
        entries[t.name] = {
            "dtype": t.dtype, "shape": list(t.shape), "layout": "tp",
            "split_dim": dim, "shards": shards,
            "source": [t.file, t.file_offset],
        }
        cursor += t.nbytes
        split_count += 1
    return {
        "format": CACHE_FORMAT,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_dir": str(model_dir),
        "tp": TP,
        "alignment": ALIGNMENT,
        "source_fingerprint": _source_fingerprint(model_dir),
        "tensor_count": len(entries),
        "split_tensor_count": split_count,
        "replica_tensor_count": replica_count,
        "source_bytes": source_bytes,
        "blob_bytes": cursor,
        "entries": entries,
    }


def _copy_contiguous(src_fd: int, dst_fd: int, src_off: int, dst_off: int,
                     nbytes: int, chunk: int = 1 << 30) -> None:
    left = nbytes
    while left:
        n = min(left, chunk)
        data = os.pread(src_fd, n, src_off)
        if len(data) != n:
            raise RuntimeError("short checkpoint read")
        written = os.pwrite(dst_fd, data, dst_off)
        if written != n:
            raise RuntimeError("short cache write")
        src_off += n
        dst_off += n
        left -= n


def _copy_dim1(src_mm: mmap.mmap, src_off: int, dst_mm: mmap.mmap,
               dst_off: int, shape: tuple[int, ...], itemsize: int) -> None:
    if len(shape) != 2:
        raise RuntimeError(f"dim1 repack currently requires 2D tensor, got {shape}")
    rows, cols = shape
    if cols % TP:
        raise RuntimeError(f"dim1 not divisible by TP{TP}: {shape}")
    row_bytes = cols * itemsize
    shard_row_bytes = row_bytes // TP
    nbytes = rows * row_bytes
    src = np.ndarray((rows, row_bytes), dtype=np.uint8,
                     buffer=src_mm, offset=src_off)
    shard_bytes = nbytes // TP
    for rank in range(TP):
        dst = np.ndarray((rows, shard_row_bytes), dtype=np.uint8,
                         buffer=dst_mm, offset=dst_off + rank * shard_bytes)
        dst[:] = src[:, rank * shard_row_bytes:(rank + 1) * shard_row_bytes]


def build(model_dir: Path = MODEL_DIR, cache_dir: Path = CACHE_DIR) -> None:
    plan = make_plan(model_dir)
    parent = cache_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / (cache_dir.name + f".building.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    blob = tmp / "weights.blob"
    _log(f"building {plan['tensor_count']} tensors, {plan['blob_bytes']/2**30:.2f} GiB")
    with blob.open("w+b") as out:
        out.truncate(plan["blob_bytes"])
        dst_mm = mmap.mmap(out.fileno(), 0, access=mmap.ACCESS_WRITE)
        fds: dict[str, int] = {}
        source_maps: dict[str, mmap.mmap] = {}
        try:
            for i, (name, e) in enumerate(plan["entries"].items(), 1):
                filename, src_off = e["source"]
                src_path = model_dir / filename
                if filename not in fds:
                    fds[filename] = os.open(src_path, os.O_RDONLY)
                fd = fds[filename]
                if e["layout"] == "replica" or e.get("split_dim") == 0:
                    dst_off = e["offset"] if e["layout"] == "replica" else e["shards"][0]["offset"]
                    nbytes = e["nbytes"] if e["layout"] == "replica" else sum(s["nbytes"] for s in e["shards"])
                    _copy_contiguous(fd, out.fileno(), src_off, dst_off, nbytes)
                else:
                    if filename not in source_maps:
                        source_maps[filename] = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                    _copy_dim1(source_maps[filename], src_off, dst_mm,
                               e["shards"][0]["offset"], tuple(e["shape"]),
                               DTYPE_BYTES[e["dtype"]])
                if i % 2000 == 0 or i == plan["tensor_count"]:
                    _log(f"copied {i}/{plan['tensor_count']}")
            dst_mm.flush()
        finally:
            dst_mm.close()
            for src_mm in source_maps.values():
                src_mm.close()
            for fd in fds.values():
                os.close(fd)
    plan["built"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_json(tmp / "manifest.json", plan)
    _atomic_json(tmp / "status.json", {"state": "ready", "pid": None,
                 "blob_bytes": plan["blob_bytes"], "updated": time.time()})
    old = parent / (cache_dir.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if cache_dir.exists():
        os.replace(cache_dir, old)
    os.replace(tmp, cache_dir)
    if old.exists():
        shutil.rmtree(old)
    _log(f"ready: {cache_dir}")


def load_manifest(cache_dir: Path = CACHE_DIR) -> dict:
    p = cache_dir / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"wcache manifest missing: {p}")
    m = json.loads(p.read_text())
    if m.get("format") != CACHE_FORMAT or m.get("tp") != TP:
        raise RuntimeError("incompatible wcache format")
    if m.get("source_fingerprint") != _source_fingerprint(Path(m["model_dir"])):
        raise RuntimeError("wcache source fingerprint is stale")
    blob = cache_dir / "weights.blob"
    if blob.stat().st_size != m["blob_bytes"]:
        raise RuntimeError("wcache blob size mismatch")
    return m


def _sample_names(entries: dict, full: bool) -> Iterable[str]:
    if full:
        return entries.keys()
    names = list(entries)
    picked = {names[0], names[-1]}
    for marker in ("embed.weight", "head.weight", ".experts.0.w1.weight",
                   ".experts.0.w2.weight", ".wq_b.weight", ".wo_b.weight",
                   "mtp.2.markov_head.markov_w1.weight"):
        picked.update(n for n in names if marker in n)
    return sorted(picked)


def verify(cache_dir: Path = CACHE_DIR, full: bool = False) -> None:
    m = load_manifest(cache_dir)
    model_dir = Path(m["model_dir"])
    blob_path = cache_dir / "weights.blob"
    checked = 0
    with blob_path.open("rb") as bf:
        blob_mm = mmap.mmap(bf.fileno(), 0, access=mmap.ACCESS_READ)
        source_maps: dict[str, tuple[object, mmap.mmap]] = {}
        try:
            for name in _sample_names(m["entries"], full):
                e = m["entries"][name]
                filename, src_off = e["source"]
                if filename not in source_maps:
                    f = (model_dir / filename).open("rb")
                    source_maps[filename] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
                src = source_maps[filename][1]
                nbytes = int(np.prod(e["shape"])) * DTYPE_BYTES[e["dtype"]]
                if e["layout"] == "replica":
                    got = blob_mm[e["offset"]:e["offset"] + nbytes]
                    expect = src[src_off:src_off + nbytes]
                    if got != expect:
                        raise RuntimeError(f"verify mismatch: {name}")
                elif e["split_dim"] == 0:
                    off = e["shards"][0]["offset"]
                    got = blob_mm[off:off + nbytes]
                    expect = src[src_off:src_off + nbytes]
                    if got != expect:
                        raise RuntimeError(f"verify mismatch: {name}")
                else:
                    rows, cols = e["shape"]
                    item = DTYPE_BYTES[e["dtype"]]
                    rowb, shardb = cols * item, cols * item // TP
                    src_a = np.ndarray((rows, rowb), np.uint8, buffer=src, offset=src_off)
                    for rank, s in enumerate(e["shards"]):
                        got = np.ndarray((rows, shardb), np.uint8, buffer=blob_mm,
                                         offset=s["offset"])
                        expect = src_a[:, rank * shardb:(rank + 1) * shardb]
                        if not np.array_equal(got, expect):
                            raise RuntimeError(f"verify mismatch: {name} rank={rank}")
                checked += 1
        finally:
            blob_mm.close()
            for f, mm in source_maps.values():
                mm.close(); f.close()
    _log(f"verify PASS ({checked} tensors, full={full})")


def hold(cache_dir: Path = CACHE_DIR) -> None:
    m = load_manifest(cache_dir)
    blob = (cache_dir / "weights.blob").open("rb")
    mm = mmap.mmap(blob.fileno(), 0, access=mmap.ACCESS_READ)
    stop = False
    def _stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    _atomic_json(cache_dir / "status.json", {"state": "held", "pid": os.getpid(),
                 "blob_bytes": m["blob_bytes"], "updated": time.time()})
    _log(f"holding {m['blob_bytes']/2**30:.2f} GiB pid={os.getpid()}")
    try:
        while not stop:
            _atomic_json(cache_dir / "status.json", {"state": "held", "pid": os.getpid(),
                         "blob_bytes": m["blob_bytes"], "updated": time.time()})
            time.sleep(10)
    finally:
        mm.close(); blob.close()
        _atomic_json(cache_dir / "status.json", {"state": "ready", "pid": None,
                     "blob_bytes": m["blob_bytes"], "updated": time.time()})


# ======================================================== final-tree cache ====
# The source manifest/blob are used only while constructing final device-layout
# weights.  The files below serialize the same final dataclass tree consumed by
# Engine, matching the reference wcache contract.
_TREE_DC = {
    cls.__name__: cls for cls in (
        W.QuantPair, W.Compressor, W.Indexer, W.Attention, W.ExpertBank,
        W.Router, W.MoE, W.Layer, W.MTP, W.Weights,
    )
}


class _TreeAlloc:
    def __init__(self):
        self.offset = 0

    def take(self, nbytes: int) -> int:
        offset = self.offset
        self.offset = _align(offset + nbytes)
        return offset


def _encode_tree(obj, alloc: _TreeAlloc, writes: list, blob: str):
    """Encode a final dataclass tree; tensor leaves retain their CUDA rank."""
    if isinstance(obj, torch.Tensor):
        tensor = obj.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        offset = alloc.take(nbytes)
        writes.append((tensor, offset, nbytes))
        return {"t": [blob, offset, nbytes, list(tensor.shape),
                      str(tensor.dtype).split(".")[1], tensor.device.index]}
    if isinstance(obj, list):
        return {"l": [_encode_tree(item, alloc, writes, blob) for item in obj]}
    if dataclasses.is_dataclass(obj):
        name = type(obj).__name__
        if name not in _TREE_DC:
            raise TypeError(f"unregistered weight dataclass {name}")
        return {
            "d": name,
            "f": {
                field.name: _encode_tree(getattr(obj, field.name), alloc,
                                         writes, blob)
                for field in dataclasses.fields(obj)
            },
        }
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return {"v": obj}
    raise TypeError(f"unhandled final-tree node {type(obj)}")


def _collect_tree_refs(node, out: list) -> None:
    if "t" in node:
        out.append(node["t"])
    elif "l" in node:
        for child in node["l"]:
            _collect_tree_refs(child, out)
    elif "d" in node:
        for child in node["f"].values():
            _collect_tree_refs(child, out)


def snapshot_tree(obj, cache_dir: Path = CACHE_DIR,
                  stem: str = "tp8") -> dict:
    """Atomically snapshot a final CUDA weight tree."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    alloc, writes = _TreeAlloc(), []
    tree = _encode_tree(obj, alloc, writes, stem)
    size = alloc.offset
    blob_path = cache_dir / f"{stem}.blob"
    blob_tmp = cache_dir / f"{stem}.blob.tmp"
    manifest_path = cache_dir / f"{stem}_manifest.json"
    if blob_tmp.exists():
        blob_tmp.unlink()
    try:
        with blob_tmp.open("wb") as file:
            file.truncate(size)
        mm = np.memmap(blob_tmp, dtype=np.uint8, mode="r+")
        done = 0
        for tensor, offset, nbytes in writes:
            dst = torch.from_numpy(mm[offset:offset + nbytes])
            dst.copy_(tensor.view(torch.uint8).reshape(-1))
            done += nbytes
            if done // (4 << 30) != (done - nbytes) // (4 << 30):
                _log(f"final-tree D2H {done / 2**30:.1f}/{size / 2**30:.1f} GiB")
        mm.flush()
        del mm
        os.replace(blob_tmp, blob_path)
        manifest = {
            "format": "ljq-dsv4f-final-tree-v1",
            "built": time.strftime("%F %T"),
            "stem": stem,
            "blob": str(blob_path),
            "blob_bytes": size,
            "tensor_count": len(writes),
            "tree": tree,
        }
        _atomic_json(manifest_path, manifest)
        _log(f"{stem} final tree ready: {size / 2**30:.3f} GiB, "
             f"{len(writes)} tensors")
        return manifest
    except BaseException:
        if blob_tmp.exists():
            blob_tmp.unlink()
        raise


def load(stem: str = "tp8", cache_dir: Path = CACHE_DIR,
         verbose: bool = True, rank: int = None):
    """Rebuild the final weight tree on eight GPUs and release host mappings.

    rank=None: loop-TP8 mode, full tree across 8 GPUs (original behaviour).
    rank=i: NCCL mode, only this rank's shard of each TP-sharded list is
    loaded (as a bare tensor, not a list) onto cuda:i; replicated tensors
    are loaded onto cuda:i as well.
    """
    started = time.perf_counter()
    cache_dir = Path(cache_dir)
    manifest = json.loads(
        (cache_dir / f"{stem}_manifest.json").read_text()
    )
    if manifest.get("format") != "ljq-dsv4f-final-tree-v1":
        raise RuntimeError(
            f"bad final-tree format {manifest.get('format')!r}"
        )
    refs = []
    _collect_tree_refs(manifest["tree"], refs)
    prev_device = torch.cuda.current_device()
    for device in range(TP):
        torch.cuda.set_device(device)
        torch.cuda.current_stream(device)
    mm = np.memmap(manifest["blob"], dtype=np.uint8, mode="r+")
    host = torch.from_numpy(mm)
    tasks = [[] for _ in range(TP)]

    def decode(node):
        if "t" in node:
            _blob, offset, nbytes, shape, dtype_name, device = node["t"]
            if not 0 <= device < TP:
                raise RuntimeError(f"invalid tensor device {device}")
            if rank is not None:
                device = rank
            dtype = getattr(torch, dtype_name)
            host_view = host[offset:offset + nbytes].view(dtype).reshape(shape)
            with torch.cuda.device(device):
                dst = torch.empty(
                    shape, dtype=dtype, device=f"cuda:{device}"
                )
            tasks[device].append((host_view, dst))
            return dst
        if "l" in node:
            children = node["l"]
            if rank is not None and all("t" in c for c in children):
                if len(children) == TP:      # TP-sharded: keep only our shard
                    return decode(children[rank])
                if len(children) == 1:       # replicated
                    return decode(children[0])
            return [decode(child) for child in children]
        if "d" in node:
            cls = _TREE_DC[node["d"]]
            return cls(**{
                name: decode(value)
                for name, value in node["f"].items()
            })
        return node["v"]

    result = decode(manifest["tree"])

    def drain(device: int) -> None:
        torch.cuda.set_device(device)
        for host_view, dst in tasks[device]:
            dst.copy_(host_view)

    with ThreadPoolExecutor(max_workers=TP) as pool:
        list(pool.map(drain, range(TP)))
    if verbose:
        _log(f"{stem} H2D done in {time.perf_counter() - started:.2f}s")
    tasks.clear()
    del host
    mm._mmap.close()
    torch.cuda.set_device(prev_device)
    if verbose:
        _log(f"{stem} weights ready: "
             f"{manifest['blob_bytes'] / 2**30:.3f} GiB, "
             f"{len(refs)} tensors, {time.perf_counter() - started:.2f}s")
    return result


def build_final(cache_dir: Path = CACHE_DIR, *, load_mtp: bool = True) -> dict:
    """Build the complete final-tree cache from the source-byte cache."""
    weights = W.load_tp8(cache_dir, load_mtp=load_mtp, progress=True)
    try:
        return snapshot_tree(weights, cache_dir, "tp8")
    finally:
        del weights
        torch.cuda.empty_cache()


def status(cache_dir: Path = CACHE_DIR) -> None:
    if not cache_dir.exists():
        print(json.dumps({"state": "missing", "path": str(cache_dir)})); return
    try:
        m = load_manifest(cache_dir)
        s = json.loads((cache_dir / "status.json").read_text()) if (cache_dir / "status.json").exists() else {}
        print(json.dumps({"path": str(cache_dir), "format": m["format"],
                          "tensor_count": m["tensor_count"],
                          "blob_bytes": m["blob_bytes"], **s}, indent=2))
    except Exception as e:
        print(json.dumps({"state": "invalid", "path": str(cache_dir), "error": str(e)}, indent=2))
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("plan", "build", "build-final", "verify", "hold", "status", "clean"))
    ap.add_argument("--full", action="store_true", help="verify every tensor")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    if args.action == "plan":
        p = make_plan(args.model_dir)
        print(json.dumps({k: v for k, v in p.items() if k != "entries"}, indent=2))
    elif args.action == "build": build(args.model_dir, args.cache_dir)
    elif args.action == "build-final": build_final(args.cache_dir)
    elif args.action == "verify": verify(args.cache_dir, args.full)
    elif args.action == "hold": hold(args.cache_dir)
    elif args.action == "status": status(args.cache_dir)
    elif args.action == "clean":
        if args.cache_dir.exists(): shutil.rmtree(args.cache_dir)
        _log(f"removed {args.cache_dir}")

if __name__ == "__main__":
    main()
