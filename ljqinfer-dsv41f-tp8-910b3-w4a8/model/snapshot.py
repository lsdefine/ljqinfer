"""Contiguous rank-blob snapshots: build, pinned-ring H2D, tmpfs residency hold.

Trusted immutable prepared cache: blob hashes are checked during preparation,
not rescanned during hot start.
"""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .checkpoint_weights import LAYOUT_ABI, atomic_json, digest, tensor_spec
from .wcache import WeightCache, identity, require_memory


FORMAT = 'v41-v4-contiguous-256-v1'


def plan(manifest):
    if not manifest['complete'] or manifest['pack_abi'] != LAYOUT_ABI:
        raise ValueError('incomplete or incompatible source manifest')
    offset, refs = 0, {}
    for unit in manifest['units']:
        for name, spec in unit['tensors'].items():
            if name in refs:
                raise ValueError('duplicate tensor: ' + name)
            offset = (offset + 255) // 256 * 256
            refs[name] = dict(spec, offset=offset)
            offset += spec['nbytes']
    if not refs or sum(s['nbytes'] for s in refs.values()) != manifest['payload_bytes']:
        raise ValueError('snapshot payload mismatch')
    return (offset + 255) // 256 * 256, refs


def build(cache_root, snapshot_root, rank):
    if rank not in range(8):
        raise ValueError('invalid rank')
    cache_root, root = Path(cache_root), Path(snapshot_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / f'rank{rank}.json'
    source = json.loads(manifest_path.read_text())
    if source['rank'] != rank:
        raise ValueError('wrong source rank')
    size, refs = plan(source)
    blob = root / f'rank{rank}.bin'
    published = root / f'rank{rank}.json'
    if blob.exists() or published.exists():
        raise FileExistsError('snapshot already exists; do not overwrite live mappings')
    tmp = blob.with_suffix('.building')
    with tmp.open('xb') as f:
        os.posix_fallocate(f.fileno(), 0, size)
    mm = np.memmap(tmp, dtype=np.uint8, mode='r+', shape=(size,))
    cache = WeightCache(cache_root)
    for i, unit in enumerate(source['units']):
        key = identity(source=source['source_sha256'], unit=unit['unit'],
                       rank=rank, pack_abi=LAYOUT_ABI)
        if key != unit['key'] or cache.path(key).name != unit['file']:
            raise ValueError('source unit identity mismatch')
        if digest(cache.path(key)) != unit['sha256']:
            raise ValueError('source unit hash mismatch')
        tensors = cache.load(key)
        if tensors is None or set(tensors) != set(unit['tensors']):
            raise ValueError('missing source tensors')
        for name, tensor in tensors.items():
            if tensor_spec(tensor) != unit['tensors'][name]:
                raise ValueError('tensor spec mismatch: ' + name)
            ref = refs[name]
            raw = tensor.reshape(-1).view(torch.uint8).numpy()
            mm[ref['offset']:ref['offset'] + ref['nbytes']] = raw
        del tensors
        if i % 100 == 0:
            print('SNAPSHOT', rank, i, flush=True)
    mm.flush()
    del mm
    checksum = digest(tmp)
    os.chmod(tmp, 0o444)
    tmp.rename(blob)
    result = dict(format=FORMAT, complete=True, rank=rank, pack_abi=LAYOUT_ABI,
                  source_sha256=source['source_sha256'],
                  source_manifest_sha256=digest(manifest_path),
                  blob_bytes=size, payload_bytes=source['payload_bytes'],
                  sha256=checksum, tensors=refs)
    atomic_json(published, result)
    print('SNAPSHOT_DONE', rank, size, flush=True)
    return result


def staged(mm, rank_off, rank_size, r: int):
    """Bounded pinned ring H2D: parallel host copy overlapped with device copy.

    Pageable copy leaves both the host read and the transfer single threaded, so
    eight concurrent ranks fall far below memory and link bandwidth. A small
    pinned ring keeps host bytes moving on several cores while the previous
    chunk crosses the link on a side stream.
    """
    torch.npu.set_device(r)
    size = rank_size[r]
    src_ptr = int(mm.ctypes.data) + rank_off[r]
    chunk = int(os.environ.get('V41_H2D_CHUNK_MIB', '512')) * 1024**2
    workers = int(os.environ.get('V41_H2D_THREADS', '16'))
    piece = int(os.environ.get('V41_H2D_PIECE_MIB', '32')) * 1024**2
    device = torch.empty(size, dtype=torch.uint8, device=f'npu:{r}')
    ring = [torch.empty(min(chunk, size), dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    stream = torch.npu.Stream(device=f'npu:{r}')
    events = [torch.npu.Event(), torch.npu.Event()]
    pool = ThreadPoolExecutor(workers)

    def fill(slot, offset, nbytes):
        def move(start):
            length = min(piece, nbytes - start)
            ctypes.memmove(int(ring[slot].data_ptr()) + start, src_ptr + offset + start, length)
        list(pool.map(move, range(0, nbytes, piece)))

    with torch.npu.stream(stream):
        offset, slot, pending = 0, 0, 0
        while offset < size:
            nbytes = min(chunk, size - offset)
            if pending >= 2:
                events[slot].synchronize()
            fill(slot, offset, nbytes)
            device[offset:offset + nbytes].copy_(ring[slot][:nbytes], non_blocking=True)
            events[slot].record(stream)
            offset += nbytes
            slot = 1 - slot
            pending += 1
    torch.npu.synchronize(r)
    pool.shutdown()
    return device


def load_snapshot(weights, root, *, verify_copy=False):
    root = require_memory(root)
    require_memory(weights.cache_root)
    rank = weights.rank
    meta = json.loads((root / f'rank{rank}.json').read_text())
    size, refs = plan(weights.manifest)
    if (meta['format'] != FORMAT or not meta['complete']
            or meta['rank'] != rank or meta['blob_bytes'] != size
            or meta['pack_abi'] != weights.manifest['pack_abi']
            or meta['source_sha256'] != weights.manifest['source_sha256']
            or meta['source_manifest_sha256'] != digest(weights.cache_root / f'rank{rank}.json')
            or meta['tensors'] != refs):
        raise ValueError('incompatible snapshot manifest')
    blob = root / f'rank{rank}.bin'
    if blob.stat().st_size != size or blob.stat().st_mode & 0o222:
        raise ValueError('snapshot must be complete and read-only')
    if torch.device(weights.device).type != 'npu':
        raise ValueError('bulk snapshot requires NPU')
    mm = np.memmap(blob, dtype=np.uint8, mode='r', shape=(size,))
    device = staged(mm, {rank: 0}, {rank: size}, rank)
    if verify_copy:
        for offset in range(0, size, 64 * 1024**2):
            end = min(size, offset + 64 * 1024**2)
            if not np.array_equal(device[offset:end].cpu().numpy(), mm[offset:end]):
                raise ValueError('snapshot H2D byte mismatch')
    data = {}
    for name, ref in refs.items():
        dtype = getattr(torch, ref['dtype'].removeprefix('torch.'))
        begin, length = ref['offset'], ref['nbytes']
        data[name] = device[begin:begin + length].view(dtype).reshape(ref['shape'])
    weights.data = data
    weights.specs = {n: {k: v for k, v in s.items() if k != 'offset'} for n, s in refs.items()}
    weights._snapshot_storage = device
    if weights.nbytes != weights.manifest['payload_bytes']:
        raise ValueError('snapshot resident payload mismatch')
    return weights


def hold(root, *, host_only=False):
    root = Path(root)
    if host_only:
        host = json.loads((root / 'host.json').read_text())
        if not host['complete']:
            raise ValueError('incomplete host cache')
        entries = [(name, root / spec['file'], dict(
            complete=True, rank=name, blob_bytes=spec['nbytes'], sha256=spec['sha256']))
            for name, spec in host['tensors'].items()]
    else:
        entries = [(rank, root / f'rank{rank}.bin',
                    json.loads((root / f'rank{rank}.json').read_text()))
                   for rank in range(8)]
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mlock.restype = ctypes.c_int
    mappings = []
    reports = []
    for rank, path, meta in entries:
        if (not meta['complete'] or meta['rank'] != rank
                or path.stat().st_size != meta['blob_bytes']
                or path.stat().st_mode & 0o222):
            raise ValueError('incomplete or writable snapshot')
        if digest(path) != meta['sha256']:
            raise ValueError('snapshot digest mismatch')
        mm = np.memmap(path, dtype=np.uint8, mode='r')
        start = time.monotonic()
        if libc.mlock(int(mm.ctypes.data), int(mm.nbytes)) != 0:
            raise OSError(ctypes.get_errno(), 'snapshot mlock failed')
        touch = sum(int(mm[i]) for i in range(0, mm.nbytes, 2 * 1024**2))
        mappings.append(mm)
        reports.append(dict(rank=rank, bytes=int(mm.nbytes), locked=True,
                            seconds=time.monotonic() - start, touch=touch))
        print('LOCKED', rank, mm.nbytes, flush=True)
    while True:
        atomic_json(root / 'hold_status.json',
                    dict(complete=True, pid=os.getpid(), heartbeat=time.time(), ranks=reports))
        time.sleep(10)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--hold', action='store_true')
    parser.add_argument('--host-only', action='store_true')
    args = parser.parse_args()
    if args.hold:
        hold(args.snapshot, host_only=args.host_only)
    else:
        build(args.cache, args.snapshot, args.rank)
