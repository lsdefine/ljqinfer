"""Audited checkpoint inventory, rank slicing and packed weight stores.

Source identity is the full SHA256 of the released checkpoint, never a path.
Rank-local packed units are snapshotted through the weight cache, so a warm
start maps prepared bytes instead of re-slicing the checkpoint.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import torch
from safetensors import safe_open
from .engram_weights import HostEngram
from .weights import LAYOUT_ABI, placement, prepare_rank, WORLD
from .wcache import WeightCache, identity


DTYPE_BYTES = {'BF16': 2, 'F32': 4, 'I8': 1}
TORCH_DTYPE = {'BF16': torch.bfloat16, 'F32': torch.float32, 'I8': torch.int8}
INDEX_NAME = 'quant_model_weights.safetensors.index.json'


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(path)


def inventory(root):
    root = Path(root)
    index = json.loads((root / INDEX_NAME).read_text())
    entries, files = {}, []
    for filename in sorted(set(index['weight_map'].values())):
        path = root / filename
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            if length > 64 * 1024 * 1024:
                raise ValueError('unreasonable safetensors header')
            header = json.loads(f.read(length))
        stat = path.stat()
        end = 0
        for name, entry in header.items():
            if name == '__metadata__':
                continue
            lo, hi = entry['data_offsets']
            size = math.prod(entry['shape']) * DTYPE_BYTES[entry['dtype']]
            if hi - lo != size or lo < 0 or hi + 8 + length > stat.st_size:
                raise ValueError('invalid tensor extent: ' + name)
            end = max(end, hi)
            owner = index['weight_map'].get(name)
            if owner != filename:
                # The index is authoritative. A sidecar shard may carry a [1,1]
                # stub for a tensor whose payload lives in its own dedicated file;
                # anything larger means the index and the shards truly disagree.
                if owner is None or math.prod(entry['shape']) != 1:
                    raise ValueError('index mismatch: ' + name)
                continue
            if name in entries:
                raise ValueError('duplicate tensor: ' + name)
            placement(name)  # Unknown weights must never silently replicate.
            entries[name] = dict(entry, file=filename, nbytes=size)
        if end + 8 + length != stat.st_size:
            raise ValueError('truncated/trailing shard: ' + filename)
        files.append(dict(name=filename, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    if set(entries) != set(index['weight_map']):
        raise ValueError('incomplete checkpoint')
    return entries, files


def attest(root, output, workers=4):
    """Offline full-content hashing. Detect source changes during the read."""
    root = Path(root)
    entries, files = inventory(root)
    def hash_one(item):
        path = root / item['name']
        sha = digest(path)
        st = path.stat()
        if (st.st_size, st.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise RuntimeError('checkpoint changed during attestation')
        print('HASHED', item['name'], flush=True)
        return dict(item, sha256=sha)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        files = list(executor.map(hash_one, files))
    content = {x['name']: x['sha256'] for x in files}
    for name in ['config.json', INDEX_NAME]:
        content[name] = digest(root / name)
    source = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    result = dict(format=1, source_sha256=source, files=files,
                  metadata_sha256={k: v for k, v in content.items() if not k.endswith('.safetensors')},
                  tensor_count=len(entries), total_bytes=sum(x['nbytes'] for x in entries.values()))
    atomic_json(output, result)
    return result


def check_attestation(root, attestation):
    """Fast build-time immutability guard; full hashes are established offline."""
    root = Path(root)
    for item in attestation['files']:
        st = (root / item['name']).stat()
        if (st.st_size, st.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise ValueError('source changed; repeat full attestation: ' + item['name'])
    for name, sha in attestation['metadata_sha256'].items():
        if digest(root / name) != sha:
            raise ValueError('source metadata changed')


class Checkpoint:
    def __init__(self, root, entries):
        self.root, self.entries = Path(root), entries
        self.stack, self.handles = ExitStack(), {}

    def raw(self, name):
        filename = self.entries[name]['file']
        if filename not in self.handles:
            self.handles[filename] = self.stack.enter_context(
                safe_open(self.root / filename, framework='pt', device='cpu'))
        return self.handles[filename].get_tensor(name)

    def close(self):
        self.stack.close()


def rank_accounting(entries, rank):
    if rank not in range(WORLD):
        raise ValueError('rank must be 0..7')
    result = dict(backbone=0, dspark=0, vision=0, host=0)
    for name, entry in entries.items():
        p = placement(name)
        if p.kind == 'dropped' or ('.experts.' in name and name.endswith('.scale_bias')):
            continue
        if p.kind == 'host':
            result['host'] += entry['nbytes']
            continue
        size = entry['nbytes']
        if p.kind == 'tp':
            if entry['shape'][p.axis] % WORLD:
                raise ValueError('nondivisible TP: ' + name)
            size //= WORLD
        category = 'dspark' if name.startswith('mtp.') else (
            'vision' if name.startswith(('vision.', 'aligner.', 'image_')) else 'backbone')
        if '.experts.' in name and name.endswith('.weight'):
            shape = list(entry['shape'])
            if p.kind == 'tp':
                shape[p.axis] //= WORLD
            packed_rows, k = shape
            n = packed_rows * 2
            kp = (k + 63) // 64 * 64
            # Replace source packed-byte storage, then add FP32 HP correction.
            size = n * kp // 2 + n * 4
        result[category] += size
    result['device'] = sum(result[k] for k in ('backbone', 'dspark', 'vision'))
    return result


def units(entries, rank):
    groups = {}
    for name in sorted(entries):
        p = placement(name)
        if p.kind == 'host':
            continue
        unit = (name.split('.experts.')[0] + '.experts' if '.experts.' in name
                else name.rsplit('.', 1)[0])
        groups.setdefault(unit, []).append(name)
    return groups


def tensor_spec(t):
    return dict(shape=list(t.shape), dtype=str(t.dtype), nbytes=t.numel() * t.element_size())


def cache_engram_rotation(root, cache_root, manifest):
    """Attach the released auxiliary basis by its own full content identity."""
    root, cache = Path(root), WeightCache(cache_root)
    config = json.loads((root / 'config.json').read_text())['engram_rotation_config']
    if (config.get('value_basis') != 'quarot_global'
            or config.get('key_and_gate_basis') != 'original'
            or config.get('value_projection_rotated') is not True
            or config.get('runtime_delta_rotation') is not False):
        raise ValueError('unsupported Engram basis contract')
    path = root / 'optional/quarot.safetensors'
    sha = digest(path)
    with safe_open(path, framework='pt', device='cpu') as f:
        rotation = f.get_tensor('global_rotation').float()
    block = rotation[:32, :32].contiguous()
    if (rotation.shape != (5120, 5120) or not torch.isfinite(rotation).all()
            or not torch.equal(rotation, torch.block_diag(*([block] * 160)))):
        raise ValueError('Engram rotation is not a repeated 32-channel basis')
    if digest(path) != sha:
        raise ValueError('rotation changed during preparation')
    if any('engram.rotation' in u['tensors'] for u in manifest['units']):
        raise ValueError('rotation already cached')
    unit = 'engram.rotation.' + sha
    key = identity(source=manifest['source_sha256'], unit=unit,
                   rank=manifest['rank'], pack_abi=LAYOUT_ABI)
    tensors = cache.get_or_build(key, lambda: {'engram.rotation': block})
    if set(tensors) != {'engram.rotation'} or not torch.equal(tensors['engram.rotation'], block):
        raise ValueError('cached rotation differs from released basis')
    manifest['units'].append(dict(unit=unit, key=key, file=cache.path(key).name,
        sha256=digest(cache.path(key)), auxiliary_sha256=sha,
        tensors={n: tensor_spec(t) for n, t in tensors.items()}))
    manifest['accounting']['backbone'] += block.numel() * block.element_size()
    manifest['accounting']['device'] += block.numel() * block.element_size()


def build_rank(root, cache_root, attestation_path, rank):
    attestation = json.loads(Path(attestation_path).read_text())
    check_attestation(root, attestation)
    entries, _ = inventory(root)
    cache = WeightCache(cache_root)
    source = Checkpoint(root, entries)
    manifest = dict(format=1, rank=rank, pack_abi=LAYOUT_ABI,
                    source_sha256=attestation['source_sha256'], units=[], complete=False,
                    accounting=rank_accounting(entries, rank))
    try:
        for i, (unit, names) in enumerate(units(entries, rank).items()):
            key = identity(source=attestation['source_sha256'], unit=unit,
                           rank=rank, pack_abi=LAYOUT_ABI)
            def build():
                return prepare_rank({n: source.raw(n) for n in names}, rank)
            tensors = cache.get_or_build(key, build)
            record = dict(unit=unit, key=key, file=cache.path(key).name,
                          sha256=digest(cache.path(key)),
                          tensors={n: tensor_spec(t) for n, t in tensors.items()})
            manifest['units'].append(record)
            del tensors
            if i % 50 == 0 or unit.endswith('.experts'):
                print('CACHE', rank, i, unit, flush=True)
    finally:
        source.close()
    cache_engram_rotation(root, cache_root, manifest)
    manifest['payload_bytes'] = sum(t['nbytes'] for u in manifest['units'] for t in u['tensors'].values())
    if manifest['payload_bytes'] != manifest['accounting']['device']:
        raise ValueError('packed payload differs from exact placement budget')
    check_attestation(root, attestation)
    manifest['complete'] = True
    atomic_json(Path(cache_root) / f'rank{rank}.json', manifest)
    return manifest


class DeviceWeights:
    """Cache-hit load performs no checkpoint read, TP slicing or expert stacking."""
    def __init__(self, cache_root, rank, device):
        self.cache_root, self.rank, self.device = Path(cache_root), rank, device
        self.data, self.specs = {}, {}
        self.manifest = json.loads((self.cache_root / f'rank{rank}.json').read_text())
        m = self.manifest
        if (rank not in range(WORLD) or not m['complete'] or m['rank'] != rank
                or m['pack_abi'] != LAYOUT_ABI or not m['units']
                or m['payload_bytes'] != m['accounting']['device']):
            raise ValueError('incomplete/wrong-rank/incompatible weight manifest')

    def load(self, *, verify_hashes=False, verify_copy=False):
        # Hashing 41 GiB costs 43 of the 52 seconds a cold load used to take, and it
        # re-proves what the cache already proves: every unit carries the identity of
        # its source, rank and pack ABI, and tmpfs hands back the bytes it was given.
        # Builders and anyone suspecting a bad cache pass verify_hashes=True.
        if self.data:
            raise RuntimeError('weights already loaded')
        cache = WeightCache(self.cache_root)
        for i, unit in enumerate(self.manifest['units']):
            expected = identity(source=self.manifest['source_sha256'], unit=unit['unit'],
                                rank=self.rank, pack_abi=LAYOUT_ABI)
            if unit['key'] != expected or cache.path(expected).name != unit['file']:
                raise ValueError('cache unit identity mismatch')
            if verify_hashes and digest(self.cache_root / unit['file']) != unit['sha256']:
                raise ValueError('weight cache content corrupt: ' + unit['unit'])
            tensors = cache.load(expected)
            if tensors is None or set(tensors) != set(unit['tensors']):
                raise ValueError('missing/incomplete weight cache unit')
            for name, cpu in tensors.items():
                if name in self.data or tensor_spec(cpu) != unit['tensors'][name]:
                    raise ValueError('duplicate/malformed weight: ' + name)
                dev = cpu.to(self.device)
                if verify_copy:
                    # Whole-buffer bitwise roundtrip, not sparse samples or float equality.
                    if not torch.equal(dev.cpu().view(torch.uint8), cpu.view(torch.uint8)):
                        raise ValueError('H2D byte verification failed: ' + name)
                self.data[name] = dev
                self.specs[name] = unit['tensors'][name]
            del tensors
            if i % 100 == 0:
                print('LOAD', self.rank, i, flush=True)
        if self.nbytes != self.manifest['payload_bytes']:
            raise ValueError('resident weight payload mismatch')
        return self

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.data.values())

    def __getitem__(self, name):
        return self.data[name]


def build_host(root, cache_root, identity_path):
    root, cache = Path(root), Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    identity = json.loads(Path(identity_path).read_text())
    check_attestation(root, identity)
    entries, _ = inventory(root)
    source = Checkpoint(root, entries)
    manifest = dict(source_sha256=identity['source_sha256'], complete=False, tensors={})
    try:
        for name, entry in sorted(entries.items()):
            if placement(name).kind != 'host':
                continue
            filename = identity['source_sha256'] + '.' + name + '.bin'
            path = cache / filename
            # An existing committed manifest is needed to trust an existing file.
            previous = cache / 'host.json'
            old = json.loads(previous.read_text()) if previous.exists() else {}
            record = old.get('tensors', {}).get(name, {})
            if (old.get('source_sha256') == identity['source_sha256']
                    and record.get('file') == filename and path.exists()
                    and record.get('dtype') == entry['dtype']
                    and path.stat().st_size == entry['nbytes'] and digest(path) == record.get('sha256')):
                manifest['tensors'][name] = record
                continue
            tensor = source.raw(name).view(torch.uint8)
            view = memoryview(tensor.numpy()).cast('B')
            sha = hashlib.sha256()
            tmp = path.with_suffix('.building')
            with tmp.open('wb') as output:
                for start in range(0, len(view), 64 * 1024 * 1024):
                    block = view[start:start + 64 * 1024 * 1024]
                    output.write(block)
                    sha.update(block)
                output.flush()
                os.fsync(output.fileno())
            tmp.replace(path)
            manifest['tensors'][name] = dict(file=filename, shape=entry['shape'],
                dtype=entry['dtype'], nbytes=entry['nbytes'], sha256=sha.hexdigest())
            print('HOST_CACHED', name, entry['nbytes'], flush=True)
        check_attestation(root, identity)
        manifest['complete'] = True
        atomic_json(cache / 'host.json', manifest)
    finally:
        source.close()
    return manifest


class HostWeights:
    def __init__(self, cache_root, source_sha256):
        cache = Path(cache_root)
        self.manifest = json.loads((cache / 'host.json').read_text())
        if not self.manifest['complete'] or self.manifest['source_sha256'] != source_sha256:
            raise ValueError('incomplete/incompatible host cache')
        self.raw, self.engrams = {}, {}
        for name, spec in self.manifest['tensors'].items():
            path = cache / spec['file']
            if path.stat().st_size != spec['nbytes']:
                raise ValueError('truncated host cache')
            # MAP_PRIVATE: reads share physical file pages, accidental writes stay local.
            tensor = torch.from_file(str(path), shared=False, size=spec['nbytes'], dtype=torch.uint8)
            self.raw[name] = tensor.view(TORCH_DTYPE[spec['dtype']]).reshape(spec['shape'])
        for name in self.raw:
            if name.endswith('.weight'):
                prefix = name[:-7]
                self.engrams[prefix] = HostEngram(self.raw[name], self.raw[prefix + '.scale'])

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.raw.values())

    def verify_hashes(self, cache_root):
        for spec in self.manifest['tensors'].values():
            if digest(Path(cache_root) / spec['file']) != spec['sha256']:
                raise ValueError('corrupt host cache')
