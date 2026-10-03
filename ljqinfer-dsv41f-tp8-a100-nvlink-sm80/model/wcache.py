"""Unit-granular final-layout snapshots for fast engine restart (Linux).

Source identity must be an immutable checkpoint revision/digest, not a path.
Pack ABI includes kernel layout/architecture when packed bytes depend on them.
No source access, sharding or packing occurs on a hit. Files are mmap-loaded on
CPU; H2D and persistent device allocation remain the model loader's job.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import fcntl
from safetensors import safe_open
from safetensors.torch import save_file
from .weights import LAYOUT_ABI


def identity(*, source, unit, rank, pack_abi=LAYOUT_ABI):
    if not source or not unit or not pack_abi or rank not in (*range(8), 'host'):
        raise ValueError('immutable source, unit, pack ABI and rank required')
    return json.dumps(dict(format=1, source=source, unit=unit, rank=rank,
                           pack_abi=pack_abi), sort_keys=True, separators=(',', ':'))


class WeightCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        return self.root / (hashlib.sha256(key.encode()).hexdigest() + '.safetensors')

    def load(self, key):
        """Return CPU mapped tensors; missing is a miss, corrupt is an error."""
        path = self.path(key)
        if not path.exists():
            return None
        with safe_open(path, framework='pt', device='cpu') as f:
            if (f.metadata() or {}).get('identity') != key:
                raise ValueError('cache identity mismatch')
            return {name: f.get_tensor(name) for name in f.keys()}

    @contextmanager
    def _lock(self, key):
        with self.path(key).with_suffix('.lock').open('a+b') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def get_or_build(self, key, build):
        """Builder returns a complete bounded unit, already in consumption layout.

        One unit per expert bank/projection keeps conversion scratch bounded.
        Builders must not return the whole model. Use host identity for Engram
        storage; rank builders must exclude the full host table.
        """
        hit = self.load(key)
        if hit is not None:
            return hit
        with self._lock(key):
            hit = self.load(key)
            if hit is not None:
                return hit
            tensors = build()
            if not tensors or any(t.device.type != 'cpu' or not t.is_contiguous()
                                  for t in tensors.values()):
                raise ValueError('builder must return nonempty contiguous CPU tensors')
            fd, name = tempfile.mkstemp(prefix='.building-', dir=self.root)
            os.close(fd)
            try:
                save_file(tensors, name, metadata={'identity': key})
                with open(name, 'rb') as f:
                    os.fsync(f.fileno())
                # Verify header and all tensor extents before publication.
                with safe_open(name, framework='pt', device='cpu') as f:
                    for tensor_name in f.keys():
                        f.get_tensor(tensor_name)
                os.replace(name, self.path(key))
                fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            del tensors
        return self.load(key)
