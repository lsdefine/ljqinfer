"""Checkpoint-backed canonical EP8/TP8 weights; CPU-mapped host tables.

Final rank-local layouts are snapshotted through the weight cache, so a warm
engine start maps packed units instead of re-slicing the released checkpoint.
"""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import json
import torch
from safetensors import safe_open
from model.weights import shard, quant_pair, pack_experts
from model.wcache import WeightCache, identity
from model.engram_weights import HostEngram

# Final-layout units live in shared memory: a warm start maps, never repacks.
CACHE_ROOT = '/dev/shm/ljqinfer_dsv41f'


class _RawView:
    """Bounded name->tensor view over the checkpoint for unit builders."""
    def __init__(self, weights):
        self.weights = weights

    def __getitem__(self, name):
        return self.weights.raw(name)


class CheckpointWeights:
    def __init__(self, root, device, rank, cache=CACHE_ROOT):
        if not 0 <= rank < 8:
            raise ValueError('rank must be 0..7')
        self.root, self.device, self.rank = Path(root), device, rank
        text = (self.root/'model.safetensors.index.json').read_text()
        self.index = json.loads(text)['weight_map']
        # Content digest of the released index is an immutable source identity.
        self.source = 'sha256:'+hashlib.sha256(text.encode()).hexdigest()
        self.cache = WeightCache(cache) if cache else None
        self.data, self.handles = {}, {}
        self.frozen = False
        self.stack = ExitStack()

    def raw(self, name):
        filename = self.index[name]
        if filename not in self.handles:
            self.handles[filename] = self.stack.enter_context(
                safe_open(self.root/filename, framework='pt', device='cpu'))
        value = self.handles[filename].get_tensor(name)
        if name.endswith('.scale') and value.dtype == torch.float8_e8m0fnu:
            value = value.view(torch.uint8)
        return value

    def final(self, name):
        """Rank-local consumption layout on CPU: sharded, paired or dequantized."""
        value = self.raw(name)
        if name.endswith('.weight') and value.dtype in (torch.int8, torch.float8_e4m3fn):
            prefix = name[:-7]
            w, s = quant_pair(prefix, value, self.raw(prefix+'.scale'), self.rank)
            if name.endswith('.attn.wo_a.weight'):
                # Released convert.py dequantizes grouped wo_a before execution.
                n, k = w.shape
                decoded = (w.reshape(n//32,32,k//32,32).float()
                           * (s.float()-127).exp2()[:,None,:,None])
                return {name: decoded.reshape(n,k).bfloat16().contiguous()}
            return {name: w, prefix+'.scale': s}
        value = shard(name, value, self.rank)
        if value is None:
            raise ValueError('not a rank-local device weight: '+name)
        return {name: value}

    def __getitem__(self, name):
        if name in self.data:
            return self.data[name]
        if self.frozen:
            raise KeyError('weight not loaded: '+name)
        if name.endswith('.scale'):
            self[name[:-5]+'weight']
            return self.data[name]
        self._place(self.final(name))
        return self.data[name]

    def _place(self, tensors):
        for name, value in tensors.items():
            self.data[name] = value.to(self.device, non_blocking=True)

    def _unit(self, unit, build):
        if self.cache is None:
            return build()
        return self.cache.get_or_build(
            identity(source=self.source, unit=unit, rank=self.rank), build)

    def host_tables(self, layers):
        return {i: HostEngram(self.raw(f'layers.{i}.engram.embed.weight'),
                             self.raw(f'layers.{i}.engram.embed.scale')) for i in layers}

    def dense_names(self, layer):
        """Layer parameters outside the expert bank and the host Engram table."""
        prefix = f'layers.{layer}.'
        return sorted(n for n in self.index
                      if n.startswith(prefix) and '.ffn.experts.' not in n
                      and not n.startswith(prefix+'engram.embed')
                      and not n.endswith('.scale'))

    def load_dense(self, layers=range(40)):
        for layer in layers:
            def build(layer=layer):
                unit = {}
                for name in self.dense_names(layer):
                    unit.update(self.final(name))
                return unit
            self._place(self._unit(f'layers.{layer}.dense', build))

    def load_experts(self, layers=range(40)):
        for layer in layers:
            prefix = f'layers.{layer}.ffn'
            def build(prefix=prefix):
                return pack_experts(_RawView(self), prefix, self.rank)
            bank = self._unit(prefix+'.local_experts', build)
            self._place({prefix+'.local_experts.'+n: t for n, t in bank.items()})
        torch.cuda.synchronize()

    def dspark_names(self, stage):
        """Draft-stage parameters outside the expert bank."""
        prefix = f'mtp.{stage}.'
        return sorted(n for n in self.index
                      if n.startswith(prefix) and '.ffn.experts.' not in n
                      and not n.endswith('.scale'))

    def load_dspark(self, stages=range(3)):
        """The draft head rides the same cached units under mtp.* names."""
        for stage in stages:
            def build(stage=stage):
                unit = {}
                for name in self.dspark_names(stage):
                    unit.update(self.final(name))
                return unit
            self._place(self._unit(f'mtp.{stage}.dense', build))
            prefix = f'mtp.{stage}.ffn'
            def experts(prefix=prefix):
                return pack_experts(_RawView(self), prefix, self.rank)
            bank = self._unit(prefix+'.local_experts', experts)
            self._place({prefix+'.local_experts.'+n: t for n, t in bank.items()})
        torch.cuda.synchronize()

    def close(self):
        self.stack.close()
