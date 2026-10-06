"""Symmetric TP vision: head-parallel attention, column/row-parallel MLP.

Only patches are broadcast; every rank obtains the same aligned embeddings.
No vision work or collective is performed for text requests.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open

from model.vision import ViT, Aligner

VISION_OVERRIDE = '/data/models/DeepSeek-V4.1-Flash-w4a8-Ascend/vision-00001-of-00001.safetensors'


class RowLinear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        out = F.linear(x, self.weight)
        dist.all_reduce(out)
        return out if self.bias is None else out + self.bias


def load_vision(root, device, world=8, rank=0):
    root = Path(root)
    args = SimpleNamespace(**json.loads((root/'inference/config.json').read_text()))
    with torch.device('meta'):
        vision, aligner = ViT(args), Aligner(args)
    mapping = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    # The quantised pack ships its own vision shard; its aligner/delimiters differ from
    # the bf16 release and must match the LLM weights actually loaded by the engine.
    override = Path(VISION_OVERRIDE) if VISION_OVERRIDE else None
    okeys = set()
    if override is not None and override.exists():
        with safe_open(override, framework='pt', device='cpu') as f:
            okeys = set(f.keys())
    def get(name):
        if name in okeys:
            with safe_open(override, framework='pt', device='cpu') as f:
                return f.get_tensor(name)
        with safe_open(root/mapping[name], framework='pt', device='cpu') as f:
            return f.get_tensor(name)
    def bind(module, prefix):
        # Load each module directly in its final layout; never allocate a full tower on GPU.
        for name, child in list(module.named_modules()):
            if not isinstance(child, nn.Linear):
                continue
            key = prefix + name
            w = get(key+'.weight')
            b = get(key+'.bias') if child.bias is not None else None
            col = name.endswith(('attn.wqkv', 'mlp.w1')) or (prefix == 'aligner.' and name == 'w1')
            row = name.endswith(('attn.wo', 'mlp.w2')) or (prefix == 'aligner.' and name == 'w2')
            if world > 1 and col:
                groups = 3 if name.endswith('wqkv') else 2 if name.endswith('mlp.w1') else 1
                w = torch.cat([t.chunk(world, 0)[rank] for t in w.chunk(groups, 0)], 0)
                if b is not None:
                    b = torch.cat([t.chunk(world, 0)[rank] for t in b.chunk(groups, 0)], 0)
            elif world > 1 and row:
                w = w.chunk(world, 1)[rank]
            w = w.contiguous().to(device=device, dtype=torch.bfloat16)
            b = None if b is None else b.to(device=device, dtype=torch.bfloat16)
            if world > 1 and row:
                parent, _, leaf = name.rpartition('.')
                setattr(module.get_submodule(parent), leaf, RowLinear(w,b))
            else:
                child.weight = nn.Parameter(w, requires_grad=False)
                child.bias = None if b is None else nn.Parameter(b, requires_grad=False)
        for name, child in module.named_modules():
            if child.__class__.__name__ == 'RMSNorm':
                child.weight = nn.Parameter(get(prefix+name+'.weight').to(device=device, dtype=torch.float32), requires_grad=False)
    bind(vision, 'vision.')
    bind(aligner, 'aligner.')
    if world > 1:
        for block in vision.blocks:
            block.attn.n_heads //= world
    delimiters = {k:get('image_'+k).to(device=device, dtype=torch.bfloat16)
                  for k in ('start','end','newline')}
    return VisionRuntime(vision.eval(), aligner.eval(), delimiters, device)


class VisionRuntime:
    def __init__(self, vision, aligner, delimiters, device):
        self.vision, self.aligner, self.delimiters, self.device = vision, aligner, delimiters, device

    @torch.inference_mode()
    def encode(self, patches, image, out):
        nh, nw = image['grid']
        features = self.aligner(self.vision(patches, nh, nw), nh, nw)
        h, w = image['llm_grid']
        if features.shape[0] != h*w:
            raise ValueError('aligner output grid mismatch')
        span = out[:image['length']]
        if span.shape != (image['length'], features.shape[-1]):
            raise RuntimeError('vision output exceeds its reserved slot')
        span[0], span[-1] = self.delimiters['start'], self.delimiters['end']
        rows = span[1:-1].view(h,w+1,-1)
        rows[:,:w] = features.view(h,w,-1)
        rows[:,w] = self.delimiters['newline']
        return span

    def allocate_buffers(self, slots):
        from server.image_input import MAX_VISION_TOKENS
        self.max_patches = (1024 - 3) * 9
        self.meta = torch.empty(8, dtype=torch.int64)
        self.patch_host = torch.empty((self.max_patches, 588), dtype=torch.float32)
        self.patch_device = torch.empty_like(self.patch_host, device=self.device, dtype=torch.bfloat16)
        dim = self.delimiters['start'].numel()
        self.outputs = torch.empty((slots, MAX_VISION_TOKENS, dim), device=self.device,
                                   dtype=torch.bfloat16)
        return self

    @torch.inference_mode()
    def warmup(self):
        self.patch_device.zero_()
        for gh, gw in ((3, 3063), (93, 93)):
            lh, lw = (gh+2)//3, (gw+2)//3
            im = dict(grid=(gh, gw), llm_grid=(lh, lw), length=lh*(lw+1)+2)
            self.encode(self.patch_device[:gh*gw], im, self.outputs[0])
        torch.npu.synchronize()
