"""Exact positive-temperature, p=.95 nucleus selection; stable token-ID ties, read-only logits.
Caller supplies independent FP64 uniforms shared across TP ranks.
Sorted full vocabulary, FP64 block totals, only two local prefix scans.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _totals(X, SUMS, TEMP, WIDTH: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    r, p = tl.program_id(0), tl.program_id(1)
    i = p * BLOCK + tl.arange(0, BLOCK)
    m = tl.load(X+r*WIDTH).to(tl.float64)
    v = tl.load(X+r*WIDTH+i, i<WIDTH, other=-float('inf')).to(tl.float64)
    t = tl.load(TEMP+r)
    w = libdevice.exp((v-m)/t)
    tl.store(SUMS+r*PARTS+p, tl.sum(w, 0))


@triton.jit
def _choose(X, IDS, SUMS, U, OUT, TEMP, WIDTH: tl.constexpr, PARTS: tl.constexpr,
            PB: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    t = tl.load(TEMP+r)
    p = tl.arange(0, PB)
    sums = tl.load(SUMS+r*PARTS+p, p<PARTS, other=0.)
    c = tl.cumsum(sums, 0)
    total = tl.sum(sums, 0)
    cutoff = total * tl.full((), .95, tl.float64)
    bp = tl.min(tl.where((p<PARTS) & (c>=cutoff), p, PARTS), 0)
    bp = tl.minimum(bp, PARTS-1)
    before = tl.sum(tl.where(p<bp, sums, 0.), 0)
    j = tl.arange(0, BLOCK)
    i = bp*BLOCK+j
    m = tl.load(X+r*WIDTH).to(tl.float64)
    x = tl.load(X+r*WIDTH+i, i<WIDTH, other=-float('inf')).to(tl.float64)
    local = tl.cumsum(libdevice.exp((x-m)/t), 0)
    boundary = tl.min(tl.where((i<WIDTH) & (before+local>=cutoff), j, BLOCK),0)
    boundary = tl.minimum(boundary, BLOCK-1)
    mass = before + tl.sum(tl.where(j==boundary, local, 0.),0)
    u = tl.load(U+r)
    target = u * mass
    sp = tl.min(tl.where((p<PARTS) & (c>target), p, PARTS),0)
    sp = tl.minimum(sp, bp)
    start = tl.sum(tl.where(p<sp, sums, 0.),0)
    i = sp*BLOCK+j
    x = tl.load(X+r*WIDTH+i, i<WIDTH, other=-float('inf')).to(tl.float64)
    local = tl.cumsum(libdevice.exp((x-m)/t),0)
    idx = tl.min(tl.where((i<WIDTH) & (start+local>target),i,WIDTH),0)
    idx = tl.minimum(idx, bp*BLOCK+boundary)
    token = tl.load(IDS+r*WIDTH+idx)
    good = (total>0.) & (total<float('inf')) & (u>=0.) & (u<1.)
    tl.store(OUT+r, tl.where(good,token,-1))


class NucleusSampler:
    """Preallocated, graph-capturable, no host decisions or hidden RNG."""
    def __init__(self, rows, width, device):
        self.rows, self.width = rows, width
        self.block = 1024
        self.parts = triton.cdiv(width, self.block)
        self.values = torch.empty((rows,width),device=device,dtype=torch.float32)
        self.ids = torch.empty((rows,width),device=device,dtype=torch.int64)
        self.sums = torch.empty((rows,self.parts),device=device,dtype=torch.float64)
        self.out = torch.empty(rows,device=device,dtype=torch.int64)
        self.temperatures = torch.ones(rows,device=device,dtype=torch.float64)

    @torch.inference_mode()
    def run(self, logits, uniforms):
        if logits.shape != self.values.shape or logits.dtype != torch.float32:
            raise ValueError('sampler requires [rows,vocab] float32 logits')
        if uniforms.shape != (self.rows,) or uniforms.dtype != torch.float64:
            raise ValueError('sampler requires one float64 uniform per row')
        if not logits.is_contiguous() or not uniforms.is_contiguous():
            raise ValueError('sampler inputs must be contiguous')
        torch.sort(logits,dim=-1,descending=True,stable=True,out=(self.values,self.ids))
        _totals[(self.rows,self.parts)](self.values,self.sums,self.temperatures,self.width,self.parts,
                                       self.block,num_warps=4)
        _choose[(self.rows,)](self.values,self.ids,self.sums,uniforms,self.out,self.temperatures,
                             self.width,self.parts,triton.next_power_of_2(self.parts),
                             self.block,num_warps=4, enable_fp_fusion=False)
        return self.out
