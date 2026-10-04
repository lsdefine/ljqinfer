"""Fixed Q6 CSA storage and direct source execution.

Caller owns weights, scratch and canonical history. Source retains CANN
descriptors and VendorSelect storage until captured graphs are destroyed.
run(workspace, rope_table, attention_library) does not commit Past.
"""
import math
from types import MappingProxyType


def source_resource_specs(*, batch, ratio, score_width, reindex=False):
    """Host-only resource declaration; shapes are rank-local TP8, Q=6.

    Each entry is (shape, torch dtype name). All outputs are separate contiguous
    32-byte-aligned allocations. Score width is a STARTUP constant, a multiple
    of eight covering the admitted context, never recomputed per window.
    Canonical history/table/carry and canonical FP32 weights are borrowed apart
    from this scratch declaration. Reindex borrows the full source's pending
    rows; the builder must retain these until its last reuse consumer.
    """
    if (type(batch) is not int or not 1 <= batch <= 4 or ratio not in (1, 2)
            or type(score_width) is not int or not 0 < score_width <= 1048576
            or score_width % 8):
        raise ValueError('source requires B1..B4, ratio 1/2 and fixed aligned score width')
    r = batch * 6
    specs = dict(hidden=((r,5120),'bfloat16'),
        slots=((batch,),'int64'), start=((batch,),'int64'), active=((batch,),'int64'),
        token_freqs=((r,32,2),'float32'), group_freqs=((r,32,2),'float32'),
        qr=((r,1280),'bfloat16'), qn=((r,1280),'bfloat16'),
        iq_raw=((r,512),'bfloat16'), iq=((r,4,128),'bfloat16'),
        head_weight=((r,4),'bfloat16'),
        local_scores=((batch,6,score_width),'float32'),
        scores=((batch,6,score_width),'float32'),
        selected=((batch,6,512),'int64'))
    if not reindex:
        specs.update(values=((r,512),'float32' if ratio == 2 else 'bfloat16'),
            pooled=((r,512),'bfloat16'), latent=((r,512),'bfloat16'),
            ik_raw=((r,128),'bfloat16'), ik_normed=((r,128),'bfloat16'),
            pending=((batch,6,512),'bfloat16'),
            index_pending=((batch,6,128),'bfloat16'))
        if ratio == 2:
            specs.update(hidden_f32=((r,5120),'float32'), gates=((r,512),'float32'))
    return specs


def _check_tensor(value, shape, dtype, device, name):
    import torch
    if (tuple(value.shape) != tuple(shape) or value.dtype != getattr(torch, dtype)
            or value.device != device or device.type != 'npu'
            or not value.is_contiguous() or value.data_ptr() % 32):
        raise ValueError('invalid source storage: ' + name)
    return value


def _overlap(a, b):
    return (a.numel() and b.numel()
        and a.data_ptr() < b.data_ptr() + b.numel()*b.element_size()
        and b.data_ptr() < a.data_ptr() + a.numel()*a.element_size())


class Source:
    """One full/reindex layer, fixed B and score capacity, current-stream calls.

    weights is the actual DeviceWeights owner (not an FP8 adapter). tensors
    follows source_resource_specs plus index_bank[P,R,128], table[S,M]; ratio2
    full sources additionally borrow carry_values/carry_scores[S,4,512].
    prepared binds canonical FP32 storage: q_norm, and for full sources
    c_norm/i_norm; ratio2 also c_wkv/c_wgate. Each binding must have the same
    address as its DeviceWeights tensor, not a cast or detached copy. Norm
    kernels read FP32 gamma with BF16 activations; ratio2 GEMMs consume FP32
    hidden/weights and emit FP32 values/gates. Ratio1 wkv stays BF16.
    Without w8_prepared, prepared must additionally supply persistent BF16
    wqa_scale[1280,1] and wiq_scale[512,1]. The caller retains all owners;
    this plan never converts weights.

    Projection descriptors are owned here; all device storage is caller-owned.
    Serialized plans may share GEMM workspace, but not live source outputs.
    Destroy graphs and drain all work BEFORE close(), then release buffers.
    """
    def __init__(self, weights, layer, tensors, prepared, *, batch, ratio,
                 score_width, norm_library, gemm_library, eps, borrowed=None,
                 w8_prepared=None, candidate_rows=None, produce_candidates=False, parallel=None):
        specs = source_resource_specs(batch=batch, ratio=ratio,
            score_width=score_width, reindex=borrowed is not None)
        if (type(layer) is not int or layer < 0 or not math.isfinite(eps)
                or not 0 < eps <= 3.402823466e38):
            raise ValueError('invalid source layer or normalization epsilon')
        self.closed, self.operators = False, {}
        self.weights, self.layer, self.borrowed = weights, layer, borrowed
        self.batch, self.ratio, self.score_width = batch, ratio, score_width
        self.index_scale = 128 ** -0.5 * 32 ** -0.5
        self.device = tensors['hidden'].device
        self.candidate_rows, self.produce_candidates = candidate_rows, produce_candidates
        self.candidate_scoring = borrowed is not None
        if self.candidate_scoring or produce_candidates:
            _check_tensor(candidate_rows, (batch,6,16384), 'int64', self.device, 'candidate_rows')
            if ratio != 1 or (self.candidate_scoring and score_width != 16384):
                raise ValueError('HSI requires ratio1 and fixed 16384 candidate scores')
        self.norm_library, self.gemm_library = norm_library, gemm_library
        self.tensors = MappingProxyType(dict(tensors))
        self.prepared = MappingProxyType(dict(prepared))
        t, p, device = self.tensors, self.prepared, self.device
        if device != weights.device or weights.rank != device.index:
            raise ValueError('source and DeviceWeights rank/device differ')
        for name, (shape, dtype) in specs.items():
            _check_tensor(t[name], shape, dtype, device, name)
        bank, table = t['index_bank'], t['table']
        if (bank.ndim != 3 or bank.shape[2] != 128 or min(bank.shape) < 1
                or table.ndim != 2 or min(table.shape) < 1
                or table.shape[0] < batch
                or (not self.candidate_scoring and score_width > table.shape[1]*bank.shape[1])):
            raise ValueError('invalid source index history geometry/capacity')
        _check_tensor(bank, bank.shape, 'bfloat16', device, 'index_bank')
        _check_tensor(table, table.shape, 'int64', device, 'table')
        self.geometry = (table.shape[0], bank.shape[0], table.shape[1], bank.shape[1])
        prefix = f'layers.{layer}.attn.'
        retained = {}
        def weight(name, shape, dtype):
            key = prefix + name
            value = _check_tensor(weights.data[key], shape, dtype, device, key)
            retained[key] = value
            return value
        def canonical_f32(name, key, shape):
            value = weight(name, shape, 'float32')
            bound = _check_tensor(p[key], shape, 'float32', device, key)
            if bound.data_ptr() != value.data_ptr():
                raise ValueError('source FP32 binding must borrow canonical storage: ' + key)
            return value
        # Canonical W8 scale is [N,1], not the old UINT8 block-scale grid.
        wqa = weight('wq_a.weight', (1280,5120), 'int8')
        weight('wq_a.scale', (1280,1), 'float32')
        wiq = weight('indexer.wq_b.weight', (512,1280), 'int8')
        weight('indexer.wq_b.scale', (512,1), 'float32')
        wh = weight('indexer.weights_proj.weight', (4,5120), 'bfloat16')
        canonical_f32('q_norm.weight', 'q_norm', (1280,))
        if borrowed is not None:
            if (not isinstance(borrowed, Source) or borrowed.closed
                    or borrowed.borrowed is not None or borrowed.weights is not weights
                    or borrowed.layer >= layer or borrowed.batch != batch
                    or borrowed.ratio != ratio or borrowed.device != device
                    or borrowed.geometry != self.geometry):
                raise ValueError('reindex requires an earlier canonical full source')
            for key in ('index_bank', 'table', 'slots', 'start', 'active'):
                if t[key].data_ptr() != borrowed.tensors[key].data_ptr():
                    raise ValueError('reindex must borrow canonical source ' + key)
            self.pending, self.index_pending = borrowed.pending, borrowed.index_pending
            self.projected_values = self.projected_gates = None
        else:
            canonical_f32('compressor.norm.weight', 'c_norm', (512,))
            canonical_f32('indexer.k_norm.weight', 'i_norm', (128,))
            wk = weight('indexer.wk.weight', (128,512), 'bfloat16')
            if ratio == 2:
                ckv = canonical_f32('compressor.wkv.weight', 'c_wkv', (512,5120))
                cg = canonical_f32('compressor.wgate.weight', 'c_wgate', (512,5120))
                for name in ('carry_values', 'carry_scores'):
                    _check_tensor(t[name], (table.shape[0],4,512), 'float32', device, name)
            else:
                ckv = weight('compressor.wkv.weight', (512,5120), 'bfloat16')
            self.pending, self.index_pending = t['pending'], t['index_pending']
            self.projected_values = t['values']
            self.projected_gates = t['gates'] if ratio == 2 else None
        self.selected, self.query_latent = t['selected'], t['qn']
        self.token_freqs = t['token_freqs']
        self.canonical_weights = MappingProxyType(retained)
        # Fail at build on accidental scratch/history/weight aliasing. Pending
        # source rows must survive all downstream reindex/reuse consumers.
        scratch = [t[k] for k in specs]
        cached = {}
        if w8_prepared is not None:
            for key, name, shape in (('wqa', 'wq_a', (1280,5120)),
                                     ('wiq', 'indexer.wq_b', (512,1280))):
                cached[key] = _check_tensor(w8_prepared[prefix+name], shape,
                                           'bfloat16', device, prefix+name)
        readonly = [bank, table, *retained.values(), *p.values(), *cached.values()]
        if candidate_rows is not None:
            readonly.append(candidate_rows)
        if borrowed is not None:
            readonly += [self.pending, self.index_pending]
        elif ratio == 2:
            readonly += [t['carry_values'], t['carry_scores']]
        for i, a in enumerate(scratch):
            if any(_overlap(a, b) for b in scratch[:i] + readonly):
                raise ValueError('source scratch aliases live storage')
        # Import only after validation. CPU contract tests need no CANN loader.
        from ops.decode.gemm import Matmul, W8Matmul
        try:
            self.operators['wqa'] = (Matmul(t['hidden'], cached['wqa'], t['qr']) if cached
                else W8Matmul(t['hidden'], wqa, p['wqa_scale'], t['qr']))
            self.operators['wiq'] = (Matmul(t['qn'], cached['wiq'], t['iq_raw']) if cached
                else W8Matmul(t['qn'], wiq, p['wiq_scale'], t['iq_raw']))
            self.operators['head'] = Matmul(t['hidden'], wh, t['head_weight'])
            if borrowed is None:
                source = t['hidden_f32'] if ratio == 2 else t['hidden']
                self.operators['values'] = Matmul(source, ckv, t['values'])
                if ratio == 2:
                    self.operators['gates'] = Matmul(source, cg, t['gates'])
                self.operators['index_key'] = Matmul(t['latent'], wk, t['ik_raw'])
            self.workspace_bytes = max(plan.workspace_bytes for plan in self.operators.values())
            self.eps = eps
            from ops.decode.vendor_select import VendorSelect
            self.vendor = (VendorSelect(bank=t['index_bank'], table=t['table'],
                slots=t['slots'], start=t['start'], pending=self.index_pending,
                batch=batch, hccl_library=parallel.lib, hccl_comm=parallel.comm.value,
                device=self.device) if borrowed is None else borrowed.vendor)
        except BaseException:
            self.close()
            raise

    def run(self, workspace, rope_table, attention_library):
        from model.decode_ops import native
        if self.closed:
            raise RuntimeError('source is closed')
        t = dict(self.tensors, **self.prepared, rope_table=rope_table,
                 index_pending=self.index_pending, pending=self.pending,
                 candidate_rows=self.candidate_rows)
        b, ratio = self.batch, self.ratio
        nslots, pages, maxpages, rpp = self.geometry
        native(attention_library, 'dec_source_prepare',
               tuple(t.get(n) for n in
                     'hidden rope_table slots start active token_freqs group_freqs hidden_f32'.split()),
               b,ratio,nslots,len(rope_table),int(ratio==2 and self.borrowed is None))
        if self.borrowed is None:
            self.operators['values'](workspace)
            if ratio == 2:
                self.operators['gates'](workspace)
            native(attention_library, 'dec_source_pool',
                   tuple(t.get(n) for n in
                         'values gates carry_values carry_scores slots start active pooled'.split()),
                   b,ratio,nslots)
            native(self.norm_library, 'dec_rms_norm_f32', (t['pooled'],t['c_norm'],t['latent']),b,512,self.eps)
            self.operators['index_key'](workspace)
            native(self.norm_library, 'dec_rms_norm_f32',(t['ik_raw'],t['i_norm'],t['ik_normed']),b,128,self.eps)
            native(self.norm_library, 'dec_rope',(t['ik_normed'],t['group_freqs'],self.index_pending),b,1,128,0)
            native(attention_library, 'dec_source_fp4',(self.index_pending,self.index_pending),b*6,128,32,0)
            native(self.norm_library, 'dec_rope',(t['latent'],t['group_freqs'],self.pending),b,1,512,0)
            native(attention_library, 'dec_source_fp4',(self.pending,self.pending),b*6,512,16,1)
        self.operators['wqa'](workspace)
        native(self.norm_library, 'dec_rms_norm_f32',(t['qr'],t['q_norm'],t['qn']),b,1280,self.eps)
        self.operators['wiq'](workspace)
        native(self.norm_library, 'dec_rope',(t['iq_raw'],t['token_freqs'],t['iq']),b,4,128,0)
        native(attention_library, 'dec_source_fp4',(t['iq'],t['iq']),b*24,128,32,0)
        self.operators['head'](workspace)
        if self.borrowed is None:
            self.vendor.prepare()
        self.vendor.dispatch(t['iq'], t['head_weight'], self.selected)
        return self.selected

    def close(self):
        if not self.closed:
            if self.borrowed is None and hasattr(self, 'vendor'):
                self.vendor.close()
            for op in reversed(tuple(self.operators.values())):
                op.close()
            self.closed = True
