"""Fixed-storage decode window leaves. Caller owns buffers and graph lifetime.

meta/result: int64[B,8], tokens/greedy: int64[B,6]. meta columns are
slot/start/active/reserved; result is slot/start/accepted/next/active/error/0/0.
prepare/embed precede target inference; accept runs AFTER target greedy output.
No allocation, upload, communication or capture is hidden in these leaves.
"""
import ctypes as C
from ops.queued import queued

import torch

Q = 6




def commit_resource_specs(batch, windows, sources):
    """Caller allocations; host_descriptors is CPU, others are owning NPU.

    Inputs borrowed separately: result INT64[B,8], canonical pos INT32[S],
    table INT64[S,max_pages], each window's BF16[B*6,512] pending + main_kv,
    and each full source's pending/index_pending + canonical pools/carry.
    checked is the output receipt; col5 is a batch-wide error, not a host read.
    No hidden workspace. Descriptor tensors stay immutable after upload.
    """
    if (type(batch) is not int or not 1 <= batch <= 4
            or type(windows) is not int or not 1 <= windows <= 43
            or type(sources) is not int or not 0 <= sources <= 4):
        raise ValueError('commit requires B1..4, 1..43 windows, 0..4 full sources')
    return dict(descriptors=((windows+sources,12), 'int64'),
                host_descriptors=((windows+sources,12), 'int64'),
                checked=((batch,8), 'int64'))


class Commit:
    """One native accepted-prefix transaction; no prefill, allocation or fallback.

    windows: sequence of (WindowPast, TargetLayer.pending_kv).
    sources: sequence of (SourcePast, Source) for FULL sources only; reindex
    aliases are rejected, not silently skipped. Bind every canonical window and
    source needed by this decode, exactly once. Layer identity/completeness is
    the model builder's responsibility, not inferred from memory addresses.

    Constructor fills caller host_descriptors only. Caller must explicitly do
    descriptors.copy_(host_descriptors) and complete the upload BEFORE capture.
    All resources and Past objects are retained for graph lifetime. No resize,
    rebinding or concurrent writer to these resources is allowed.

    __call__(stream) accepts the live stream handle, never caches a stream.
    Native rc validates static ABI; checked[:,5] reports device metadata errors.
    Invalid dynamic input aborts the entire batch before canonical writes.
    Synchronize/read the receipt at the scheduler's existing acceptance boundary;
    do not continue on error. Pos is published last on that stream. Host Past.pos
    is NOT modified here: scheduler mirrors successful receipt after sync, before
    cold export/dismiss/prefill. This binding makes no host-device synchronization.
    """
    def __init__(self, result, pos, table, descriptors, host_descriptors, checked,
                 *, windows, sources, page_tokens, max_seq, library):
        b = result.shape[0]
        windows, sources = tuple(windows), tuple(sources)
        specs = commit_resource_specs(b, len(windows), len(sources))
        device = result.device
        slots, mp = table.shape
        ring, pad = windows[0][0].ring, windows[0][0].pad
        pages = sources[0][0].pt.n_pages if sources else 1
        geometry = (slots, pages, mp, page_tokens, ring, pad, max_seq)
        if (any(type(x) is not int or not 0 < x < 2**32 for x in geometry)
                or slots < b or page_tokens < Q or page_tokens % 2
                or ring < 144 or pad < 16 or ring+pad >= 2**32
                or max_seq >= 2**31 or mp != (max_seq+page_tokens-1)//page_tokens):
            raise ValueError('invalid canonical commit geometry')
        buffers = []
        def check(t, shape, dtype):
            if (tuple(t.shape) != tuple(shape) or t.dtype != getattr(torch,dtype)
                    or not t.is_contiguous() or device.type != 'npu'
                    or t.device != device or t.data_ptr() % 32):
                raise ValueError('commit tensor shape/dtype/device/alignment mismatch')
            lo, hi = t.data_ptr(), t.data_ptr()+t.numel()*t.element_size()
            if any(lo < x.data_ptr()+x.numel()*x.element_size() and x.data_ptr() < hi
                   for x in buffers):
                raise ValueError('commit resources must be pairwise disjoint')
            buffers.append(t)
            return lo
        for name, t in (('descriptors',descriptors), ('checked',checked)):
            check(t, *specs[name])
        check(result,(b,8),'int64');check(pos,(slots,),'int32')
        check(table,(slots,mp),'int64')
        shape, _ = specs['host_descriptors']
        if (host_descriptors.device.type != 'cpu' or host_descriptors.dtype != torch.int64
                or tuple(host_descriptors.shape) != shape or not host_descriptors.is_contiguous()):
            raise ValueError('preallocated CPU INT64 descriptor staging required')
        rows = []
        for window, pending in windows:
            if (window.n_slots != slots or window.ring != ring or window.pad != pad):
                raise ValueError('all windows must use the canonical geometry')
            p = check(pending,(b*Q,512),'bfloat16')
            bank = check(window.main_kv,(slots,pad+ring,512),'bfloat16')
            rows.append((0,p,bank,0,0,0,0,0,0,0,0,0))
        for source, plan in sources:
            ratio = source.ratio
            if (ratio not in (1,2) or plan.ratio != ratio or plan.batch != b
                    or plan.borrowed is not None or plan.closed
                    or plan.layer != source.source_layer
                    or source.pt.table.data_ptr() != table.data_ptr()
                    or (source.pt.n_slots,source.pt.n_pages,source.pt.max_pages,
                        source.pt.page_tokens,source.pt.max_seq) != (slots,pages,mp,page_tokens,max_seq)
                    or plan.tensors['index_bank'].data_ptr() != source.index_pool.data.data_ptr()
                    or plan.tensors['table'].data_ptr() != table.data_ptr()):
                raise ValueError('full source must match canonical Past, not reindex')
            p = check(plan.pending,(b,Q,512),'bfloat16')
            bank = check(source.ckv_pool.data,(pages,page_tokens//ratio,512),'bfloat16')
            ip = check(plan.index_pending,(b,Q,128),'bfloat16')
            ib = check(source.index_pool.data,
                       (pages+source.index_pool.reserve,page_tokens//ratio,128),'bfloat16')
            carry = (0,0,0,0)
            if ratio == 2:
                if (plan.tensors['carry_values'].data_ptr() != source.kv_state.data_ptr()
                        or plan.tensors['carry_scores'].data_ptr() != source.score_state.data_ptr()):
                    raise ValueError('source projections must have read this canonical carry')
                carry = (check(plan.projected_values,(b*Q,512),'float32'),
                         check(plan.projected_gates,(b*Q,512),'float32'),
                         check(source.kv_state,(slots,4,512),'float32'),
                         check(source.score_state,(slots,4,512),'float32'))
            rows.append((ratio,p,bank,ip,ib,*carry,0,0,0))
        host_descriptors.numpy()[:] = rows
        self.resources = tuple(buffers)+(host_descriptors,windows,sources)
        self.lib = C.CDLL(str(library))
        self.fn = self.lib.dec_commit_prefix
        self.fn.argtypes = [C.c_void_p]*6 + [C.c_uint32]*10
        self.fn.restype = C.c_int
        self.fn = queued(self.fn, self.fn.argtypes, check_status=True)
        self.args = tuple(C.c_void_p(t.data_ptr()) for t in
                          (result,pos,table,descriptors,checked)) + (
                          b,slots,pages,mp,page_tokens,ring,pad,max_seq,len(windows),len(sources))
        self.checked = checked

    def __call__(self, stream):
        rc = self.fn(stream, *self.args)
        if rc:
            raise RuntimeError(f'dec_commit_prefix failed: {rc}')
