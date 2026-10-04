"""Fixed-resource Q6 target and NativeDSpark decode; no arena, prefill operators or second engine.

Build owns storage/descriptors only; canonical weights, Past and TP communicator
remain borrowed. All production verification uses startup-captured B1..B4 graphs.
"""
import ctypes as C
from pathlib import Path
from model.decode_ops import native
from ops.queued import queued

P, U, F = C.c_void_p, C.c_uint32, C.c_float


def _function(lib, name, args, result=C.c_int):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, result
    return queued(fn, args, name, check_status=True)


def _check(rc, op):
    if rc:
        raise RuntimeError(f'{op} failed: {rc}')


class NativeDecode:
    """One serialized B1..B4 target/draft runtime, borrowing weights/Past/TP ownership.

    Startup builds every descriptor and tensor. capture() records verify/seed/draft;
    commit is an explicit native transaction after verify, never a capture side
    effect. Admission seeds CED features; NativeDSpark produces every Q6 window.
    budget_bytes covers owned device storage, not borrowed weights/Past, CANN
    descriptors, graph-pool overhead or ACL pinned host staging (10368*sum(B)).
    """
    def __init__(self, weights, past, *, parallel, metadata, host_tables,
                 capacity, budget_bytes, batches=(1,2,3,4), libraries=None,
                 routed_bank13=None, prepared_weights=None):
        import json
        import numpy as np
        import torch
        from model.decode_engram import DecodeHash
        batches = tuple(batches)
        if (1 not in batches or len(set(batches)) != len(batches)
                or any(type(b) is not int or not 1 <= b <= 4 for b in batches)
                or max(batches) > past.n_slots or not 6 <= capacity <= past.max_seq
                or parallel.world != 8 or parallel.rank != weights.rank
                or parallel.device != weights.device or not budget_bytes > 0):
            raise ValueError('fixed B1..4, TP8 owners, capacity and explicit budget required')
        self.weights, self.past, self.parallel = weights, past, parallel
        self.capacity, self.limit = capacity, budget_bytes
        self.state, self.current, self.plans = 'building', None, {}
        self.draft_state = {}  # slot -> last accepted receipt, independent of batch row
        self.storage, self.prepared, self.shared = [], {}, {}
        self.w8_prepared = {}
        self.w4_scales = {}  # One encoded scale per projection, shared by all batches.
        # Borrow immutable conversions from the owning prefill engine.
        for (name, dtype), value in (prepared_weights or {}).items():
            original = weights[name]
            if (value.shape != original.shape or value.dtype != dtype
                    or value.device != weights.device or not value.is_contiguous()):
                raise ValueError('invalid borrowed prepared weight: ' + name)
            self.prepared[name, dtype] = value
        self.draft_windows = tuple(past.windows[l] for l in (40,41,42))
        self.draft_slots = tuple(tuple(w.main_kv[slot] for w in self.draft_windows)
                                for slot in range(past.n_slots))
        self.used_bytes = 0
        self.config = json.loads(Path(__file__).with_name('v41_config.json').read_text())
        if self.config['n_layers'] != 40 or set(host_tables) != set(self.config['engram_layer_ids']):
            raise ValueError('all 40 target layers and both canonical Engram tables required')
        self.hasher = DecodeHash(metadata)
        self.host_tables = dict(host_tables)
        root = Path(__file__).resolve().parents[1] / 'ops/decode'
        paths = {n:root / ('libdecode_'+n+'.so') for n in ('norm','attention','gemm','window','source_cube')}
        if libraries is not None:
            paths.update(libraries)
        self.paths = paths
        self.libs = {n:C.CDLL(str(p)) for n,p in paths.items()}
        self.acl = C.CDLL('libascendcl.so')
        self.stream = torch.npu.current_stream(weights.device)
        try:
            # Shared transient score bank; prefill/decode are owner-serialized.
            shape = (384,576,5120)
            if routed_bank13 is not None:
                if (tuple(routed_bank13.shape) != shape
                        or routed_bank13.dtype != torch.bfloat16
                        or routed_bank13.device != weights.device
                        or not routed_bank13.is_contiguous()):
                    raise ValueError('invalid borrowed decode expert bank')
                self.shared['routed_bank13'] = routed_bank13
            else:
                self.shared['routed_bank13'] = self.alloc(shape, torch.bfloat16)
            # CPU-only released YaRN formula; no prefill helper or device temporary.
            c = self.config
            def inverse(compressed):
                base = c['compress_rope_theta'] if compressed else c['rope_theta']
                f = 1 / (base ** (np.arange(32,dtype=np.float64)/32))
                if compressed:
                    import math
                    correction = lambda r: 64*math.log(c['original_seq_len']/(r*2*math.pi))/(2*math.log(base))
                    low, high = (max(math.floor(correction(c['beta_fast'])),0),
                                 min(math.ceil(correction(c['beta_slow'])),63))
                    ramp = np.clip((np.arange(32)-low)/max(high-low,1e-3),0,1)
                    f = f/c['rope_factor']*ramp + f*(1-ramp)
                return f.astype(np.float32)
            self.inv_freq = self.alloc((32,),torch.float32)
            self.inv_freq.copy_(torch.from_numpy(inverse(False)))
            self.rope = self.alloc((capacity,32,2),torch.float32)
            # Bounded host staging; all slicing/uploads happen before capture.
            f = inverse(True)
            for start in range(0,capacity,4096):
                angles = np.arange(start,min(start+4096,capacity),dtype=np.float32)[:,None]*f
                rows = np.stack((np.cos(angles),np.sin(angles)),axis=-1)
                self.rope[start:start+len(rows)].copy_(torch.from_numpy(rows))
            # Immutable per-batch tiling shared by all layers; charged at startup.
            for prefix, filename in (('wo_b', 'tiling_m%d.bin'),
                                     ('draft_vocab', 'draft_vocab_tiling_m%d.bin'),
                                     ('markov', 'markov_tiling_m%d.bin'),
                                     ('draft_down', 'draft_down_tiling_m%d.bin')):
                for b in batches:
                    rows = b if prefix == 'markov' else 6*b
                    raw = (root / (filename % rows)).read_bytes()
                    if len(raw) != 200:
                        raise ValueError(prefix + ' Cube tiling ABI mismatch')
                    tile = self.alloc((224,), torch.uint8)
                    tile.zero_()
                    tile[:200].copy_(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()))
                    self.shared[prefix + '_tiling_%d' % b] = tile
            for n in (1024, 2048):
                raw = (root / ('source_cube_tiling_n%d.bin' % n)).read_bytes()
                if len(raw) != 200:
                    raise ValueError('source Cube tiling ABI mismatch')
                tile = self.alloc((224,), torch.uint8)
                tile.zero_()
                tile[:200].copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
                self.shared['source_cube_tiling_n%d' % n] = tile
            self._prepare_w8()
            for b in batches:
                plan = DecodeBatch(self, b)
                self.plans[b] = plan
            size = max(p.workspace_bytes for p in self.plans.values())
            self.workspace = self.alloc((max(32,size),),torch.uint8)
            # Source stages and batch plans are serialized on the owner stream.
            # Allocate once: growing during bind retained both physical banks.
            logits_size = max(source.batch*source.geometry[2]*24*source.geometry[3]
                              for plan in self.plans.values()
                              for source in plan.sources.values())
            self.cube_logits = self.alloc((logits_size,),torch.float32)
            self.stream.synchronize()
            self.state = 'idle'
        except BaseException:
            self.close()
            raise

    def alloc(self, shape, dtype):
        import math
        import torch
        n = math.prod(shape)*torch.empty((),dtype=dtype,device='cpu').element_size()
        if self.used_bytes+n > self.limit:
            raise MemoryError(f'decode budget exceeded: {self.used_bytes+n} > {self.limit}')
        value = torch.empty(shape,dtype=dtype,device=self.weights.device)
        self.storage.append(value)
        self.used_bytes += n
        return value

    def _prepare_w8(self):
        """Fixed 710 MiB/rank whitelist, shared by B1..4 and source/attention.

        Startup only; charge the existing explicit budget, never shrink Past or
        consume its reserve. Keep wq_b, shared/routed experts and draft dynamic.
        The SAME native INT8->FP16->FP32*scale->BF16(RINT) kernel fills the cache.
        Weights/scales and these buffers are immutable until graphs are reset.
        """
        import torch
        if self.state != 'building' or self.w8_prepared:
            raise RuntimeError('prepare W8 exactly once before capture')
        specs = [(f'layers.{l}.attn.{name}', shape)
                 for l in range(40)
                 for name, shape in (('wq_a', (1280,5120)), ('wkv', (512,5120)))]
        specs += [(f'layers.{l}.attn.indexer.wq_b', (512,1280))
                  for l in self.config['index_source_layers']]
        required = sum(n*k*2 for _, (n,k) in specs)
        if required != 710*1024*1024:
            raise ValueError('unexpected fixed W8 whitelist geometry')
        if self.used_bytes + required > self.limit:
            raise MemoryError('fixed W8 cache exceeds decode startup budget')
        # Validate ALL canonical inputs before allocating the first cache entry.
        for name, shape in specs:
            w, sc = self.weights[name+'.weight'], self.weights[name+'.scale']
            if (tuple(w.shape) != shape or w.dtype != torch.int8
                    or tuple(sc.shape) != (shape[0],1) or sc.dtype != torch.float32
                    or any(t.device != self.weights.device or not t.is_contiguous()
                           or t.data_ptr() % 32 for t in (w,sc))):
                raise ValueError('invalid cached W8 weight/scale: '+name)
        expand = _function(self.libs['gemm'], 'dec_w8_expand', [P]*4+[U]*2)
        with torch.npu.stream(self.stream):
            for name, shape in specs:
                w, sc = self.weights[name+'.weight'], self.weights[name+'.scale']
                dense = self.alloc(shape, torch.bfloat16)
                _check(expand(P(self.stream.npu_stream), P(w.data_ptr()),
                    P(sc.data_ptr()), P(dense.data_ptr()), *shape), 'startup W8 expand')
                self.w8_prepared[name] = dense

    def routed_scale(self, name):
        """Encode a persisted FP32 expert scale once, outside graph capture.

        The allocation belongs to this decode owner, not individual layer/batch
        plans. Original FP32 scales remain available to the independent prefill
        reader. CANN's temporary conversion result is startup-only.
        """
        import torch
        import torch_npu
        if name in self.w4_scales:
            return self.w4_scales[name]
        if self.state != 'building':
            raise RuntimeError('routed scales must be prepared before capture')
        if not name.endswith('.scale'):
            raise ValueError('expected a routed projection scale name')
        stem = name[:-len('.scale')]
        scale = self.weights[name]
        weight = self.weights[stem+'.weight']
        bias = self.weights[stem+'.hp_bias']
        e = 128 if stem.startswith('mtp.') else 384
        if stem.endswith('.ffn.w13'):
            kp,n = 5120,576
        elif stem.endswith('.ffn.w2'):
            kp,n = 320,5120
        else:
            raise ValueError('not a routed expert projection')
        for value,shape,dtype in ((scale,(e,n),torch.float32),
                                  (weight,(e,kp,n//8),torch.int32),
                                  (bias,(e,n),torch.float32)):
            if (tuple(value.shape)!=shape or value.dtype!=dtype
                    or value.device!=self.weights.device or not value.is_contiguous()
                    or value.data_ptr()%32):
                raise ValueError('incompatible persisted routed NZ projection: '+stem)
        out = self.alloc((e,1,n),torch.int64)
        with torch.npu.stream(self.stream):
            encoded = torch_npu.npu_trans_quant_param(scale.reshape(-1))
            out.copy_(encoded.reshape(e,1,n))
        self.w4_scales[name] = out
        return out

    def weight(self, name, dtype):
        value = self.weights[name]
        if value.dtype == dtype:
            return value
        key = (name,dtype)
        if key not in self.prepared:
            out = self.alloc(tuple(value.shape),dtype)
            out.copy_(value)
            self.prepared[key] = out
        return self.prepared[key]

    @property
    def budget(self):
        return dict(device_bytes=self.used_bytes, limit_bytes=self.limit,
                    pinned_engram_bytes=10368*sum(self.plans),
                    workspace_bytes=self.workspace.numel(),
                    w8_cache_bytes=sum(t.numel()*t.element_size() for t in self.w8_prepared.values()),
                    w4_scale_bytes=sum(t.numel()*t.element_size() for t in self.w4_scales.values()),
                    excludes='borrowed weights/Past/transport, graph pools, CANN descriptors')

    def _idle(self):
        if self.state != 'idle':
            raise RuntimeError('decode requires idle state; finish/discard the pending window')

    def capture(self):
        """Startup only, all TP ranks in identical B order, on the owning stream.

        Warm/capture dummy inactive rows. No canonical Past writes occur. Graph
        replay is the only production verify path; there is no eager fallback.
        """
        import torch
        self._idle()
        if any(p.graph is not None for p in self.plans.values()):
            raise RuntimeError('capture exactly once before serving')
        # HCCL config is process-wide. Only decode graphs use the faster
        # standard reduction; restore the caller's policy before prefill runs.
        reduction_policy = C.c_int32()
        self.parallel._call('HcclGetConfig', 0, C.byref(reduction_policy))
        self.state = 'capturing'
        try:
            torch.npu.synchronize()
            # Captured reductions remain nondeterministic after policy restore; greedy output may vary.
            self.parallel._call('HcclSetConfig', 0, 0)
            # Build may run on the default stream; capture must own a side stream.
            build_stream = self.stream
            self.stream = torch.npu.Stream(device=self.weights.device)
            self.stream.wait_stream(build_stream)
            with torch.npu.stream(self.stream):
                for p in self.plans.values():
                    p.stage(((0,)*6,)*p.batch,(-1,)*p.batch,(0,)*p.batch,(0,)*p.batch,((),)*p.batch)
                    p.prefetch.upload(self.stream)
                    p.checked.zero_()
                    for name,run in (('graph',p.verify),('seed_graph',p.seed),
                                     ('draft_graph',p.propose)):
                        run()
                        self.stream.synchronize()
                        graph = torch.npu.NPUGraph()
                        setattr(p,name,graph)
                        with torch.npu.graph(graph,stream=self.stream):
                            run()
                    self.stream.synchronize()
                    p.drain(completed_stream=self.stream.npu_stream)
            self.state = 'idle'
        except BaseException:
            self.state = 'failed'
            raise
        finally:
            try:
                torch.npu.synchronize()
            finally:
                self.parallel._call('HcclSetConfig', 0, reduction_policy.value)
        return self

    def seed_prefill(self, *, slot, anchor, main_hidden, producer_stream):
        """Admit the canonical CED tail, right-shifted into draft rings.

        main_hidden is borrowed BF16 [T,15360], right-aligned for short prompts.
        The producer stream is explicit: the prefill lane need not be current.
        Reuse the B1 Q6 seed graph in chunks; never run a prefill operator here.
        Synthetic seed metadata describes already committed prompt rows, not a
        target commit. No Past cursor is advanced by this operation.
        """
        import torch
        self._idle()
        p = self.plans[1]
        if p.seed_graph is None or p.draft_graph is None:
            raise RuntimeError('capture all decode graphs before admission')
        if (type(slot) is not int or not 0 <= slot < self.past.n_slots
                or slot in self.past.free_slots or slot in self.past.replay_pending
                or slot in self.draft_state or type(anchor) is not int
                or not 0 <= anchor < 129280):
            raise ValueError('one newly admitted hot slot and valid anchor required')
        end = self.past.pos[slot]
        count = min(end,128)
        if (end < 1 or main_hidden.ndim != 2 or main_hidden.shape[1] != 15360
                or main_hidden.shape[0] < count or main_hidden.dtype != torch.bfloat16
                or main_hidden.device != self.weights.device):
            raise ValueError('complete canonical CED feature tail required')
        self.state = 'failed'
        self.stream.wait_stream(producer_stream)
        with torch.npu.stream(self.stream):
            # A recycled slot must not retain an earlier request's draft history.
            for ring in self.draft_slots[slot]:
                ring.zero_()
            base = end-count
            for offset in range(0,count,6):
                n = min(6,count-offset)
                p.features.zero_()
                lo = main_hidden.shape[0]-count+offset
                p.feature_rows[n].copy_(main_hidden[lo:lo+n])
                receipt = [slot,base+offset,n,anchor,1,0,0,0]
                p.checked_view[0] = receipt
                p.checked.copy_(p.host_checked,non_blocking=True)
                p.seed_graph.replay()
                # The same pinned staging buffer is reused by the next chunk.
                self.stream.synchronize()
        self.draft_state[slot] = receipt
        self.state = 'idle'

    def release(self, slot):
        self._idle()
        self.draft_state.pop(slot,None)

    def begin(self, *, slots, history_tokens, temps=None):
        """Generate Q6 internally, then prepare CPU Engram hashes for verify.

        The only token source is the captured NativeDSpark proposal graph.
        D2H uses startup pinned storage; Engram currently requires host IDs.
        """
        import torch
        self._idle()
        slots = tuple(slots)
        p = self.plans.get(len(slots))
        if p is None or p.graph is None or p.draft_graph is None:
            raise RuntimeError('batch must have a startup-captured graph group')
        histories = tuple(tuple(r) for r in history_tokens)
        if (len(histories) != len(slots) or len(set(slots)) != len(slots)
                or any(type(s) is not int or not 0 <= s < self.past.n_slots for s in slots)):
            raise ValueError('one distinct valid slot and history per row required')
        starts = tuple(self.past.pos[s] for s in slots)
        receipts = []
        for slot,start,hist in zip(slots,starts,histories):
            row = self.draft_state.get(slot)
            if (slot in self.past.free_slots or slot in self.past.replay_pending
                    or row is None or row[1]+row[2] != start
                    or len(hist) != min(start,3)
                    or any(type(t) is not int or not 0 <= t < 129280 for t in hist)
                    or start+6 > self.capacity
                    or self.past.pt.n_alloc(slot)*self.past.page_tokens < start+6):
                raise ValueError('decode requires seeded hot slots, reserved Q6 pages and exact history')
            receipts.append(row)
        self.current, self.state = p, 'staging'
        try:
            self.stream.wait_stream(torch.npu.current_stream(self.weights.device))
            with torch.npu.stream(self.stream):
                p.checked_view[:] = receipts
                p.checked.copy_(p.host_checked,non_blocking=True)
                p.draft_graph.replay()
                p.host['tokens'].copy_(p.draft.ids,non_blocking=True)
            self.stream.synchronize()
            tokens = tuple(tuple(r) for r in p.host_views['tokens'].tolist())
            if any(row[0] != receipt[3] or any(not 0 <= t < 129280 for t in row)
                   for row,receipt in zip(tokens,receipts)):
                raise RuntimeError('NativeDSpark returned invalid Q6 IDs')
            with torch.npu.stream(self.stream):
                p.stage(tokens,slots,starts,(1,)*p.batch,histories,temps=temps)
            self.state = 'prepared'
        except BaseException:
            self.state = 'failed'
            raise
        return p

    def verify(self):
        import torch
        if self.state != 'prepared':
            raise RuntimeError('begin before verify')
        self.state = 'failed'
        with torch.npu.stream(self.stream):
            p = self.current
            if max(p.starts) + 6 > self.capacity:
                raise RuntimeError('verify window exceeds capacity')
            p.graph.replay()
        p.prefetch.upload(p.engram_stream)
        self.state = 'verified'
        return self.current.result

    def commit(self):
        """Submit exactly once; finish() is mandatory before any Past consumer."""
        import torch
        if self.state != 'verified':
            raise RuntimeError('verify before accepted-prefix commit')
        self.state = 'failed'
        with torch.npu.stream(self.stream):
            self.current.commit(P(self.stream.npu_stream))
        self.state = 'committing'

    def finish(self):
        """D2H at acceptance boundary, batch-wide validation, host pos mirror."""
        import torch
        if self.state != 'committing':
            raise RuntimeError('commit before finish')
        p = self.current
        self.state = 'failed'
        with torch.npu.stream(self.stream):
            p.host_checked.copy_(p.checked,non_blocking=True)
        self.stream.synchronize()
        rows = p.checked_view.tolist()
        # Validate the complete receipt before updating ANY host cursor.
        for row,slot,start in zip(rows,p.slots,p.starts):
            if (row[5] or row[0] != slot or row[1] != start or row[4] != 1
                    or not 1 <= row[2] <= 6 or not 0 <= row[3] < 129280
                    or self.past.pos[slot] != start):
                raise RuntimeError(f'decode commit receipt failed: {rows}')
        # Seed accepted target features before any target scratch is reused.
        with torch.npu.stream(self.stream):
            p.seed_graph.replay()
        self.stream.synchronize()
        for row in rows:
            self.past.pos[row[0]] = row[1]+row[2]
            self.draft_state[row[0]] = row
        p.drain(completed_stream=self.stream.npu_stream)
        self.current, self.state = None, 'idle'
        return rows

    def discard(self):
        """Drop an uncommitted verify; never discard a submitted commit receipt."""
        if self.state not in ('prepared','verified','staging'):
            raise RuntimeError('only an uncommitted window may be discarded')
        self.stream.synchronize()
        self.current.drain(completed_stream=self.stream.npu_stream)
        self.current, self.state = None, 'idle'

    def close(self):
        if self.state == 'closed':
            return
        self.stream.synchronize()
        for plan in reversed(tuple(self.plans.values())):
            plan.close()
        self.plans.clear()
        self.draft_state.clear()
        self.draft_windows = self.draft_slots = ()
        self.current, self.state = None, 'closed'
        self.storage.clear()
        self.prepared.clear()
        self.shared.clear()
        self.w8_prepared.clear()
        self.w4_scales.clear()
        self.workspace = self.cube_logits = self.inv_freq = self.rope = None
        self.host_tables.clear()
        self.weights = self.past = self.parallel = self.hasher = None


class DecodeBatch:
    """Fixed addresses for one batch; no owning weights, Past or transport."""
    def __init__(self, owner, batch):
        import torch
        from model.decode_source import Source, source_resource_specs
        from model.decode_engram import NativeEngram, EngramPrefetch, engram_resource_specs
        from model.decode_head import NativeHead, NativeDSpark, dspark_resource_specs
        from ops.decode.window import Commit, commit_resource_specs
        self.owner, self.batch, self.graph = owner, batch, None
        self.layers, self.sources, self.engrams, self.targets = [], {}, {}, {}
        self.head = self.draft = None
        self.seed_graph = self.draft_graph = None
        self.shared_stream = torch.npu.Stream(device=owner.weights.device)
        self.hc_stream = torch.npu.Stream(device=owner.weights.device)
        self.engram_stream = torch.npu.Stream(device=owner.weights.device)
        owner.plans[batch] = self
        o, b, r = owner, batch, batch*6
        c, past, parallel = o.config, o.past, o.parallel
        bf, fp, ix = torch.bfloat16, torch.float32, torch.int64
        a = o.alloc
        norm, attn, gemm, window = (o.libs[k] for k in ('norm','attention','gemm','window'))
        self.tokens, self.meta = a((b,6),ix), a((b,8),ix)
        self.slot_ids, self.start, self.active = (a((b,),ix) for _ in range(3))
        self.greedy, self.result, self.checked = a((b,6),ix), a((b,8),ix), a((b,8),ix)
        # Sampling state. temp[B] is per-request (0 = greedy, reproducing the
        # greedy kernel token for token); sample_seed[1] is the shared draw
        # counter, identical on every rank and bumped inside the graph.
        self.temp, self.sample_seed = a((b,),fp), a((1,),torch.int32)
        self.temp.zero_()
        self.sample_seed.fill_(1)
        self.host = {k:torch.empty(tuple(t.shape),dtype=t.dtype,device='cpu',pin_memory=True)
                     for k,t in (('tokens',self.tokens),('meta',self.meta),('slots',self.slot_ids),
                                 ('start',self.start),('active',self.active),('temp',self.temp))}
        self.host_checked = torch.empty((b,8),dtype=ix,device='cpu',pin_memory=True)
        self.host_views = {k:t.numpy() for k,t in self.host.items()}
        self.checked_view = self.host_checked.numpy()
        self.uploads = tuple((t,self.host[k]) for k,t in (
            ('tokens',self.tokens),('meta',self.meta),('slots',self.slot_ids),
            ('start',self.start),('active',self.active),('temp',self.temp)))
        self.features = a((r,15360),bf)
        self.feature_rows = {n:self.features[:n] for n in range(1,7)}
        hidden, alternate = a((r,20480),bf), a((r,20480),bf)
        pre, freq = a((r,4),fp), a((r,32,2),fp)
        self.hidden, self.freqs, self.legacy_pre = hidden, freq, a((b,6,8),fp)
        pre.zero_()
        pre[:,0].fill_(1.)
        self.events = tuple(torch.npu.Event() for _ in range(4))
        scratch = dict(o.shared,normalized=a((r,5120),bf))
        transient = {}
        def resources(specs, supplied, persistent=()):
            out = dict(supplied)
            for name,(shape,dtype) in specs.items():
                if name in out:
                    continue
                key = (name,shape,dtype)
                if name in persistent:
                    out[name] = a(shape,getattr(torch,dtype))
                else:
                    if key not in transient:
                        transient[key] = a(shape,getattr(torch,dtype))
                    out[name] = transient[key]
            return out
        # Dedicated per-query HSI pool: live from layer 20 through layer 36.
        self.candidate_rows = a((b,6,16384),ix)
        for layer in range(40):
            view = past.views[layer]
            source = None
            if view.mode in ('full','reindex'):
                canonical = past.sources[view.kv_source_layer]
                borrowed = self.sources.get(view.kv_source_layer) if view.mode == 'reindex' else None
                width = ((o.capacity+view.ratio-1)//view.ratio+7)//8*8
                if view.mode == 'reindex':
                    width = 16384
                specs = source_resource_specs(batch=b,ratio=view.ratio,score_width=width,
                                              reindex=borrowed is not None)
                supplied = dict(hidden=scratch['normalized'],slots=self.slot_ids,start=self.start,
                    active=self.active,index_bank=canonical.index_pool.data,table=past.pt.table)
                # Source scoring/reduction/selection finishes before this layer's
                # FFN expands experts on the same stream. Only selected IDs and
                # pending rows survive; those remain separate persistent storage.
                score_bank = o.shared['routed_bank13'].view(-1).view(fp)
                score_count = b*6*width
                if 2*score_count > score_bank.numel():
                    raise ValueError('decode scores exceed shared expert scratch')
                supplied.update(
                    local_scores=score_bank[:score_count].view(b,6,width),
                    scores=score_bank[score_count:2*score_count].view(b,6,width))
                if view.ratio == 2:
                    supplied.update(carry_values=canonical.kv_state,carry_scores=canonical.score_state)
                tensors = resources(specs,supplied,('pending','index_pending','values','gates',
                                                       'selected','token_freqs','group_freqs'))
                prefix = f'layers.{layer}.attn.'
                names = {'q_norm':'q_norm.weight'}
                if borrowed is None:
                    names.update(c_norm='compressor.norm.weight',i_norm='indexer.k_norm.weight')
                    if view.ratio == 2:
                        names.update(c_wkv='compressor.wkv.weight',c_wgate='compressor.wgate.weight')
                prepared = {k:o.weight(prefix+name,fp) for k,name in names.items()}
                source = Source(o.weights,layer,tensors,prepared,batch=b,ratio=view.ratio,
                    score_width=width,norm_library=norm,gemm_library=gemm,eps=c['norm_eps'],
                    borrowed=borrowed,w8_prepared=o.w8_prepared,
                    candidate_rows=self.candidate_rows if layer >= c['candidate_source_layer'] else None,
                    produce_candidates=layer == c['candidate_source_layer'], parallel=o.parallel)
                self.sources[layer] = source
            compressed = None
            layer_freq = freq
            if view.ratio:
                kv, index = self.sources[view.kv_source_layer], self.sources[view.index_source_layer]
                compressed = dict(bank=past.sources[view.kv_source_layer].ckv_pool.data,
                    table=past.pt.table,compressed_pending=kv.pending,ids=index.selected)
                layer_freq = index.token_freqs
            if layer in c['engram_layer_ids']:
                specs = engram_resource_specs(batch=b)
                eh = hidden.view(r,4,5120)
                # Packed rows must survive host staging of BOTH Engram layers.
                et = resources(specs,dict(hidden=eh,out=eh),('packed',))
                prefix = f'layers.{layer}.engram.'
                key = ('engram_qk',layer)
                if key not in o.prepared:
                    qk = a((4,5120),fp)
                    torch.mul(o.weight(prefix+'q_weight',fp),
                              o.weight(prefix+'k_weight',fp),out=qk)
                    o.prepared[key] = qk
                engram = NativeEngram(o.weights,layer,o.hasher,o.host_tables[layer],et,
                    {'qk':o.prepared[key]},batch=b,norm_library=norm,acl=o.acl,
                    parallel=parallel,eps=c['norm_eps'])
                self.engrams[layer] = engram
            # Keep the last three full HC states for the public DSpark boundary.
            output = a((r,20480),bf) if layer in c['dspark_target_layer_ids'] else alternate
            target = TargetLayer(o.weights,c,layer=layer,batch=b,parallel=parallel,
                past=past.windows[layer],x=hidden,incoming=pre,output=output,freqs=layer_freq,
                slots=self.slot_ids,start=self.start,active=self.active,norm_library=norm,
                attention_library=attn,gemm_library=gemm,scratch=scratch,compressed=compressed,
                allocator=a,prepare_weight=o.weight,w8_prepared=o.w8_prepared,
                prepare_routed_scale=o.routed_scale,
                wo_b_tiling=o.shared['wo_b_tiling_%d' % b],
                query_latent=self.sources[layer].query_latent if layer in self.sources else None)
            self.layers.append(target)
            hidden, alternate, pre = output, hidden, target.pre
            if layer in c['dspark_target_layer_ids']:
                self.targets[layer] = (hidden,pre)
        self.prefetch = EngramPrefetch(self.engrams,
            {layer:a((4,),ix) for layer in self.engrams},torch.npu.Event(),o.acl)
        self.head = NativeHead(o.weights,batch=b,hidden=hidden.view(b*6,20480),pre=pre,
            norm_weight=a((5120,),fp),normalized=a((r,5120),bf),logits=a((r,16160),fp),
            norm_library=norm,eps=c['norm_eps'],
            projection_input=(a((r,5120),fp)
                              if o.weights['head.weight'].dtype == fp else None))
        size = window.dec_greedy_workspace_bytes
        size.argtypes, size.restype = [U], C.c_uint64
        self.partial, self.pair, self.gathered = a((size(b),),torch.uint8),a((r,2),fp),a((8,r,2),fp)
        greedy = _function(window,'dec_sample_tp8',[P]*10+[U]*3)
        accept = _function(window,'dec_accept',[P]*5+[U]*2)
        comm = parallel.comm.value if isinstance(parallel.comm,P) else parallel.comm
        self.greedy_fn, self.accept_fn = greedy, accept
        self.greedy_args = tuple(P(t.data_ptr()) for t in
            (self.head.logits,self.partial,self.pair,self.gathered,self.greedy,
             self.temp,self.sample_seed)) + (
                P(comm),C.cast(parallel.lib.HcclAllGather,P),b,parallel.rank,8)
        self.accept_args = tuple(P(t.data_ptr()) for t in
            (self.tokens,self.greedy,self.meta,self.result))+(b,past.n_slots)
        # Draft direct projections use small explicit quantized scratch.
        draft_resources = {}
        for key,(shape,dtype) in dspark_resource_specs(b,o.weights).items():
            if key in draft_resources:
                continue
            if key.startswith(('dense:', 'fp:')):
                # Immutable, identically initialized before any graph capture.
                shared_key = 'draft:' + key
                if shared_key not in o.shared:
                    o.shared[shared_key] = (o.weight(key[3:],dtype)
                                           if key.startswith('fp:') else a(shape,dtype))
                draft_resources[key] = o.shared[shared_key]
            else:
                draft_resources[key] = a(shape,dtype)
        self.draft = NativeDSpark(o.weights,batch=b,features=self.features,
            result=self.checked,windows=o.draft_windows,resources=draft_resources,
            norm_library=norm,window_library=window,opapi=C.CDLL('libopapi.so'),
            parallel=parallel,comm=comm,all_gather=C.cast(parallel.lib.HcclAllGather,P).value,
            config=c,wo_b_tiling=o.shared['wo_b_tiling_%d' % b],
            draft_vocab_tiling=o.shared['draft_vocab_tiling_%d' % b],
            markov_tiling=o.shared['markov_tiling_%d' % b],
            draft_down_tiling=o.shared['draft_down_tiling_%d' % b],
            prepare_routed_scale=o.routed_scale)
        spec = commit_resource_specs(b,40,4)
        descriptors = a(spec['descriptors'][0],ix)
        host_descriptors = torch.empty(spec['host_descriptors'][0],dtype=ix,device='cpu')
        self.commit = Commit(self.result,past.pos_dev,past.pt.table,descriptors,host_descriptors,
            self.checked,windows=[(past.windows[l],t.pending_kv) for l,t in enumerate(self.layers)],
            sources=[(past.sources[l],self.sources[l]) for l in c['kv_source_layers']],
            page_tokens=past.page_tokens,max_seq=past.max_seq,library=o.paths['window'])
        descriptors.copy_(host_descriptors)
        self.workspace_bytes = max([self.head.workspace_bytes,self.draft.workspace_bytes]+
            [x.workspace_bytes for x in self.layers]+[x.workspace_bytes for x in self.sources.values()]+
            [x.workspace_bytes for x in self.engrams.values()])

    def stage(self, tokens, slots, starts, active, histories, temps=None):
        self.slots, self.starts = slots, starts
        if temps is None:
            temps = (0.,)*self.batch
        if len(temps) != self.batch or any(not 0. <= float(t) < 100. for t in temps):
            raise ValueError('one finite non-negative temperature per batch row')
        for key,value in (('tokens',tokens),('slots',slots),('start',starts),
                          ('active',active),('temp',temps)):
            self.host_views[key][:] = value
        meta = self.host_views['meta']
        meta.fill(0)
        meta[:,0],meta[:,1],meta[:,2] = slots,starts,active
        for device,host in self.uploads:
            device.copy_(host,non_blocking=True)
        self.prefetch.stage(tokens,starts,histories)

    def seed(self):
        self.draft.seed(self.owner.workspace)

    def propose(self):
        # NativeDSpark owns the single prepare that refreshes IDs/phases after
        # rebatching. It does not publish seed KV; seed has its own prepare.
        self.draft.propose(self.owner.workspace)

    def verify(self):
        import torch
        o = self.owner
        native(o.libs['window'], 'dec_window_prepare',
               (self.meta,o.inv_freq,self.freqs,self.legacy_pre), self.batch)
        native(o.libs['window'], 'dec_embed_shard',
               (self.tokens,self.meta,o.weights['embed.weight'],self.hidden),
               self.batch,16160,o.weights.rank*16160)
        o.parallel.sum(self.hidden)
        for layer in self.layers:
            if layer.layer in (37,38,39):
                native(o.libs['norm'], 'dec_ds_features',
                       (layer.am['x'],self.features), self.batch,layer.layer-37)
            if layer.layer in self.engrams:
                self.prefetch.run_device(layer.layer, o.workspace)
            layer.execute(o.workspace, self.sources.get(layer.layer),
                          self.shared_stream, self.hc_stream, self.events, o)
        self.head.run(self.owner.workspace)
        stream = P(torch.npu.current_stream(self.owner.weights.device).npu_stream)
        _check(self.greedy_fn(stream,*self.greedy_args),'TP8 sample')
        _check(self.accept_fn(stream,*self.accept_args),'accept')

    def drain(self, *, completed_stream=None):
        for engram in self.engrams.values():
            engram.drain(completed_stream=completed_stream)

    def close(self):
        for name in ('graph','seed_graph','draft_graph'):
            graph = getattr(self,name)
            if graph is not None:
                graph.reset()
                setattr(self,name,None)
        for component in [self.draft,self.head,*reversed(self.layers),*reversed(tuple(self.sources.values())),
                          *reversed(tuple(self.engrams.values()))]:
            if component is not None:
                component.close()
        # Returned plans may outlive the runtime; drop tensors and closure cycles.
        self.__dict__.clear()
        self.graph = self.seed_graph = self.draft_graph = None
        self.head = self.draft = None
        self.layers, self.sources, self.engrams = [], {}, {}


class TargetLayer:
    """Fixed Q6 target-layer storage; all plans are built before graph capture.

    A runtime owns x/incoming/output, canonical Past, and the shared scratch
    dictionary. Layers execute serially; scratch must never be used concurrently.
    bind() partitions the caller workspace for concurrent expert branches.
    This is target attention+FFN, not Engram/source staging or a draft decoder.
    """
    def __init__(self, weights, config, *, layer, batch, parallel, past,
                 x, incoming, output, freqs, slots, start, active,
                 norm_library, attention_library, gemm_library, scratch,
                 compressed=None, allocator=None, prepare_weight=None, wo_b_tiling=None,
                 w8_prepared=None, prepare_routed_scale=None, query_latent=None):
        import torch
        from ops.decode.gemm import Matmul, W8Matmul
        from ops.decode.w4a8 import W4A8GroupedMatmul
        from ops.decode.wo_b import WoB
        from ops.decode.hc_project import HCProject
        if not 1 <= batch <= 4 or not 0 <= layer < config['n_layers']:
            raise ValueError('invalid fixed target layer')
        if parallel.world != 8 or x.device != parallel.device:
            raise ValueError('target layer requires its owning TP8 device')
        self.batch, self.config, self.parallel = batch, config, parallel
        self.norm, self.attention, self.gemm = norm_library, attention_library, gemm_library
        self.layer, self.projections = layer, []
        self.storage, self.scratch = [], scratch
        self.closed = False
        t, pairs, device = batch * 6, batch * 36, x.device
        bf, fp, i64 = torch.bfloat16, torch.float32, torch.int64
        def alloc(shape, dtype=bf, name=None):
            shape = tuple(shape)
            if name is not None and name in scratch:
                value = scratch[name]
                if tuple(value.shape) != shape or value.dtype != dtype or value.device != device:
                    raise ValueError('incompatible decode scratch: '+name)
            else:
                value = (torch.empty(shape, dtype=dtype, device=device) if allocator is None
                         else allocator(shape, dtype))
                if name is not None:
                    scratch[name] = value
            self.storage.append(value)
            return value
        def weight(name, dtype=None, shape=None):
            value = weights[name]
            if value.device != device or not value.is_contiguous():
                raise ValueError('weight device/layout mismatch: '+name)
            if dtype is not None and value.dtype != dtype:
                value = (value.to(dtype=dtype) if prepare_weight is None
                         else prepare_weight(name, dtype))
            if shape is not None:
                value = value.view(shape)
            self.storage.append(value)
            return value
        def keep(plan):
            self.projections.append(plan)
            return plan
        def linear(src, name, dst):
            w = weight(name+'.weight')
            if w8_prepared is not None and name in w8_prepared:
                dense = w8_prepared[name]
                if (w.dtype != torch.int8 or dense.shape != w.shape
                        or dense.dtype != bf or dense.device != device
                        or not dense.is_contiguous() or dense.data_ptr() % 32):
                    raise ValueError('invalid prepared W8 binding: '+name)
                return keep(Matmul(src, dense, dst))
            if w.dtype == torch.int8:
                sc = weight(name+'.scale', bf)
                projected = (alloc(tuple(dst.shape), name='w8_projected')
                             if dst.dtype == fp else None)
                return keep(W8Matmul(src, w, sc, dst, projected=projected))
            if w.dtype != src.dtype:
                raise ValueError('unexpected unquantized projection dtype: '+name)
            return keep(Matmul(src, w, dst))
        prefix = f'layers.{layer}.'
        def hc(kind, value):
            m = dict(x=value,
                stats=alloc((t,),fp,name='hc_stats'),
                z=alloc((t,24),fp,name='hc_z'),
                pre=alloc((t,4),fp), post=alloc((t,4),fp,name='hc_post'),
                comb=alloc((t,4,4),fp,name='hc_comb'),
                weight=weight(prefix+f'hc_{kind}_fn',fp,(24,20480)),
                scale=weight(prefix+f'hc_{kind}_scale',fp,(3,)),
                base=weight(prefix+f'hc_{kind}_base',fp,(24,)))
            return m, keep(HCProject(m['x'],m['weight'],m['z'],m['stats'],
                                     eps=config['hc_eps']))
        middle = alloc((t,20480),name='attention_residual')
        self.am, self.ah = hc('attn',x)
        self.fm, self.fh = hc('ffn',middle)
        normed = alloc((t,5120),name='normalized')
        value = alloc((t,5120),name='residual_value')
        def residual(src, pre, mix, dst, norm):
            return dict(x=src,pre=pre,post=mix['post'],comb=mix['comb'],
                norm_weight=weight(prefix+norm,fp),normed=normed,value=value,out=dst)
        self.ar = residual(x,incoming,self.am,middle,'attn_norm.weight')
        self.fr = residual(middle,self.am['pre'],self.fm,output,'ffn_norm.weight')
        self.pre = self.fm['pre']
        a = dict(qr=alloc((t,1280),name='qr'),
            qn=query_latent if query_latent is not None else alloc((t,1280),name='qn'),
            qraw=alloc((t,4096),name='qraw'), q=alloc((t,8,512),name='q'),
            kr=alloc((t,512),name='kr'), kn=alloc((t,512),name='kn'),
            krot=alloc((t,512),name='krot'), kv=alloc((t,512)),
            ao=alloc((t,8,512),name='ao'), inv=alloc((t,8,512),name='inv'),
            freq=freqs, ring=past.main_kv, slots=slots, start=start, active=active,
            q_norm=weight(prefix+'attn.q_norm.weight',fp),
            kv_norm=weight(prefix+'attn.kv_norm.weight',fp),
            sink=weight(prefix+'attn.attn_sink',fp))
        self.ratio = int(config['compress_ratios'][layer])
        if self.ratio:
            if compressed is None:
                raise ValueError('compressed target layer needs explicit source storage')
            a.update(compressed)
        elif compressed is not None:
            raise ValueError('SWA target layer cannot bind compressed storage')
        self.a, self.pending_kv = a, a['kv']
        low = alloc((t,1024),name='output_low')
        local = alloc((t,5120),fp,name='attention_local')
        self.ap = (
            linear(normed,prefix+'attn.wq_a',a['qr']),
            linear(a['qn'],prefix+'attn.wq_b',a['qraw']),
            linear(normed,prefix+'attn.wkv',a['kr']),
            linear(a['inv'].view(t,4096),prefix+'attn.wo_a',low),
            keep(WoB(low,weight(prefix+'attn.wo_b.weight'),local,wo_b_tiling)))
        # Both expert branches write disjoint FP32 views of caller-owned storage.
        # The decode MoE module owns the single packed TP collective.
        self.moe_reduction = alloc((2,t,5120),fp,name='moe_reduction')
        mt = dict(x=normed,ids=alloc((t,6),i64,name='expert_ids'),
            probabilities=alloc((t,6),fp,name='expert_probability'),
            sorted_probability=alloc((pairs,),fp,name='sorted_probability'),
            inverse=alloc((pairs,),i64,name='route_inverse'),
            out=self.moe_reduction[0])
        router_in = alloc((t,5120),fp,name='router_in')
        logits = alloc((t,384),fp,name='router_logits')
        self.route = keep(Matmul(router_in,weight(prefix+'ffn.gate.weight',fp),logits))
        self.bias = weight(prefix+'ffn.gate.bias',fp)
        rows = alloc((pairs,5120),name='expert_rows')
        hidden = alloc((pairs,576),name='expert_hidden')
        activated = alloc((pairs,288),name='expert_activated')
        down = alloc((pairs,5120),name='expert_down')
        ends = alloc((384,),i64,name='expert_ends')
        if prepare_routed_scale is None:
            raise ValueError('target requires owner-shared encoded routed scales')
        projections = []
        for name,src,dst,kp in (('w13',rows,hidden,5120),('w2',activated,down,320)):
            stem = prefix+'ffn.'+name
            q = alloc((pairs,kp),torch.int8,name='expert_quantized_'+name)
            counts = alloc((384,),i64,name='expert_counts_'+name)
            projections.append(keep(W4A8GroupedMatmul(src,weight(stem+'.weight'),
                prepare_routed_scale(stem+'.scale'),weight(stem+'.hp_bias'),ends,dst,
                quantized=q,counts=counts,
                raw_quantized=alloc(tuple(src.shape),torch.int8,name='expert_raw_'+name),
                token_scale=alloc((pairs,),fp,name='expert_token_scale_'+name),
                projected=alloc(tuple(dst.shape),torch.float16,name='expert_projected_'+name))))
        self.routed = tuple(projections)
        sg = alloc((t,288),name='shared_gate')
        su = alloc((t,288),name='shared_up')
        sa = alloc((t,288),name='shared_activation')
        st = self.moe_reduction[1]
        self.shared = tuple(linear(src,prefix+'ffn.shared_experts.'+n,dst)
            for n,src,dst in [('w1',normed,sg),('w3',normed,su),('w2',sa,st)])
        self.moe = mt
        self.modulus, self.pad = past.ring, past.pad

    @property
    def workspace_bytes(self):
        main = max(p.workspace_bytes for p in self.projections if p not in self.shared)
        side = max(p.workspace_bytes for p in self.shared)
        return max(32, (main+31)//32*32) + max(32, side)

    def execute(self, workspace, source, side_stream, hc_stream, events, owner):
        import torch
        main_size = max(p.workspace_bytes for p in self.projections if p not in self.shared)
        split = max(32, (main_size+31)//32*32)
        if workspace.numel() < self.workspace_bytes:
            raise ValueError('target dual-stream workspace is too small')
        main_workspace, side_workspace = workspace[:split], workspace[split:]
        for op, mix, residual, attention in ((self.ah,self.am,self.ar,True),
                                           (self.fh,self.fm,self.fr,False)):
            main = torch.npu.current_stream(self.parallel.device)
            fork, ready = events[:2]
            fork.record(main)
            with torch.npu.stream(hc_stream):
                hc_stream.wait_event(fork)
                op(workspace)
                native(self.norm, 'dec_hc_scale_gates',
                       tuple(mix[k] for k in ('z','stats','scale','base','pre','post','comb')),
                       self.batch, self.config['hc_sinkhorn_iters'], float(self.config['hc_eps']))
                ready.record(hc_stream)
            try:
                native(self.norm, 'dec_hc_norm',
                       tuple(residual[k] for k in ('x','pre','norm_weight','normed')),
                       self.batch, float(self.config['norm_eps']))
                if attention:
                    if source is not None:
                        source.run(workspace, owner.rope, owner.libs['attention'])
                    self.attend(main_workspace, source)
                    self.parallel.sum(self.ap[4].storage[2])
                    native(self.norm, 'dec_tp_cast', (self.ap[4].storage[2],residual['value']), self.batch)
                else:
                    self.experts(main_workspace, side_workspace, side_stream, events[2:])
            finally:
                main.wait_event(ready)
            native(self.norm, 'dec_hc_expand',
                   tuple(residual[k] for k in ('value','x','post','comb','out')), self.batch)

    def attend(self, workspace, source):
        a, b, eps = self.a, self.batch, float(self.config['norm_eps'])
        p, ws = self.ap, workspace
        if source is None:
            p[0](ws)
            native(self.norm, 'dec_rms_norm_f32', (a['qr'],a['q_norm'],a['qn']), b,1280,eps)
        p[1](ws)
        native(self.norm, 'dec_rope', (a['qraw'],a['freq'],a['q']), b,8,512,0)
        p[2](ws)
        native(self.norm, 'dec_rms_norm_f32', (a['kr'],a['kv_norm'],a['kn']), b,512,eps)
        native(self.norm, 'dec_rope', (a['kn'],a['freq'],a['krot']), b,1,512,0)
        native(self.attention, 'dec_kv_qdq', (a['krot'],a['kv']), b)
        args = tuple(a[k] for k in ('q','ring','kv','slots','start','active','sink','ao'))
        dims = (b,8,a['ring'].shape[0],self.modulus,self.pad)
        if self.ratio:
            native(self.attention, 'dec_csa_q6', args+tuple(a[k] for k in
                   ('bank','table','compressed_pending','ids')), *dims,
                   a['bank'].shape[0],a['table'].shape[1],a['bank'].shape[1],self.ratio)
        else:
            native(self.attention, 'dec_swa_q6', args, *dims)
        native(self.norm, 'dec_rope', (a['ao'],a['freq'],a['inv']), b,8,512,1)
        p[3](ws)
        p[4](ws)

    def experts(self, workspace, side_workspace, side_stream, events):
        import torch
        b, t, ws = self.batch, self.moe, workspace
        native(self.norm, 'dec_gate_cast', (t['x'],self.route.storage[0]), b)
        self.route(ws)
        native(self.norm, 'dec_route', (self.route.storage[2],self.bias,t['ids'],t['probabilities']),
               b,1.0,float(self.config['route_scale']))
        main = torch.npu.current_stream(self.parallel.device)
        fork, done = events
        fork.record(main)
        with torch.npu.stream(side_stream):
            side_stream.wait_event(fork)
            gate, up, down = self.shared
            gate(side_workspace)
            up(side_workspace)
            native(self.norm, 'dec_swiglu', (gate.storage[2],up.storage[2],down.storage[0]), b,288,10.0)
            down(side_workspace)
            done.record(side_stream)
        first, second = self.routed
        native(self.norm, 'dec_dispatch', (t['x'],t['ids'],t['probabilities'],first.storage[0],
               t['sorted_probability'],t['inverse'],first.storage[3]), b)
        first(ws)
        native(self.norm, 'dec_routed_act', (first.storage[2],t['sorted_probability'],second.storage[0]), b)
        second(ws)
        native(self.norm, 'dec_routed_combine', (second.storage[2],t['inverse'],t['out']), b)
        main.wait_event(done)
        self.parallel.sum(self.moe_reduction)
        native(self.norm, 'dec_moe_finish', (t['out'],self.shared[2].storage[2],self.fr['value']), b)

    def close(self):
        if not self.closed:
            for op in reversed(self.projections):
                op.close()
            self.closed = True
