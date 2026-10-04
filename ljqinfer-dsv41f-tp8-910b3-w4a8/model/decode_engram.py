"""Decode-only host metadata and raw Engram row gathering.

Tensor objects are storage owners only. Cold metadata is copied to Python;
request hashing and byte gathers never call Torch/ATen or prefill functions.
NativeEngram binds the native decode device pipeline with caller-owned fixed
resources. See engram_resource_specs and NativeEngram for build/lifetime ABI.
"""
import ctypes as C
from ops.queued import queued

# Must match the compiled ops/decode/norm.cpp capacity ABI. B8 needs a rebuild.
Q, MAX_BATCH = 6, 4


class DecodeHash:
    def __init__(self, metadata):
        layout = metadata.layout
        if (layout.max_ngram_size, layout.n_heads, layout.head_dim) != (4, 8, 256):
            raise ValueError('decode requires the released 4-gram/8-head/256 layout')
        # Cold storage-to-control-plane conversion, not tensor computation.
        self.token_map = tuple(metadata.token_map.tolist())
        self.pad_id = int(metadata.pad_id)
        self.layers = tuple(layout.layer_ids)
        self.primes = tuple(tuple(tuple(int(p) for p in heads) for heads in layer)
                            for layer in layout.primes)
        self.multipliers = tuple(tuple(row) for row in metadata.multipliers.tolist())
        self.sizes = tuple(layout.num_embeddings)
        self.offsets = []
        if not self.token_map or len(self.multipliers) != len(self.layers):
            raise ValueError('incomplete hash metadata')
        if len(self.primes) != len(self.layers) or len(self.sizes) != len(self.layers):
            raise ValueError('incomplete bucket layout')
        for groups, multipliers, size in zip(self.primes, self.multipliers, self.sizes):
            if len(groups) != 3 or any(len(g) != 8 for g in groups) or len(multipliers) != 4:
                raise ValueError('invalid hash metadata dimensions')
            offsets, offset = [], 0
            for group in groups:
                for prime in group:
                    if prime <= 0:
                        raise ValueError('nonpositive hash modulus')
                    offsets.append(offset)
                    offset += prime
            if offset != size:
                raise ValueError('hash bucket total differs from host table')
            self.offsets.append(tuple(offsets))
        largest = max(max(self.token_map), self.pad_id)
        if min(self.token_map) < 0 or self.pad_id < 0 or any(
                m <= 0 or largest * m > (1 << 63) - 1
                for row in self.multipliers for m in row):
            raise ValueError('hash products must fit nonnegative signed int64')
        self.offsets = tuple(self.offsets)
        self.rank_columns = tuple(tuple(tuple(
            (col//8, self.primes[li][col//8][col%8], self.offsets[li][col])
            for col in range(rank*3, (rank+1)*3)) for rank in range(8))
            for li in range(len(self.layers)))

    def rows(self, layer, tokens, *, start, history_tokens=(), rank):
        """Six positions x three rank-owned hash columns; exact raw history.

        This text-only decode path has no masked tokens. Masked/multimodal
        windows must use a separately specified boundary, never silently pad.
        """
        if type(rank) is not int or not 0 <= rank < 8:
            raise ValueError('TP8 rank required')
        tokens, history_tokens = tuple(tokens), tuple(history_tokens)
        if type(start) is not int or start < 0 or len(history_tokens) != min(start, 3):
            raise ValueError('exact preceding raw token history required')
        if len(tokens) != Q:
            raise ValueError('decode Engram requires Q=6')
        raw = history_tokens + tokens
        if any(type(t) is not int or not 0 <= t < len(self.token_map) for t in raw):
            raise ValueError('token outside hash vocabulary')
        li = self.layers.index(layer)
        source = tuple(self.token_map[t] for t in raw)
        result = []
        for q in range(Q):
            pos = len(history_tokens) + q
            products = tuple((source[pos - shift] if pos >= shift else self.pad_id) * m
                             for shift, m in enumerate(self.multipliers[li]))
            x2 = products[0] ^ products[1]
            x3 = x2 ^ products[2]
            rolling = (x2, x3, x3 ^ products[3])
            result.append(tuple(rolling[group] % prime + offset
                                for group,prime,offset in self.rank_columns[li][rank]))
        return tuple(result)


class RawRows:
    """Borrow canonical host table storage; copy only selected raw bytes.

    Destination is caller-owned pinned storage: INT8[B*6*3,256] followed by
    FP32[B*6*3,8]. Caller drains its H2D stream before reusing or freeing it.
    """
    def __init__(self, table, expected_rows):
        weight, scale = table.weight, table.scale
        if (weight.device.type != 'cpu' or scale.device.type != 'cpu'
                or str(weight.dtype) != 'torch.int8'
                or str(scale.dtype) != 'torch.float32'
                or tuple(weight.shape) != (expected_rows, 256)
                or tuple(scale.shape) != (expected_rows, 8)
                or not weight.is_contiguous() or not scale.is_contiguous()):
            raise ValueError('canonical host Engram storage mismatch')
        # Retain the actual storages even if the caller replaces table fields.
        self.owner = (table, weight, scale)
        self.weight, self.scale = weight.data_ptr(), scale.data_ptr()
        self.count = expected_rows

    def copy(self, rows, destination, capacity):
        ids = tuple(i for request in rows for position in request for i in position)
        if not rows or any(len(request) != Q or any(len(pos) != 3 for pos in request)
                           for request in rows):
            raise ValueError('raw gather requires Bx6x3 hash IDs')
        if any(type(i) is not int or not 0 <= i < self.count for i in ids):
            raise ValueError('hash row outside canonical table')
        if (type(destination) is not int or destination <= 0
                or type(capacity) is not int or capacity < len(ids)*288):
            raise ValueError('invalid pinned row destination/capacity')
        if any(destination < src+size and src < destination+len(ids)*288
               for src,size in ((self.weight,self.count*256), (self.scale,self.count*32))):
            raise ValueError('gather destination aliases canonical host storage')
        scales = destination + len(ids) * 256
        for j, row in enumerate(ids):
            C.memmove(destination + j * 256, self.weight + row * 256, 256)
            C.memmove(scales + j * 32, self.scale + row * 32, 32)
        return len(ids)


def engram_resource_specs(*, batch):
    """Caller-owned, contiguous, 32-byte-aligned NPU storage (B1..B4, Q6).

    hidden/out may be the SAME complete tensor; all other ranges are disjoint.
    Serialized layers may share scratch, but only after their last consumer.
    No padding/active mask: every one of the B*6 supplied tokens participates.
    """
    if type(batch) is not int or not 1 <= batch <= MAX_BATCH:
        raise ValueError('Engram requires fixed B1..B4 and Q6')
    r = batch * Q
    return dict(packed=((batch*Q*3*288,), 'uint8'),
                rows=((r,768), 'bfloat16'),
                hidden=((r,4,5120), 'bfloat16'),
                hidden_f32=((r,4,5120), 'float32'),
                rotated=((r,4,5120), 'float32'),
                local_kv=((r,25600), 'bfloat16'),
                send=((r,25600), 'float32'),
                kv=((r,25600), 'float32'),
                out=((r,4,5120), 'bfloat16'))


def _tensor(value, shape, dtype, device, name):
    import torch
    if (tuple(value.shape) != tuple(shape) or value.dtype != getattr(torch, dtype)
            or value.device != device or device.type != 'npu'
            or not value.is_contiguous() or not value.data_ptr()
            or value.data_ptr() % 32):
        raise ValueError('invalid Engram storage: ' + name)
    return value


def _overlap(a, b):
    return (a.numel() and b.numel()
            and a.data_ptr() < b.data_ptr()+b.numel()*b.element_size()
            and b.data_ptr() < a.data_ptr()+a.numel()*a.element_size())


def _function(lib, name, args, *, device=False):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, C.c_int
    return queued(fn, args, name, check_status=True) if device else fn


def _check(status, operation):
    if status:
        raise RuntimeError(f'{operation} failed: {status}')


class NativeEngram:
    """Fixed-resource TP8 Engram, independent of NativeDecode's retired ABI.

    Build with actual DeviceWeights, DecodeHash, HostEngram and tensors from
    engram_resource_specs. prepared['qk'] is an explicitly budgeted FP32
    [4,5120] product q_weight.float()*k_weight.float(), NOT a BF16 product.
    No implicit conversion or fallback. The canonical FP32 rotation [32,32]
    is passed directly to Matmul, whose physical [N,K] weight means x@w.T.
    Only the query returns to the original basis; V is NOT rotated again.

    This object owns two GEMM descriptors and one ACL pinned host allocation.
    It BORROWS all device tensors, weights, norm/ACL libraries and parallel
    (world=8, rank, comm, lib.HcclAllReduce). They are retained until close.
    The communicator must remain live and every rank must submit the same
    collective order. Workspace is caller-owned and bound once, explicitly.

    Eager lifecycle: prepare(hashes) [CPU only] -> upload() [current stream] ->
    run() [same current stream] -> drain() before reuse. run() also performs
    upload when merely prepared. run_device() is the device-only entry for a
    caller-managed graph: upload outside capture, warm first, capture/replay on
    the SAME stream, and drain all replays/destroy graphs before close(). It
    cannot observe an external replay or storage rebinding; caller owns both.
    No device allocation, weight arithmetic, implicit synchronization, or
    framework tensor computation on the submission path. drain/close explicitly
    synchronize the submitted stream. Cross-stream use requires a new drained
    cycle; no hidden event, side stream or allocator lifetime assumptions.
    """
    def __init__(self, weights, layer, hasher, table, tensors, prepared, *,
                 batch, norm_library, acl, parallel, eps):
        import math
        from types import MappingProxyType
        specs = engram_resource_specs(batch=batch)
        if (type(layer) is not int or layer not in hasher.layers
                or not math.isfinite(eps) or not 0 < eps <= 3.402823466e38
                or C.c_float(eps).value <= 0):
            raise ValueError('invalid Engram layer/FP32 epsilon')
        if (type(weights.rank) is not int or not 0 <= weights.rank < 8
                or parallel.world != 8 or parallel.rank != weights.rank):
            raise ValueError('Engram requires matching TP8 weights/communicator')
        self.closed, self.state, self.operators = False, 'idle', {}
        self.host, self._stream, self.workspace = C.c_void_p(), None, None
        self.weights, self.layer, self.hasher = weights, layer, hasher
        self.batch, self.rank, self.eps = batch, weights.rank, eps
        self.norm_library, self.acl, self.parallel = norm_library, acl, parallel
        self.tensors = MappingProxyType(dict(tensors))
        self.prepared_weights = MappingProxyType(dict(prepared))
        t, device = self.tensors, tensors['hidden'].device
        self.device = device
        if device != weights.device:
            raise ValueError('Engram tensors and DeviceWeights device differ')
        for name, (shape, dtype) in specs.items():
            _tensor(t[name], shape, dtype, device, name)
        prefix = f'layers.{layer}.engram'
        retained = {suffix: weights[prefix+'.'+suffix] for suffix in
                    ('wkv.weight', 'q_weight', 'k_weight')}
        retained['rotation'] = weights['engram.rotation']
        for name in ('q_weight', 'k_weight'):
            _tensor(retained[name], (4,5120), 'bfloat16', device, name)
        wkv = _tensor(retained['wkv.weight'], (25600,768), 'bfloat16', device, 'wkv')
        rotation = _tensor(retained['rotation'], (32,32), 'float32', device, 'rotation')
        qk = _tensor(prepared['qk'], (4,5120), 'float32', device, 'qk')
        self.canonical_weights = MappingProxyType(retained)
        named = [(k,t[k]) for k in specs]
        for i, (name, a) in enumerate(named):
            for other, b in named[:i]:
                if ({name,other} == {'hidden','out'} and a.data_ptr() == b.data_ptr()):
                    continue
                if _overlap(a,b):
                    raise ValueError(f'Engram scratch aliases: {name}/{other}')
            if any(_overlap(a,b) for b in (*retained.values(),qk)):
                raise ValueError('Engram scratch aliases a weight')
        if any(_overlap(qk,b) for b in retained.values()):
            raise ValueError('prepared qk aliases canonical weights')
        self.raw = RawRows(table, hasher.sizes[hasher.layers.index(layer)])
        self.raw_bytes, self.nrows = batch*Q*3*288, batch*Q*3
        self.out = t['out']
        P, U, I = C.c_void_p, C.c_uint64, C.c_int32
        # Bind actual exports eagerly; missing libraries/ABIs fail at build.
        self._malloc = _function(acl, 'aclrtMallocHost', [C.POINTER(P), U])
        self._free = _function(acl, 'aclrtFreeHost', [P])
        self._copy = _function(acl, 'aclrtMemcpyAsync', [P,U,P,U,I,P], device=True)
        self._unpack = _function(norm_library, 'dec_engram_rows_i8', [P]*4+[C.c_uint32], device=True)
        self._widen = _function(norm_library, 'dec_engram_widen', [P]*3+[C.c_uint32]*2, device=True)
        self._gate = _function(norm_library, 'dec_engram_gate', [P]*6+[C.c_uint32,C.c_float], device=True)
        self._reduce = _function(parallel.lib, 'HcclAllReduce', [P,P,U,I,I,P,P], device=True)
        comm = parallel.comm
        self.comm = P(comm.value if isinstance(comm,P) else comm)
        if not self.comm.value:
            raise ValueError('Engram needs a live caller-owned TP8 communicator')
        self._ptr = {k:P(v.data_ptr()) for k,v in named}
        self._scale = P(t['packed'].data_ptr()+self.nrows*256)
        self._qk = P(qk.data_ptr())
        # Views and descriptors only at build, never in upload/run.
        from ops.decode.gemm import Matmul
        try:
            self.operators['projection'] = Matmul(t['rows'], wkv, t['local_kv'])
            self.operators['rotation'] = Matmul(t['hidden_f32'].view(-1,32), rotation,
                                            t['rotated'].view(-1,32))
            self.workspace_bytes = max(p.workspace_bytes for p in self.operators.values())
            _check(self._malloc(C.byref(self.host), self.raw_bytes), 'Engram pinned allocation')
            if not self.host.value:
                raise RuntimeError('Engram pinned allocator returned null')
        except BaseException:
            self.close()
            raise

    def _ready(self):
        if self.closed:
            raise RuntimeError('Engram must be live with bound workspace')

    def hashes(self, tokens, starts, histories):
        if self.closed:
            raise RuntimeError('Engram is closed')
        if any(len(v) != self.batch for v in (tokens,starts,histories)):
            raise ValueError('Engram metadata must cover the fixed batch')
        return tuple(self.hasher.rows(self.layer,row,start=start,
                     history_tokens=history,rank=self.rank)
                     for row,start,history in zip(tokens,starts,histories))

    def prepare(self, hashes):
        """CPU gather only. Reuse is forbidden until explicit drain succeeds."""
        self._ready()
        if self.state != 'idle':
            raise RuntimeError('drain Engram before preparing another window')
        if len(hashes) != self.batch:
            raise ValueError('Engram hashes must cover the fixed batch')
        self.raw.copy(hashes, self.host.value, self.raw_bytes)
        self.state = 'prepared'

    def _current_stream(self):
        import torch
        # Matmul also obtains CURRENT stream at call time; require its device.
        if torch.npu.current_device() != self.device.index:
            raise RuntimeError('select the Engram device before submission')
        value = torch.npu.current_stream(self.device).npu_stream
        if not value:
            raise RuntimeError('Engram requires a non-null current stream')
        return value

    def upload(self):
        self._ready()
        if self.state != 'prepared':
            raise RuntimeError('prepare Engram before upload')
        stream = self._current_stream()
        # Mark in-flight BEFORE submission: a partial failure still needs drain.
        self._stream, self.state = stream, 'failed'
        _check(self._copy(self._ptr['packed'], self.raw_bytes, self.host,
                         self.raw_bytes, 1, C.c_void_p(stream)), 'Engram raw H2D')
        self.state = 'uploaded'

    def run(self, workspace):
        self._ready()
        if self.state == 'prepared':
            self.upload()
        if self.state != 'uploaded':
            raise RuntimeError('Engram run requires a fresh upload')
        return self.run_device(workspace)

    def run_device(self, workspace):
        """Device-only submission/capture. Repeated calls reuse uploaded rows.

        External graph replay must stay on this stream and keep these rows,
        hidden/output/workspace, weights and communicator live. It is the
        caller's responsibility not to overwrite resources before completion.
        """
        self._ready()
        if self.state not in ('uploaded','submitted'):
            raise RuntimeError('upload Engram rows before device submission')
        stream = self._current_stream()
        if stream != self._stream:
            raise RuntimeError('Engram upload and device work require the same stream')
        self.state = 'failed'
        s, p, b = C.c_void_p(stream), self._ptr, self.batch
        _check(self._unpack(s,p['packed'],self._scale,p['rows'],b), 'Engram INT8 rows')
        self.operators['projection'](workspace)  # Local GEMM MUST round to BF16 before SUM.
        _check(self._widen(s,p['local_kv'],p['send'],b,25600), 'Engram local widening')
        _check(self._reduce(p['send'],p['kv'],b*Q*25600,4,0,self.comm,s),
               'Engram FP32 TP8 SUM')  # HCCL_DATA_TYPE_FP32=4, SUM=0.
        _check(self._widen(s,p['hidden'],p['hidden_f32'],b,20480), 'Engram hidden widening')
        self.operators['rotation'](workspace)  # h @ rotation.T, in FP32; V stays rotated.
        _check(self._gate(s,p['hidden'],p['rotated'],p['kv'],self._qk,p['out'],b,self.eps),
               'Engram gate')  # Native gate rounds the TP sum back to BF16.
        self.state = 'submitted'
        return self.out

    def drain(self, *, completed_stream=None):
        """Drain, or reuse a matching stream already synchronized by the owner."""
        if self.closed:
            return
        if self._stream is not None and self._stream != completed_stream:
            import torch
            # Drain PTA submissions as well as device work before host-buffer reuse.
            torch.npu.synchronize(self.device)
        self._stream, self.state = None, 'idle'

    def close(self):
        """Destroy graphs first. Does not free borrowed tensors or communicator."""
        if self.closed:
            return
        self.drain()  # On sync error retain ALL resources, allowing explicit retry.
        for plan in reversed(tuple(self.operators.values())):
            plan.close()
        self.operators.clear()
        if self.host.value:
            _check(self._free(self.host), 'Engram pinned free')
            self.host = C.c_void_p()
        self.closed = True


class EngramPrefetch:
    """Owner must drain consumers before stage/reuse and destroy graphs first.

    stage runs on the consumer stream. upload runs on an explicit stream;
    capture warmup uses the consumer, replay uses the preallocated side stream.
    CPU gather overlaps an already submitted graph; no background Python worker.
    """
    def __init__(self, engrams, flags, reset_event, acl):
        import torch
        if set(engrams) != set(flags):
            raise ValueError('one gate per Engram required')
        for flag in flags.values():
            if (flag.device.type != 'npu' or flag.dtype != torch.int64
                    or tuple(flag.shape) != (4,) or not flag.is_contiguous()
                    or flag.data_ptr() % 32):
                raise ValueError('gate requires aligned INT64[4] device storage')
        self.engrams, self.flags = dict(engrams), dict(flags)
        self.reset_event, self.acl = reset_event, acl
        self.ptrs = {k:C.c_void_p(v.data_ptr()) for k,v in flags.items()}
        self.write, self.wait = acl.aclrtValueWrite, acl.aclrtValueWait
        for fn in (self.write, self.wait):
            fn.argtypes = [C.c_void_p,C.c_uint64,C.c_uint32,C.c_void_p]
            fn.restype = C.c_int
        self.write = queued(self.write, self.write.argtypes, check_status=True)
        self.wait = queued(self.wait, self.wait.argtypes, check_status=True)
        self.inputs = None

    @staticmethod
    def check(rc):
        if rc:
            raise RuntimeError(f'Engram graph gate failed: {rc}')

    def stage(self, tokens, starts, histories):
        import torch
        stream = torch.npu.current_stream()
        for ptr in self.ptrs.values():
            self.check(self.write(ptr,0,0,C.c_void_p(stream.npu_stream)))
        # Producer must not publish 1 before the consumer has cleared old 1.
        self.reset_event.record(stream)
        self.inputs = tokens, starts, histories

    def upload(self, stream):
        import torch
        if self.inputs is None:
            raise RuntimeError('stage before Engram prefetch')
        with torch.npu.stream(stream):
            stream.wait_event(self.reset_event)
            s = C.c_void_p(stream.npu_stream)
            try:
                for layer, engram in self.engrams.items():
                    engram.prepare(engram.hashes(*self.inputs))
                    engram.upload()
                    self.check(self.write(self.ptrs[layer],1,0,s))
            except BaseException:
                # Unblock a submitted graph for teardown only. Caller must
                # fail the window, never commit outputs after a staging error.
                for ptr in self.ptrs.values():
                    self.check(self.write(ptr,1,0,s))
                raise
        self.inputs = None

    def run_device(self, layer, workspace):
        import torch
        s = C.c_void_p(torch.npu.current_stream().npu_stream)
        self.check(self.wait(self.ptrs[layer],1,1,s))
        return self.engrams[layer].run_device(workspace)
