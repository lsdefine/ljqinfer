"""Fixed-resource target head, not a sampler or a Past transaction.

The builder supplies all storage; DeviceWeights preserves checkpoint dtypes.
Construction copies/casts norm.weight into caller FP32 storage once, then creates a
repeatable decode Matmul. Bind workspace before warmup/capture. run() submits
HC collapse/RMS and vocabulary GEMM on the current NPU stream, allocating no
storage. Output is borrowed FP32 [B*6,16160], NOT a gathered vocabulary.
"""
import ctypes as C
from ops.queued import queued
import math

import torch

from ops.decode.gemm import Matmul
from model.decode_ops import native
from ops.decode.wo_b import WoB, DraftVocab, MarkovHead

P, U, F = C.c_void_p, C.c_uint32, C.c_float


class NativeHead:
    """One TP8 shard; all tensors and the kernel library outlive captured graphs.

    hidden is BF16 [B*6,20480], pre is compact FP32 [B*6,4] from the last
    FFN (not its POST, nor the legacy interleaved PRE|POST [B*6,8]).
    norm_weight is FP32 [5120] startup scratch. normalized is BF16 [B*6,5120].
    logits is FP32 [B*6,16160]. Inputs and writable resources must be disjoint.
    No dependence on the obsolete native arena, linear library or prefill code.
    """

    def __init__(self, weights, *, batch, hidden, pre, norm_weight, normalized,
                 logits, norm_library, eps, projection_input=None):
        self.projection, self.closed = None, False
        if (not isinstance(batch, int) or not 1 <= batch <= 4
                or not math.isfinite(eps) or not 0 < eps <= 3.402823466e38
                or weights.rank not in range(8)):
            raise ValueError('head requires B1..4Q6, TP8 weights and finite epsilon')
        self.weights, self.library = weights, norm_library
        self.batch, self.device = batch, hidden.device
        self.vocab_offset = weights.rank * 16160
        self.vocab_size = 129280
        self.logits, self.normalized = logits, normalized
        rows = batch * 6
        norm, weight = weights['norm.weight'], weights['head.weight']
        if norm.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('head norm.weight must be BF16 or FP32')
        if weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('head.weight must be BF16 or FP32')
        if (weight.dtype == torch.float32) != (projection_input is not None):
            raise ValueError('FP32 head requires caller-owned projection_input only')
        checks = (
            ('hidden', hidden, (rows,20480), torch.bfloat16),
            ('pre', pre, (rows,4), torch.float32),
            ('norm_weight', norm_weight, (5120,), torch.float32),
            ('normalized', normalized, (rows,5120), torch.bfloat16),
            ('logits', logits, (rows,16160), torch.float32),
            ('norm.weight', norm, (5120,), norm.dtype),
            ('head.weight', weight, (16160,5120), weight.dtype))
        if projection_input is not None:
            checks += (('projection_input', projection_input, (rows,5120), torch.float32),)
        for name, t, shape, dtype in checks:
            if (tuple(t.shape) != shape or t.dtype != dtype
                    or t.device != self.device or self.device.type != 'npu'
                    or self.device.index != weights.rank or not t.is_contiguous()
                    or t.data_ptr() % 32):
                raise ValueError('invalid fixed head storage: ' + name)
        self.storage = tuple(t for _, t, _, _ in checks)
        for i, a in enumerate(self.storage):
            for b in self.storage[:i]:
                if self._overlap(a, b):
                    raise ValueError('head storage overlaps')
        self.norm_inputs = (hidden, pre, norm_weight, normalized)
        self.projection_input, self.eps = projection_input, eps
        try:
            self.projection = Matmul(normalized if projection_input is None else projection_input, weight, logits)
            self.workspace_bytes = self.projection.workspace_bytes
            # Startup only: caller orders this stream before any graph replay.
            # copy_ casts directly into supplied storage; no .float() temporary.
            with torch.no_grad():
                norm_weight.copy_(norm)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _overlap(a, b):
        return (a.data_ptr() < b.data_ptr()+b.numel()*b.element_size()
                and b.data_ptr() < a.data_ptr()+a.numel()*a.element_size())

    def run(self, workspace):
        """Return local logits, without collective, accept or canonical writes."""
        if self.closed:
            raise RuntimeError('bind head workspace before execution')
        # Matmul also uses the current device/stream: fail instead of splitting
        # this dependent pair across devices if the worker lost its context.
        if torch.npu.current_device() != self.device.index:
            raise RuntimeError('head worker must select its owning NPU')
        native(self.library, 'dec_hc_norm', self.norm_inputs,
               self.batch, self.eps, check_status=True)
        if self.projection_input is not None:
            native(self.library, 'dec_gate_cast',
                   (self.normalized, self.projection_input), self.batch,
                   check_status=True)
        return self.projection(workspace)

    def close(self):
        """Caller destroys graphs and drains their streams before closing."""
        if self.projection is not None:
            self.projection.close()
            self.projection = None
        self.closed = True



def dspark_resource_specs(batch, weights):
    """Pure shape query: caller allocates every returned (shape, torch.dtype).

    No weights copied, no device calls. Scratch is reusable ONLY sequentially.
    Main input features are BF16 [B*6,15360], concatenated collapsed target
    layers 37,38,39 in that order (not 4-copy HC residuals). All six rows are
    computed; result.accepted masks publication. result is validated commit
    metadata int64[B,8] = slot,verify_start,accepted,bonus,active,error,0,0.
    seed positions are verify_start+q+1; draft positions start+accepted+1..+5.
    Ring slot 0 must already be initialized by the initial prefill handoff.
    """
    if type(batch) is not int or not 1 <= batch <= 4:
        raise ValueError('DSpark supports B1..4Q6')
    r, n = batch*6, batch*18
    bf, fp, ix = torch.bfloat16, torch.float32, torch.int64
    specs = {}
    def add(name, shape, dtype=bf):
        specs[name] = (tuple(shape), dtype)
    for name in ('seed_freq','draft_freq'): add(name,(r,64),fp)
    add('inv',(32,),fp)
    add('ids',(batch,6),ix)
    for name in ('main_projected','main','x','attention_out','moe_out','head_norm'):
        add(name,(r,5120))
    for name in ('h0','h1'): add(name,(r,20480))
    for name in ('pre0','pre1','pre2','post'): add(name,(r,4),fp)
    add('comb',(r,16),fp)
    add('flat',(r,20480),fp);add('stats',(r,),fp);add('z',(r,24),fp)
    add('router_x',(r,5120),fp);add('router_logits',(r,128),fp)
    add('route_ids',(r,3),ix);add('prob',(r,3),fp)
    add('sorted_prob',(n,),fp);add('inverse',(n,),ix);add('ends',(128,),ix)
    add('routed_rows',(n,5120))
    add('quantized13',(n,5120),torch.int8);add('quantized2',(n,320),torch.int8)
    add('counts13',(128,),ix);add('counts2',(128,),ix)
    add('raw13',(n,5120),torch.int8);add('raw2',(n,288),torch.int8)
    add('token_scale13',(n,),fp);add('token_scale2',(n,),fp)
    add('projected13',(n,576),torch.float16);add('projected2',(n,5120),torch.float16)
    add('routed_hidden',(n,576));add('routed_act',(n,288));add('routed_down',(n,5120))
    for name in ('routed_sum','shared_sum','attention_sum'): add(name,(r,5120),fp)
    for name in ('shared_gate','shared_up','shared_act'): add(name,(r,288))
    add('qa',(r,1280));add('qn',(r,1280));add('query',(r,4096));add('qrope',(r,4096))
    for name in ('kv','krope','seed_kv','seed_rope'): add(name,(r,512))
    add('attended',(r,4096));add('unrope',(r,4096));add('oa',(r,1024))
    add('logits',(r,16160),fp)
    add('bias',(r,16160),fp)
    add('markov_embed',(batch,256))
    add('markov_bias',(batch,16160),fp)
    add('greedy_partial',(r,256,2),fp);add('greedy_pair',(r,2),fp)
    add('greedy_gathered',(8,r,2),fp);add('greedy_ids',(batch,6),ix)
    # Per-layer weights are expanded ONCE at construction, never in hot GEMMs.
    # Routed projections borrow prepared NZ weights; no runtime expansion bank.
    for name, w in weights.data.items():
        if not name.startswith('mtp.') or '.ffn.w13.' in name or '.ffn.w2.' in name: continue
        if name.endswith('.weight') and w.dtype == torch.int8:
            add('dense:'+name,tuple(w.shape))
        elif any(name.endswith(s) for s in ('.attn_norm.weight','.ffn_norm.weight',
                 '.q_norm.weight','.kv_norm.weight','.main_norm.weight','.norm.weight',
                 '.attn_sink','.hc_attn_fn','.hc_attn_scale','.hc_attn_base',
                 '.hc_ffn_fn','.hc_ffn_scale','.hc_ffn_base',
                 '.gate.weight','.gate.bias')):
            add('fp:'+name,tuple(w.shape),fp)
    # Canonical Markov tables are already vocabulary-sharded; bind in place.
    return specs




class NativeDSpark:
    """Fixed-resource three-stage / five-proposal DSpark (greedy only).

    Public lifecycle: specs -> allocate -> construct -> bind(workspace) ->
    seed() -> propose(). run() is seed()+propose(). No dynamic tensor creation
    in these three execution methods. Constructors enqueue weight conversion;
    caller must order/drain the construction stream before capture.

    `windows` is (past.windows[40], [41], [42]), canonical WindowPast objects;
    main_kv is BF16 [slots,pad+ring,512], ring>=134. No second Past, no
    positional increment here. seed() publishes ONLY accepted target features;
    rejected target rows and ALL provisional draft KV remain private scratch.
    Call seed AFTER successful target commit using its checked metadata. It is
    idempotent, but must precede target scratch reuse. Initial prompt history
    (right-shifted features) must be seeded by the owner before first propose.
    propose() consumes the rings as-is, and returns borrowed int64[B,5] view;
    ids is borrowed int64[B,6] = bonus followed by five proposals, ready for
    the next Q6 verify. Inactive/error/duplicate slots produce -1 IDs.

    parallel.sum(tensor) is the owner's allocation-free TP8 SUM on current
    stream. comm/all_gather are native HCCL handles used by existing greedy
    leaf. Every rank must run identical call order. Only B1..4 is supported.
    Confidence/temperature sampling are intentionally outside this greedy ABI.
    """
    def __init__(self, weights, *, batch, features, result, windows, resources,
                 norm_library, window_library, opapi, parallel, comm, all_gather,
                 config, wo_b_tiling, draft_vocab_tiling, markov_tiling, draft_down_tiling,
                 prepare_routed_scale):
        self.operators = {}
        self.closed = False
        self.weights,self.resources,self.windows=weights,resources,tuple(windows)
        self.libs=(norm_library,window_library,opapi)
        self.parallel,self.device=parallel,features.device
        self.batch,self.rank=batch,weights.rank
        self.features,self.result=features,result
        self.comm,self.all_gather=comm,all_gather
        self.wo_b_tiling=wo_b_tiling
        self.draft_vocab_tiling=draft_vocab_tiling
        self.markov_tiling=markov_tiling
        self.draft_down_tiling=draft_down_tiling
        if not callable(prepare_routed_scale):
            raise ValueError("draft requires owner-shared encoded routed scales")
        self.prepare_routed_scale=prepare_routed_scale
        if (parallel.world!=8 or parallel.rank!=self.rank or parallel.device!=self.device
                or self.device.type!='npu' or self.device.index!=self.rank
                or len(self.windows)!=3 or not comm or not all_gather):
            raise ValueError('DSpark requires owning NPU, TP8 and three canonical windows')
        expected={'dim':5120,'hc_mult':4,'n_heads':64,'q_lora_rank':1280,
                  'head_dim':512,'rope_head_dim':64,'o_groups':8,'o_lora_rank':1024,
                  'dspark_block_size':5,'dspark_noise_token_id':128799,
                  'n_mtp_layers':3,'dspark_n_routed_experts':128,
                  'dspark_n_activated_experts':3,'dspark_markov_rank':256}
        for k,v in expected.items():
            if config[k]!=v: raise ValueError('unsupported DSpark config: '+k)
        if config['dspark_target_layer_ids']!=[37,38,39]: raise ValueError('target feature order')
        if config['swiglu_limit']!=10: raise ValueError('draft leaf SwiGLU limit is 10')
        self.eps=float(config['norm_eps']);self.hc_eps=float(config['hc_eps'])
        if not all(math.isfinite(v) and v>0 for v in (self.eps,self.hc_eps)):
            raise ValueError('positive finite epsilon required')
        self.iters=int(config['hc_sinkhorn_iters'])
        if not 1<=self.iters<=100: raise ValueError('HC iteration count')
        self.slots=self.windows[0].main_kv.shape[0]
        specs=dspark_resource_specs(batch,weights)
        checks=[('features',features,(batch*6,15360),torch.bfloat16),
                ('result',result,(batch,8),torch.int64)]
        checks += [(k,resources[k],s,d) for k,(s,d) in specs.items()]
        for j,w in enumerate(self.windows):
            if w.ring<134 or w.pad<0: raise ValueError('canonical draft ring capacity')
            checks.append((f'window{j}',w.main_kv,(self.slots,w.pad+w.ring,512),torch.bfloat16))
        for k,t,s,d in checks:
            if tuple(t.shape)!=s or t.dtype!=d or t.device!=self.device or not t.is_contiguous() or t.data_ptr()%32:
                raise ValueError('DSpark resource mismatch: '+k)
        self.storage=tuple(t for _,t,_,_ in checks)
        for i,a in enumerate(self.storage):
            if any(NativeHead._overlap(a,b) for b in self.storage[:i]):
                raise ValueError('DSpark live resources must be disjoint')
        self.ids=resources['ids'];self.proposals=self.ids[:,1:]
        try:
            self._initialize(config)
            self.gate_temp, self.route_scale = float(config['gate_temp']), float(config['route_scale'])
            self._operators()
        except BaseException:
            self.close();raise

    def _leaf(self, name, tensors, scalars=()):
        native(self.libs[0], name, tensors, *scalars)

    def _weight(self, name):
        return self.resources.get('dense:'+name, self.resources.get('fp:'+name, self.weights[name]))

    def _initialize(self, c):
        r = self.resources
        for key, t in r.items():
            if key.startswith('fp:'):
                t.copy_(self.weights[key[3:]])
            elif key.startswith('dense:'):
                name = key[6:]
                self._leaf('dec_ds_w8_expand',
                    (self.weights[name],self.weights[name[:-7]+'.scale'],t),tuple(t.shape))
        r['inv'].copy_(torch.tensor([c['rope_theta']**(-2*j/64) for j in range(32)],dtype=torch.float32))

    def _operators(self):
        from ops.decode.w4a8 import W4A8GroupedMatmul
        from ops.decode.wo_b import DraftSharedDown
        r, ops = self.resources, self.operators
        ops['main'] = Matmul(self.features,self._weight('mtp.0.main_proj.weight'),r['main_projected'])
        for j in range(3):
            p = f'mtp.{j}'
            specs = [('seed','main','attn.wkv.weight','seed_kv'),
                ('hc_attn','flat','hc_attn_fn','z'),('hc_ffn','flat','hc_ffn_fn','z'),
                ('qa','x','attn.wq_a.weight','qa'),('qb','qn','attn.wq_b.weight','query'),
                ('kv','x','attn.wkv.weight','kv'),
                ('route','router_x','ffn.gate.weight','router_logits'),
                ('shared_gate','x','ffn.shared_experts.w1.weight','shared_gate'),
                ('shared_up','x','ffn.shared_experts.w3.weight','shared_up')]
            for name,x,w,y in specs:
                ops[j,name] = Matmul(r[x],self._weight(p+'.'+w),r[y])
            ops[j,'oa'] = Matmul(r['unrope'],self._weight(p+'.attn.wo_a.weight').view(1024,4096),r['oa'])
            ops[j,'ob'] = WoB(r['oa'],self._weight(p+'.attn.wo_b.weight'),r['attention_sum'],self.wo_b_tiling)
            for suffix,x,y in [('13','routed_rows','routed_hidden'),('2','routed_act','routed_down')]:
                stem = p+'.ffn.w'+suffix
                ops[j,'routed'+suffix] = W4A8GroupedMatmul(r[x],self.weights[stem+'.weight'],
                    self.prepare_routed_scale(stem+'.scale'),self.weights[stem+'.hp_bias'],r['ends'],r[y],
                    quantized=r['quantized'+suffix],counts=r['counts'+suffix],raw_quantized=r['raw'+suffix],
                    token_scale=r['token_scale'+suffix],projected=r['projected'+suffix])
            ops[j,'shared_down'] = DraftSharedDown(
                r['shared_act'],self._weight(p+'.ffn.shared_experts.w2.weight'),
                r['shared_sum'],self.draft_down_tiling)
        ops['head'] = DraftVocab(r['head_norm'],self.weights['mtp.2.head.weight'],r['logits'],self.draft_vocab_tiling)
        ops['markov'] = MarkovHead(
            r['markov_embed'],self.weights['mtp.2.markov_head.head.weight'],
            r['markov_bias'],self.markov_tiling)
        self.workspace_bytes = max(op.workspace_bytes for op in ops.values())

    def _rms(self,x,w,y,width,heads=1):
        self._leaf('dec_ds_rms',(x,w,y),(self.batch,heads,width,self.eps))

    def _rope(self,x,freq,y,heads=1,inverse=False):
        self._leaf('dec_rope',(x,freq,y),(self.batch,heads,512,int(inverse)))

    def _mix(self,h,j,which,workspace):
        r, p = self.resources, f'mtp.{j}'
        self._leaf('dec_hc_prepare',(h,r['flat'],r['stats']),(self.batch,self.eps))
        self.operators[j,'hc_'+which](workspace)
        pre = r['pre1' if which=='attn' else 'pre2']
        self._leaf('dec_hc_scale_gates',(r['z'],r['stats'],self._weight(p+'.hc_'+which+'_scale'),
            self._weight(p+'.hc_'+which+'_base'),pre,r['post'],r['comb']),
            (self.batch,self.iters,self.hc_eps))

    def _collapse(self,h,pre,w,y):
        self._leaf('dec_hc_norm',(h,pre,w,y),(self.batch,self.eps))

    def _expand(self,x,h,y):
        r = self.resources
        self._leaf('dec_hc_expand',(x,h,r['post'],r['comb'],y),(self.batch,))

    def _prepare(self):
        if self.closed:
            raise RuntimeError('closed DSpark')
        r = self.resources
        self._leaf('dec_ds_prepare',(self.result,r['inv'],r['seed_freq'],r['draft_freq'],r['pre0'],r['ids']),
                   (self.batch,self.slots))

    def seed(self, workspace):
        r, b = self.resources, self.batch
        self._prepare()
        self.operators['main'](workspace)
        self._rms(r['main_projected'],self._weight('mtp.0.main_norm.weight'),r['main'],5120)
        for j,w in enumerate(self.windows):
            self.operators[j,'seed'](workspace)
            self._rms(r['seed_kv'],self._weight(f'mtp.{j}.attn.kv_norm.weight'),r['kv'],512)
            self._rope(r['kv'],r['seed_freq'],r['seed_rope'])
            self._leaf('dec_ds_kv_fp8',(r['seed_rope'],),(b,))
            self._leaf('dec_ds_seed',(r['seed_rope'],self.result,w.main_kv),(b,self.slots,w.ring,w.pad))

    def propose(self, workspace):
        r, b = self.resources, self.batch
        self._prepare()
        self._leaf('dec_ds_lookup',(r['ids'],self.weights['mtp.0.embed.weight'],r['h0']),(b,self.rank,0,0))
        self.parallel.sum(r['h0'])
        for j,w in enumerate(self.windows):
            p, a = f'mtp.{j}', f'mtp.{j}.attn'
            self._collapse(r['h0'],r['pre0' if j==0 else 'pre2'],self._weight(p+'.attn_norm.weight'),r['x'])
            self._mix(r['h0'],j,'attn',workspace)
            self.operators[j,'qa'](workspace)
            self._rms(r['qa'],self._weight(a+'.q_norm.weight'),r['qn'],1280)
            self.operators[j,'qb'](workspace)
            self._rope(r['query'],r['draft_freq'],r['qrope'],8)
            self.operators[j,'kv'](workspace)
            self._rms(r['kv'],self._weight(a+'.kv_norm.weight'),r['krope'],512)
            self._rope(r['krope'],r['draft_freq'],r['kv'])
            self._leaf('dec_ds_kv_fp8',(r['kv'],),(b,))
            self._leaf('dec_ds_attention',
                       (r['qrope'],r['kv'],w.main_kv,self.result,
                        self._weight(a+'.attn_sink'),r['attended']),
                       (b,self.slots,w.ring,w.pad))
            self._rope(r['attended'],r['draft_freq'],r['unrope'],8,True)
            self.operators[j,'oa'](workspace)
            self.operators[j,'ob'](workspace)
            self.parallel.sum(r['attention_sum'])
            self._leaf('dec_tp_cast',(r['attention_sum'],r['attention_out']),(b,))
            self._expand(r['attention_out'],r['h0'],r['h1'])
            self._collapse(r['h1'],r['pre1'],self._weight(p+'.ffn_norm.weight'),r['x'])
            self._mix(r['h1'],j,'ffn',workspace)
            self._moe(j,workspace)
            self._expand(r['moe_out'],r['h1'],r['h0'])
        self._collapse(r['h0'],r['pre2'],self._weight('mtp.2.norm.weight'),r['head_norm'])
        self.operators['head'](workspace)
        for step in range(5):
            self._leaf('dec_ds_lookup',(r['ids'],self.weights['mtp.2.markov_head.embed.weight'],r['markov_embed']),
                       (b,self.rank,step,1))
            self.parallel.sum(r['markov_embed'])
            self.operators['markov'](workspace)
            self._leaf('dec_ds_bias',(r['logits'],r['markov_bias'],r['bias']),(b,step))
            self._greedy(step)
            self._leaf('dec_ds_chain',(r['greedy_ids'],r['ids'],self.result),(b,step,self.slots))
        return self.proposals

    def _moe(self,j,workspace):
        r, b, p = self.resources, self.batch, f'mtp.{j}.ffn'
        self._leaf('dec_gate_cast',(r['x'],r['router_x']),(b,))
        self.operators[j,'route'](workspace)
        self._leaf('dec_ds_route',(r['router_logits'],self._weight(p+'.gate.bias'),r['route_ids'],r['prob']),
                   (b,self.gate_temp,self.route_scale))
        self._leaf('dec_ds_dispatch',
                   (r['x'],r['route_ids'],r['prob'],r['routed_rows'],
                    r['sorted_prob'],r['inverse'],r['ends']),(b,))
        self.operators[j,'routed13'](workspace)
        self._leaf('dec_ds_routed_act',(r['routed_hidden'],r['sorted_prob'],r['routed_act']),(b,))
        self.operators[j,'routed2'](workspace)
        self._leaf('dec_ds_combine',(r['routed_down'],r['inverse'],r['routed_sum']),(b,))
        self.operators[j,'shared_gate'](workspace)
        self.operators[j,'shared_up'](workspace)
        self._leaf('dec_swiglu',(r['shared_gate'],r['shared_up'],r['shared_act']),(b,288,10.0))
        self.operators[j,'shared_down'](workspace)
        self.parallel.sum(r['routed_sum'])
        self.parallel.sum(r['shared_sum'])
        self._leaf('dec_moe_finish',(r['routed_sum'],r['shared_sum'],r['moe_out']),(b,))

    def _greedy(self,step):
        r = self.resources
        f = self.libs[1].dec_draft_greedy_tp8
        f.argtypes, f.restype = [P]*8+[U]*4, C.c_int
        f = queued(f, f.argtypes, check_status=True)
        rc = f(P(torch.npu.current_stream(self.device).npu_stream),
            *(P(r[n].data_ptr()) for n in ('bias','greedy_partial','greedy_pair','greedy_gathered','greedy_ids')),
            P(self.comm),P(self.all_gather),self.batch,self.rank,8,step)
        if rc:
            raise RuntimeError(f'draft greedy: {rc}')

    def close(self):
        for op in reversed(tuple(self.operators.values())):
            op.close()
        self.operators.clear()
        self.closed = True
