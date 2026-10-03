"""Prefill block and dense MoE baseline; no decode dispatch or persistent state.

The routed(x, ids, probabilities) boundary includes EP dispatch/GEMM/combine.
Its result is complete on this rank; shared experts are independently TP-summed.
DenseRouted is for reduced synthetic models, not a full FP4 deployment fallback.
"""
import torch
from concurrent.futures import ThreadPoolExecutor

from ops.prefill import residual as r
from ops.decode.hc_pre import hc_pre, hc_pre_collapse
from model.graph_gate import RowGate

HC_GEMV_MAX_ROWS = 64  # decode/MTP-sized tiles only; see PrefillBlock.mix


def _gate():
    # Module, not extension: it routes bf16 to the fused kernel and the
    # explicit FP32 diagnostic chain to the float32 oracle.
    from ops.prefill import engram_gate
    return engram_gate


class DenseRouted:
    def __init__(self, prefix, n_experts, linear, limit):
        self.prefix, self.n, self.linear, self.limit = prefix, n_experts, linear, limit

    def __call__(self, x, ids, probabilities, reduce=True):
        out = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(self.n):
            token, choice = torch.where(ids == expert)
            if not len(token):
                continue
            p = f'{self.prefix}.experts.{expert}'
            z = x[token]
            y = r.swiglu(self.linear(p+'.w1', z), self.linear(p+'.w3', z),
                          self.limit, probabilities[token, choice, None])
            out.index_add_(0, token, self.linear(p+'.w2', y).float())
        return out


class MoEReduceBuffer:
    """One BF16 staging bank shared by every layer; capacity is fixed at build."""

    def __init__(self, rows, dim, device):
        self.bank = torch.empty(rows, dim, dtype=torch.bfloat16, device=device)

    def take(self, rows, dtype):
        if rows > self.bank.shape[0] or dtype != self.bank.dtype:
            raise ValueError('staging bank capacity/dtype')
        return self.bank[:rows]


class PrefillMoE:
    def __init__(self, layer, config, weights, linear, routed, *, reduce_shared=None, buffer=None,
                 route=None, side=None, side_linear=None):
        self.p, self.c, self.w = f'layers.{layer}.ffn', config, weights
        self.linear, self.routed, self.reduce_shared = linear, routed, reduce_shared
        self.buffer = buffer
        # Decode binds a side stream (see decode_build): the shared expert is
        # three skinny GEMVs that depend only on x, so they run concurrently
        # with the routed experts instead of queueing behind them.
        self.side, self.side_linear = side, side_linear
        # Router is bound by the builder, not chosen per call: prefill streams
        # thousands of rows (batched torch GEMM wins), decode always hands this
        # block one short window (ops.decode.route_gate, 2 launches vs ~10).
        self.route = route if route is not None else r.route

    def __call__(self, x, image_mask=None):
        p, c, w, lin = self.p, self.c, self.w, self.linear
        # self.route is bound once by the builder (see __init__): prefill keeps
        # the batched torch GEMM, decode gets ops.decode.route_gate. Both give
        # identical ids; prob differs by |d|<=3e-7.
        bias = w[p+'.gate.bias']
        if image_mask is not None:
            bias = torch.where(image_mask[:, None], w[p+'.gate.bias_vl'], bias)
        prob, ids = self.route(x, w[p+'.gate.weight'], bias,
            topk=c['n_activated_experts'], temperature=c['gate_temp'],
            scale=c['route_scale'], normalize=c['norm_topk_prob'], score=c['score_func'])
        fused = self.reduce_shared is not None
        s = p+'.shared_experts'
        if self.side is None:
            routed = self.routed(x, ids, prob, reduce=not fused)
            z = r.swiglu(lin(s+'.w1', x), lin(s+'.w3', x), c['swiglu_limit'])
            shared = lin(s+'.w2', z)
        else:
            # Fork/join inside the capture: the side branch is recorded into the
            # same graph, so a replay issues both branches and the join edge.
            cur = torch.cuda.current_stream()
            slin = self.side_linear
            self.side.wait_stream(cur)
            with torch.cuda.stream(self.side):
                gu = slin.fused((s+'.w1', s+'.w3'), x) if x.dim() == 2 else None
                z = r.swiglu_packed(gu, c['swiglu_limit'])
                if z is None:
                    z = r.swiglu(slin(s+'.w1', x), slin(s+'.w3', x), c['swiglu_limit'])
                shared = slin(s+'.w2', z)
            routed = self.routed(x, ids, prob, reduce=not fused)
            cur.wait_stream(self.side)
        if not fused:
            return (routed.float()+shared.float()).to(x.dtype)
        # One BF16 sum carries both partial sums; the model rounds here regardless.
        # torch.add(out=bf16) promotes to fp32, adds, rounds once: identical to
        # (routed.float()+shared.float()).to(bf16) but one launch instead of 3-4.
        total = self.buffer.take(x.shape[0], x.dtype)
        torch.add(routed, shared, out=total)
        self.reduce_shared(total)
        return total


class PrefillBlock:
    def __init__(self, layer, config, weights, attention, moe, *, engram=None, expand_workspace=None,
                 decode=False, side=None):
        self.layer, self.c, self.w = layer, config, weights
        self.attention, self.moe, self.engram = attention, moe, engram
        self.expand_workspace = expand_workspace
        # Decode takes the fused preamble: the mix gates and the collapse+RMS of
        # the previous sublayer's gates read the same residual and do not depend
        # on each other, so one launch does both (3 launches -> 2 per sublayer).
        self.decode = decode
        # A side stream for the gate GEMV: the gates feed the NEXT expand, so
        # this sublayer's own attention/MoE hides them (see head).
        self.side = side

    def prepare(self, h, *, slot, start, tokens, history_tokens=None, image_mask=None):
        if self.engram is None:
            return h
        kw = {} if history_tokens is None else {'history_tokens': history_tokens}
        out = self.engram(h, slot=slot, start=start, tokens=tokens, **kw)
        return out if image_mask is None else torch.where(image_mask[:, None, None], h, out)

    def __call__(self, h, pre_mix, *, past, slot, start, scratch, image_mask=None):
        p, c, w = f'layers.{self.layer}', self.c, self.w
        kw = dict(norm_eps=c['norm_eps'], hc_eps=c['hc_eps'], iters=c['hc_sinkhorn_iters'])
        def head(kind, x, ppre, gain):
            base = p+'.hc_'+kind
            args = (w[base+'_fn'], w[base+'_scale'], w[base+'_base'])
            if self.decode and self.side is not None:
                # Only the collapse+RMS is on the critical path into
                # attention/MoE; the gates feed the NEXT expand and nothing
                # here waits on them, so they fork onto the side stream and
                # the sublayer's own GEMMs cover them (6.1us of 10.6
                # measured in a graph on an A100).
                cur = torch.cuda.current_stream()
                self.side.wait_stream(cur)
                with torch.cuda.stream(self.side):
                    pre, post, comb = hc_pre(x, *args, **kw)
                return pre, post, comb, r.collapse_norm(x, ppre, gain, c['norm_eps'])
            if self.decode:
                return hc_pre_collapse(x, *args, ppre, gain, **kw)
            # hc_pre is the decode GEMV (a few rows streaming the 1MB fn once per
            # program); on a 12K-row prefill chunk it took 84% of the CUDA time
            # (torch.profiler 2026-09-15).  Long chunks go through the cuBLAS GEMM.
            mixer = r.mixes if x.shape[0] > HC_GEMV_MAX_ROWS else hc_pre
            pre, post, comb = mixer(x, *args, **kw)
            return pre, post, comb, r.collapse_norm(x, ppre, gain, c['norm_eps'])
        # PRE belongs to the NEXT sublayer, not the sublayer producing it.
        apre, post, comb, x = head('attn', h, pre_mix, w[p+'.attn_norm.weight'])
        y = self.attention(x, past=past, slot=slot, start=start, scratch=scratch)
        self._join()
        h = self._expand(y, h, post, comb, 0)
        fpre, post, comb, x = head('ffn', h, apre, w[p+'.ffn_norm.weight'])
        y = self.moe(x) if image_mask is None else self.moe(x, image_mask=image_mask)
        self._join()
        return self._expand(y, h, post, comb, 1), fpre

    def _join(self):
        # The gates must land before the expand that consumes post/comb.
        if self.side is not None:
            torch.cuda.current_stream().wait_stream(self.side)

    def _expand(self, x, h, post, comb, index):
        if self.expand_workspace is None:
            return r.expand(x, h, post, comb)
        from ops.prefill.expand import expand
        out = self.expand_workspace[index]
        if len(x) > len(out):
            raise ValueError('prefill residual workspace capacity exceeded')
        return expand(x, h, post, comb, out=out[:len(x)])


_HOST_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="engram-host")


class PrefillEngram:
    """rows(slot,start,tokens) supplies ONLY this rank's dequantized hash rows.

    Hash history comes from token ownership outside the block, not another KV
    cache. Host table/prefetch and H2D may be replaced without changing the gate.
    """
    def __init__(self, layer, weights, linear, rows, *, eps, reduce_sum=None,
                 row_cap=None):
        self.p, self.w, self.linear = f'layers.{layer}.engram', weights, linear
        self.rows, self.eps, self.reduce_sum = rows, eps, reduce_sum
        # The row store is cut once, for the widest batch the engine will
        # ever replay; a narrower round stages into a prefix of it.
        self._row_cap, self._rows_store = row_cap, None
        self._pending = None
        # Staged mode: hashing and the H2D of the rows are host work that a
        # captured graph cannot contain, so stage() parks them in a fixed
        # device buffer that the graph body reads on every replay.
        self._rows_buf = None
        # Two staging buffers: the previous round's async H2D may still be
        # reading one while the prefetch thread fills the next.
        self._pins, self._turn = None, 0
        # Staging may run after the replay launched, so the body waits on a
        # gate instead of assuming the rows are already in the buffer.
        # The gate is one flag word, cut here so a served round only ever
        # raises and clears it.
        self._gate = RowGate(weights[self.p+'.q_weight'].device)

    def stage(self, *, slot, start, tokens, history_tokens=None, gated=False):
        """Gather this window's rows into the buffer the graph body reads.

        Addressing depends on the token window alone, so with gated=True the
        gather may run after the replay launched: the H2D rides a side stream
        and raises the gate the body waits on, which puts the gather under the
        layers that precede this module instead of in front of the launch.
        """
        kw = {} if history_tokens is None else {'history_tokens': history_tokens}
        ref = self.w[self.p+'.q_weight']
        rows = self._gathered(slot, start, tokens, kw)
        # The captured body reads _rows_buf by address.  Reallocating it when the
        # row count changes (b=1 window vs a b=4 batch) leaves every graph that
        # was captured earlier reading storage nobody writes again, so reserve
        # the batch capacity once and stage into a prefix view of it.
        store = self._rows_store
        if store is None:
            # Row width belongs to the table and is known once a gather
            # returned; capacity belongs to the engine and was told at
            # build time.  Both are fixed here, for good.
            cap = max(self._row_cap or 0, rows.shape[0])
            store = torch.empty((cap,) + tuple(rows.shape[1:]),
                                device=ref.device, dtype=ref.dtype)
            self._rows_store = store
        elif tuple(store.shape[1:]) != tuple(rows.shape[1:]):
            raise RuntimeError('engram row width changed after reservation')
        if rows.shape[0] > store.shape[0]:
            raise RuntimeError('engram stage needs %d rows, reserved %d'
                               % (rows.shape[0], store.shape[0]))
        self._rows_buf = store[:rows.shape[0]]
        if not gated:
            self._rows_buf.copy_(rows, non_blocking=rows.is_pinned())
            return
        with torch.cuda.stream(self._gate.stream):
            self._rows_buf.copy_(rows, non_blocking=rows.is_pinned())
            self._gate.raise_on(self._gate.stream)

    def release_gate(self):
        """Unblock a replay whose staging never raised: the round is failing out."""
        self._gate.release()

    def prefetch(self, *, slot, start, tokens, history_tokens=None):
        """Hash/gather depend on tokens only: run them while earlier layers compute."""
        kw = {} if history_tokens is None else {'history_tokens': history_tokens}
        self._pending = ((slot, start, tokens, history_tokens, self._mask_key(slot, start)),
                         _HOST_POOL.submit(self._rows_host, slot, start, tokens, kw))

    _PIN_ROWS = 64

    def _rows_host(self, slot, start, tokens, kw):
        # A batch hashes one window per request into one block of rows, so the
        # pinned staging area grows with the number of requests in flight.
        want = self._PIN_ROWS * (1 if isinstance(start, int) else len(start))
        if self._pins is None or self._pins[0].shape[0] < want:
            self._pins = [torch.empty(want, 256, dtype=torch.bfloat16).pin_memory()
                          for _ in range(2)]
        self._turn ^= 1
        return self.rows.rows_host(slot, start, tokens, out=self._pins[self._turn], **kw)

    def _mask_key(self, slot, start):
        key = getattr(self.rows, 'mask_key', None)
        if key is None:
            return None
        return key(slot) if isinstance(start, int) else tuple(key(sl) for sl in slot)

    def _gathered(self, slot, start, tokens, kw):
        # A future left by a raised forward holds another chunk's rows, and
        # replay() never prefetches: unkeyed, it would be served those rows.
        pending, self._pending = self._pending, None
        if pending is not None:
            key, future = pending
            if key[0] == slot and key[1] == start and key[2] is tokens \
                    and key[3] is kw.get('history_tokens') \
                    and key[4] == self._mask_key(slot, start):
                return future.result()
            future.cancel()
        return self._rows_host(slot, start, tokens, kw)

    def __call__(self, h, *, slot, start, tokens, history_tokens=None):
        kw = {} if history_tokens is None else {'history_tokens': history_tokens}
        if torch.cuda.is_current_stream_capturing():
            # A no-op unless capturing: only a captured body can be made to
            # wait.  Past it, the staged rows are the only rows there are:
            # gathering here instead would quietly serve the body a second,
            # ungated copy and hide a caller that forgot to stage.
            self._gate.wait_in_graph()
            rows = self._rows_buf
            if rows.shape[0] != h.shape[0]:
                raise RuntimeError('engram staged %d rows, body has %d'
                                   % (rows.shape[0], h.shape[0]))
        else:
            rows = self._gathered(slot, start, tokens, kw) \
                       .to(device=h.device, dtype=h.dtype)
        kv = self.linear(self.p+'.wkv', rows.flatten(-2))
        if self.reduce_sum is not None:
            self.reduce_sum(kv)
        return _gate().engram_gate(h, kv, self.w[self.p+'.q_weight'],
                                   self.w[self.p+'.k_weight'], self.eps)
