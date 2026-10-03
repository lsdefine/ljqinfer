"""DSpark drafter layer computation (decode-only).

Why this is not model.decode_layer with another prefix: a DSpark stage has no
CSA2 index/global path at all (no wq_b index head, no source pages, no
compress carry), and its window is written by the *main* path while the rows
that read it are *drafts*. The two also differ in causality -- see
ops.decode.dspark.draft_attend.  Sharing DecodeAttention would mean carrying
four dead branches through the hot 3-layer draft loop.
"""
import torch

from ops.prefill import residual as r
from ops.decode import v4k
from ops.decode.hc_pre import hc_pre
from ops.decode.quant_fused import fp8_roundtrip, swiglu
from ops.decode.dspark import confidence, draft_attend, markov_refine
from ops.decode.route_gate import route
from model.past import N_BACKBONE_LAYERS


def _rows(slot, x, what):
    """Split a block-major [B*t, ...] row stream into (B, t).

    Every drafter entry point takes one shape: a device slot vector [B] and
    the rows of all B requests concatenated block by block.  A single request
    is B = 1, not a second code path.
    """
    b = slot.numel()
    if x.shape[0] % b:
        raise ValueError('%s must be block-major [B*t, ...] for B slots' % what)
    return b, x.shape[0] // b


class DSparkAttention:
    """Attention for one MTP stage over the main-path sliding window."""

    def __init__(self, stage, config, weights, linear, freqs, *, world=1,
                 reduce_sum=None):
        self.stage, self.c, self.w, self.linear = stage, config, weights, linear
        self.freqs, self.world, self.reduce_sum = freqs, world, reduce_sum
        if world != 1 and reduce_sum is None:
            raise ValueError('tensor-parallel wo_b output needs a sum collective')
        self.p = f'mtp.{stage}.attn'

    def _freqs(self, start, n):
        """RoPE table rows for [start, start+n) of every request.

        ``start`` is always a [B] device cursor (a captured drafter reads it
        off past.pos_dev), so the table rows come back flattened in the same
        block-major order as the [B*n, ...] row stream.
        """
        idx = start.reshape(-1, 1) + torch.arange(n, device=self.freqs.device)
        return self.freqs.index_select(0, idx.reshape(-1))

    def _kv(self, x, start, t):
        """Latent KV for the t rows each request holds in x [B*t, dim]."""
        p, w = self.p, self.w
        latent = v4k.rms(self.linear(p + '.wkv', x), w[p + '.kv_norm.weight'],
                         self.c['norm_eps'])
        return fp8_roundtrip(v4k.rope_(latent, self._freqs(start, t)))

    def seed(self, main_x, *, window, slot, start, pos=None):
        """Publish the main path's KV rows; the drafts only ever read them.

        Called for prefill chunks too, where nothing else in this module runs:
        a draft is meaningless before the first sampled token, but the window
        must already carry history when the first draft arrives.
        """
        nb, t = _rows(slot, main_x, 'seed rows')
        kv = self._kv(main_x, start, t)
        # ``pos`` lets the caller drop rows (negative) while ``start`` still
        # gives every row its true rotary position.
        if pos is None:
            pos = start.reshape(-1, 1) + torch.arange(t, device=kv.device)
        # A prefill chunk can be longer than the ring; only its tail survives,
        # and attention inside the chunk reads forward-local KV anyway.
        ring = window.ring
        if t > ring:
            pos, kv, t = pos[:, -ring:], kv.view(nb, t, -1)[:, -ring:], ring
        window.scatter(slot, pos, kv.reshape(nb, t, -1))

    def __call__(self, x, *, window, slot, start):
        """x [T,dim] draft rows at absolute positions [start, start+T).

        The ring must already hold the main rows below ``start`` (seed()).
        Draft KV stays forward-local: it is never written to the ring, because
        the next step's window has to contain the *accepted* main row instead.
        """
        p, c, w, lin = self.p, self.c, self.w, self.linear
        nb, t = _rows(slot, x, 'draft rows')
        eps, d = c['norm_eps'], c['head_dim']
        f = self._freqs(start, t)
        qr = v4k.rms(lin(p + '.wq_a', x), w[p + '.q_norm.weight'], eps)
        q = v4k.rope_(lin(p + '.wq_b', qr).unflatten(-1, (c['n_heads'] // self.world, d)), f)
        kv = self._kv(x, start, t)
        # Gather the whole ring from the live cursor and let the attention mask
        # drop the rows that precede the history.
        pos = (start.reshape(-1, 1) - window.window
               + torch.arange(window.window, device=x.device))
        valid = pos >= 0
        history = window.gather(slot, pos)
        sink, scale = w[p + '.attn_sink'], d ** -.5
        # One visible set per request: the batch axis keeps the slots apart.
        o = draft_attend(q.view(nb, t, *q.shape[1:]), history,
                         kv.view(nb, t, -1), sink, scale=scale,
                         valid=valid).flatten(0, 1)
        o = v4k.rope_(o, f, inverse=True).flatten(-2)
        groups, rank = c['o_groups'] // self.world, c['o_lora_rank']
        wa = w[p + '.wo_a.weight']
        if wa.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            raise TypeError('wo_a must be prepacked dense or bound to grouped GEMM')
        from ops.decode.v4k import grouped_linear_bf16
        out = grouped_linear_bf16(o.reshape(len(o), groups, -1),
                                  wa.reshape(groups, rank, -1).to(o.dtype)).flatten(-2)
        out = lin(p + '.wo_b', out)
        if self.reduce_sum is not None:
            self.reduce_sum(out)
        return out


class DSparkMoE:
    """MoE for one MTP stage.

    Deliberately not model.prefill_block.PrefillMoE with another prefix: a draft
    block is 5 rows, so the fused shared+routed staging buffer that amortises one
    AllReduce over a 12k chunk has nothing to amortise here -- one collective on
    a 5-row tensor is already minimal.
    """

    def __init__(self, stage, config, weights, linear, routed, *, reduce_sum=None):
        self.p, self.c, self.w = f'mtp.{stage}.ffn', config, weights
        self.linear, self.routed, self.reduce_sum = linear, routed, reduce_sum

    def __call__(self, x):
        p, c, w, lin = self.p, self.c, self.w, self.linear
        prob, ids = route(x, w[p + '.gate.weight'], w[p + '.gate.bias'],
            topk=c['dspark_n_activated_experts'], temperature=c['gate_temp'],
            scale=c['route_scale'], normalize=c['norm_topk_prob'], score=c['score_func'])
        fused = self.reduce_sum is not None
        routed = self.routed(x, ids, prob, reduce=not fused)
        s = p + '.shared_experts'
        z = swiglu(lin(s + '.w1', x), lin(s + '.w3', x), c['swiglu_limit'])
        out = (routed.float() + lin(s + '.w2', z).float()).to(x.dtype)
        if fused:
            # Both partial sums ride one collective, as in the backbone.
            self.reduce_sum(out)
        return out


class DSparkBlock:
    """One MTP stage: mHC-mixed attention and MoE sublayers.

    Same mHC algebra as the backbone (PRE belongs to the *next* sublayer), but
    no engram, no residual workspace and no scratch: a 5-row draft has no
    capacity pressure to manage, and every allocation here is 5 rows wide.
    """

    def __init__(self, stage, config, weights, attention, moe):
        self.p, self.c, self.w = f'mtp.{stage}', config, weights
        self.stage = stage
        self.attention, self.moe = attention, moe

    def __call__(self, h, pre_mix, *, window, slot, start):
        p, c, w = self.p, self.c, self.w
        def mix(kind, x):
            base = p + '.hc_' + kind
            return hc_pre(x, w[base + '_fn'], w[base + '_scale'], w[base + '_base'],
                norm_eps=c['norm_eps'], hc_eps=c['hc_eps'], iters=c['hc_sinkhorn_iters'])
        apre, post, comb = mix('attn', h)
        x = r.collapse_norm(h, pre_mix, w[p + '.attn_norm.weight'], c['norm_eps'])
        y = self.attention(x, window=window, slot=slot, start=start)
        h = r.expand(y, h, post, comb)
        fpre, post, comb = mix('ffn', h)
        x = r.collapse_norm(h, apre, w[p + '.ffn_norm.weight'], c['norm_eps'])
        return r.expand(self.moe(x), h, post, comb), fpre


class DSparkDrafter:
    """The B1Q6 draft head: one accepted token plus the main model's hidden
    state produce block_size speculative tokens and their acceptance scores.

    Position bookkeeping is the subtle part. ``start`` is the absolute position
    of the *accepted* token, whose KV the main path owns; the draft rows it
    conditions occupy start+1 .. start+block_size and are never published to
    the ring, so a rejected window needs no rollback here -- only the one main
    row seeded per stage persists, and it is identical on every retry.
    """

    def __init__(self, config, weights, linear, blocks, *, embed, head, markov,
                 temperature=0.0, vocab_shard=None):
        if len(blocks) != config['n_mtp_layers']:
            raise ValueError('drafter needs exactly n_mtp_layers stages')
        self.c, self.w, self.linear = config, weights, linear
        self.blocks = list(blocks)
        self.embed, self.head, self.markov = embed, head, markov
        # Set when head/markov emit this rank's vocabulary slice rather than a
        # gathered row; the sampler then reduces across ranks itself.
        self.vocab_shard = vocab_shard
        self.temperature = temperature
        self.block_size = config['dspark_block_size']
        self.noise = config['dspark_noise_token_id']
        self.last = f"mtp.{len(self.blocks) - 1}"

    def _window(self, block, past):
        return past.windows[N_BACKBONE_LAYERS + block.stage]

    def _main_x(self, main_hidden):
        """Project the concatenated target-layer hiddens into the draft stream."""
        width = self.c['dim'] * len(self.c['dspark_target_layer_ids'])
        if main_hidden.shape[-1] != width:
            raise ValueError('main_hidden must concatenate the dspark target layers')
        return v4k.rms(self.linear('mtp.0.main_proj', main_hidden),
                       self.w['mtp.0.main_norm.weight'], self.c['norm_eps'])

    def seed(self, main_hidden, *, past, slot, start, pos=None):
        """Prefill path: publish the main row into every stage window, draft nothing.

        The drafter is never asked for tokens during prefill, but its windows
        still have to carry the prompt or the first real draft would attend a
        128-row hole.
        """
        main_x = self._main_x(main_hidden)
        for block in self.blocks:
            block.attention.seed(main_x, window=self._window(block, past),
                                 slot=slot, start=start, pos=pos)
        return main_x

    def __call__(self, main_hidden, token, *, past, slot, start, temperature=None):
        """Draft one window. ``token`` is the accepted token sitting at ``start``.

        Batched call: slot [B] on device, start [B], main_hidden [B, width] and
        token [B] draft B windows in one pass and return [B, ...] rows; every
        stage stays one GEMM because the draft rows of all requests share the
        row axis, and only attention splits them by slot.
        """
        c, nb, bs = self.c, slot.numel(), self.block_size
        main_x = self._main_x(main_hidden)
        # Only the accepted token is real; the remaining slots carry the noise id
        # the drafter was trained with, and get their content from attention.
        ids = token.new_full((nb * bs,), self.noise)
        ids[::bs] = token.reshape(-1)
        x = self.embed(ids)
        if x.shape != (nb * bs, c['dim']):
            raise ValueError('embedding must return [block_size, configured dim]')
        h = x[:, None, :].expand(-1, c['hc_mult'], -1).clone()
        pre = torch.zeros(h.shape[:2], device=h.device, dtype=torch.float32)
        pre[:, 0] = 1.
        for block in self.blocks:
            window = self._window(block, past)
            block.attention.seed(main_x, window=window, slot=slot, start=start)
            h, pre = block(h, pre, window=window, slot=slot, start=start + 1)
        # collapse without the norm: the head wants normalised rows, the
        # confidence head wants the raw residual (it was trained on it).
        hidden = r.collapse(h, pre)
        logits = self.head(v4k.rms(hidden, self.w[self.last + '.norm.weight'], c['norm_eps']))
        temp = self.temperature if temperature is None else temperature
        ids, logits, embeds = markov_refine(logits.view(nb, bs, -1), token.reshape(nb),
                                            self.markov, temperature=temp,
                                            shard=self.vocab_shard)
        score = confidence(hidden.view(nb, bs, -1), embeds,
                           self.w[self.last + '.confidence_head.proj.weight'])
        return ids, logits, score