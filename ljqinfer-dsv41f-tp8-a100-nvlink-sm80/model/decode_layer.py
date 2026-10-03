"""Decode-only layer computation for one speculative window of Q rows.

Split from model.prefill_layer deliberately:
  * prefill's project_global writes source.kv_state/score_state in place and
    compress() folds the remainder into that carry. A verify window can be
    rejected, so decode compresses onto a shadow carry, publishes nothing, and
    hands the rows to decode_attend as staged state; commit happens after the
    accept length is known.
  * no chunk loop and no read_global_only replay mode: a decode step sees the
    whole window at once.
"""
import torch
from ops.prefill import attention as a, residual as r, candidates
from ops.prefill import attention as a, residual as r
from ops.decode.quant_fused import fp4_roundtrip, fp8_roundtrip
from model.decode_attention import (decode_attend, window_positions,
                                    _requests, slot_rows)
from ops.decode import v4k


class DecodeAttention:
    """One CSA2 layer over Q provisional rows. Math mirrors PrefillAttention."""

    def __init__(self, layer, config, weights, linear, freqs, *, world=1,
                 reduce_sum=None):
        self.layer, self.c, self.w, self.linear = layer, config, weights, linear
        self.freqs, self.world, self.reduce_sum = freqs, world, reduce_sum
        if world != 1 and reduce_sum is None:
            raise ValueError('TP requires explicit sum collective')

    def stage_global(self, x, *, past, slot, start, scratch=None):
        """Stage compressed rows without mutating canonical committed state."""
        c, w, lin = self.c, self.w, self.linear
        eps = c['norm_eps']
        p = f'layers.{self.layer}.attn'
        view = past.views[self.layer]
        if view.mode != 'full':
            raise ValueError('only a KV source can stage global rows')
        source = past.sources[self.layer]
        cp = p + '.compressor'
        values = lin(cp + '.wkv', x.float() if view.ratio == 2 else x)
        scores = lin(cp + '.wgate', x.float()) if view.ratio == 2 else None
        pos0 = None if scratch is None or scratch.pos_t is None else scratch.pos_t
        ck, ik = self.compress_rows(
            source, slot, values, scores, start, view, x.dtype, pos0=pos0,
            write=False, rows_per_request=len(x) // len(_requests(slot)))
        if scratch is not None:
            scratch.pending[self.layer] = (values, scores, start, x.dtype)
        return ck, ik

    def compress_rows(self, source, slot, values, scores, start, view, dtype,
                      *, pos0=None, write=True, rows_per_request=None):
        """Fold projected rows and return each completed compressed group."""
        w, lin, eps = self.w, self.linear, self.c['norm_eps']
        p = f'layers.{self.layer}.attn'
        cp, ip = p + '.compressor', p + '.indexer'
        raw, origin = source.fold(slot, start, values, scores, pos0=pos0,
                                  write=write,
                                  rows_per_request=rows_per_request)
        if not len(raw):
            # No complete group yet; commit still publishes accepted raw rows.
            return None, None
        latent = v4k.rms(raw.to(dtype), w[cp + '.norm.weight'], eps)
        if pos0 is None:
            cf = self.freqs[origin * view.ratio:(origin + len(latent)) * view.ratio:view.ratio]
        else:
            # Compression and RoPE use the same live device alignment.
            nreq = len(_requests(slot))
            # Each request folds its own window, so the compressed rows of
            # request b start at that request's live window origin.
            p0 = pos0.view(nreq, -1)[:, :1]
            idx = (p0 - p0.remainder(view.ratio) + torch.arange(
                len(latent) // nreq, device=pos0.device)
                * view.ratio).reshape(-1)
            cf = self.freqs.index_select(0, idx)
        ik = fp4_roundtrip(v4k.rope_(v4k.rms(lin(ip + '.wk', latent),
                                             w[ip + '.k_norm.weight'], eps), cf),
                           block=32, e4m3_scale=False)
        ck = fp4_roundtrip(v4k.rope_(latent, cf), block=16, e4m3_scale=True)
        return ck, ik

    def __call__(self, x, *, past, slot, start, scratch):
        c, w, lin = self.c, self.w, self.linear
        eps, d = c['norm_eps'], c['head_dim']
        p = f'layers.{self.layer}.attn'
        view = past.views[self.layer]
        t = len(x)
        if scratch.pos_t is None:
            # One device position vector per step, shared by every layer and
            # by decode_attend.  It is derived from the *device* cursor, so a
            # captured graph replayed at a later step ropes at the live
            # position instead of the one frozen at capture time.  With several
            # requests in the call the rows arrive stacked, so each request
            # contributes its own cursor-relative run of the same length.
            scratch.pos_t = window_positions(past, slot,
                                             t // len(_requests(slot)), x.device)
        # Every layer used to re-select the same rotation rows; keep one per
        # (table, position vector) pair so a step selects them once.  The draft
        # and verify graphs own distinct position vectors, so they never share.
        _fk = (id(self.freqs), id(scratch.pos_t))
        f = scratch.rope_rows.get(_fk) if scratch.rope_rows is not None else None
        if f is None:
            f = self.freqs.index_select(0, scratch.pos_t)
            if scratch.rope_rows is not None:
                scratch.rope_rows[_fk] = f
        # wq_a and wkv read the same x, so they are one packed GEMM whose two
        # halves are normalized in a single launch (four launches become two).
        packed = lin.fused((p + '.wq_a', p + '.wkv'), x) if x.dim() == 2 else None
        split = None if packed is None else v4k.rms_split2(
            packed, w[p + '.q_norm.weight'], w[p + '.kv_norm.weight'], eps)
        if split is None:
            qr = v4k.rms(lin(p + '.wq_a', x), w[p + '.q_norm.weight'], eps)
            kv = v4k.rms(lin(p + '.wkv', x), w[p + '.kv_norm.weight'], eps)
        else:
            qr, kv = split
        q = v4k.rope_(lin(p + '.wq_b', qr).unflatten(-1, (c['n_heads'] // self.world, d)), f)
        kv = fp8_roundtrip(v4k.rope_(kv, f))
        scratch.rows[self.layer] = kv
        ck, ik, iq, iw = None, None, None, None
        candidate_rows = None
        if view.mode == 'full':
            ck, ik = self.stage_global(x, past=past, slot=slot, start=start,
                                       scratch=scratch)
        if view.mode in ('full', 'reindex'):
            ip = p + '.indexer'
            iq = v4k.rope_(lin(ip + '.wq_b', qr).unflatten(
                -1, (c['index_n_heads'] // self.world, c['index_head_dim'])), f)
            iq = fp4_roundtrip(iq, block=32, e4m3_scale=False)
            iw = lin(ip + '.weights_proj', x).float()
            source_layer = c.get('candidate_source_layer', -1)
            if source_layer >= 0 and self.layer > source_layer:
                candidate_rows = scratch.candidate_prep
            if self.layer == source_layer:
                # Built once per window over history plus the rows this window
                # staged, so a draft sees the candidate blocks a prefill of the
                # same tokens would see.
                source = past.sources[view.kv_source_layer]
                from ops.decode.live_index import paged_scores
                from ops.decode.cand_blocks import candidate_prep
                block = c['candidate_block_size']
                capacity = source.index_pool.pt.row_cap // view.ratio
                held = scratch.staged.get(view.kv_source_layer, (None, None))
                staged_k = ik if ik is not None else held[1]
                n = capacity + (0 if staged_k is None else len(staged_k))
                if (n + block - 1) // block <= c['candidate_topk_blocks']:
                    scratch.candidate_rows = None
                    scratch.candidate_prep = None
                else:
                    score = paged_scores(iq, iw, source.index_pool,
                        slot_rows(_requests(slot), x.device),
                        scratch.pos_t, view.ratio, c['index_n_heads'], self.reduce_sum)
                    scratch.candidate_rows = None
                    scratch.candidate_prep = candidate_prep(score,
                        scratch.pos_t, view.ratio, block,
                        c['candidate_topk_blocks'])
        out = decode_attend(past=past, slot=slot, start=start, layer=self.layer,
                            q=q, kv=kv, sink=w[p + '.attn_sink'], scale=d ** -.5,
                            scratch=scratch, staged_ckv=ck, staged_index_k=ik,
                            index_q=iq, index_weight=iw, topk=c['index_topk'],
                            total_index_heads=c['index_n_heads'],
                            reduce_scores=self.reduce_sum,
                            candidates=candidate_rows)
        out = v4k.rope_(out, f, inverse=True)
        groups, rank = c['o_groups'] // self.world, c['o_lora_rank']
        wa = w[p + '.wo_a.weight']
        if wa.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            raise TypeError('wo_a must be prepacked dense or bound to grouped GEMM')
        out = v4k.grouped_linear_bf16(out.reshape(t, groups, -1),
                                      wa.reshape(groups, rank, -1)).flatten(-2)
        out = lin(p + '.wo_b', out)
        if self.reduce_sum is not None:
            self.reduce_sum(out)
        return out

    def _ring_plan(self, slots, starts, accepted, ring, width, device):
        """Index vectors publishing the accepted prefix of every request.

        Position ``start + k`` of request ``i`` lands on ring row
        ``(start + k) % ring`` of its slot: arithmetic the host already
        knows.  The plan is filled into space taken once at the widest
        batch and shipped in a single copy, so a served round adds no
        tensors and no per-request launch of its own.
        """
        span = ring * len(slots)
        pair = getattr(self, '_ring_pair', None)
        if pair is None or pair[0].numel() < 2 * span:
            host = torch.empty(2 * span, dtype=torch.int64, pin_memory=True)
            pair = (torch.empty(2 * span, dtype=torch.int64, device=device),
                    host, host.numpy())
            self._ring_pair = pair
        dev, host, plan = pair
        n = 0
        for i, (slot, start, a) in enumerate(zip(slots, starts, accepted)):
            # Keep only the newest ringful: duplicate index_copy_
            # destinations are not ordered on CUDA when accepted exceeds
            # the ring size.  Plain scalars fill the plan: a handful of
            # numpy stores costs less than the tensors a vectorised fill
            # would have to build on the way.
            for j in range(max(0, a - ring), a):
                plan[n] = slot * ring + (start + j) % ring
                plan[span + n] = i * width + j
                n += 1
        if not n:
            return dev[:0], dev[:0]
        dev[:2 * span].copy_(host[:2 * span], non_blocking=True)
        head = int(plan[span])
        if int(plan[span + n - 1]) - head == n - 1:
            # One unbroken run — a single request aboard, or a batch that
            # happens to line up.  The source reads as a slice; no gather.
            return dev[:n], head
        return dev[:n], dev[span:span + n]

    def commit(self, *, past, slots, starts, accepted, scratch, width):
        """Publish exactly the accepted verify prefix of every request.

        ``scratch`` carries one ``width``-row window per request packed in batch
        order, so each request reads its own slice.  The publish plan travels as
        kernel arguments: one launch covers the batch and nothing is staged to
        the device.
        """
        view = past.views[self.layer]
        kv = scratch.rows[self.layer]
        if any(not 0 <= a <= width for a in accepted):
            raise ValueError('accepted must index into the verified window')
        if len(kv) != width * len(slots):
            raise ValueError('scratch rows do not span the batch window')
        if view.mode != 'full':
            return
        source, ratio = past.sources[self.layer], view.ratio or 1
        values, scores, staged_start, _ = scratch.pending[self.layer]
        staged = (staged_start if isinstance(staged_start, tuple)
                  else (staged_start,))
        if staged != tuple(starts):
            raise ValueError('staged compressor rows belong to another window')
        if ratio > 1:
            dst, src = self._ring_plan(slots, starts, accepted, 2 * ratio,
                                       width, values.device)
            source.write_res_batch(dst, src, values, scores)
        ck, ik = scratch.staged.get(self.layer, (None, None))
        counts = tuple((start + a) // ratio - start // ratio
                       for start, a in zip(starts, accepted))
        if not any(counts):
            return
        # compress_rows packs a fixed ``rows_per_request`` segment per request
        # and the publish kernel addresses row ``b * per + i``.
        per = 0 if ck is None else len(ck) // len(slots)
        if ck is None or ik is None or any(c > per for c in counts):
            raise ValueError('commit lost rows the window completed')
        source.commit_rows(slots, tuple(s // ratio for s in starts), counts,
                           ck, ik)
