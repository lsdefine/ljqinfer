"""Greedy speculative generation over one prefilled slot.

The drafter proposes block_size continuations after the accepted token, the
backbone scores all Q = block_size + 1 rows in one pass, and the longest
proposed prefix the backbone's own argmax reproduces is kept. Row 0 carries the
already accepted token, so every step lands at least one token and the emitted
stream is exactly greedy decoding: each token is the backbone's argmax at its
own position, whatever the drafter guessed.

Nothing here owns weights, Past or the token history; the caller does. The
decoder keeps the committed token history only because Engram rows for a draft
window are keyed by the n-gram sitting before it.
"""
from dataclasses import dataclass
from collections import Counter
import time
import torch
from ops.decode.argmax import argmax_rows, sample_rows, temp_rows
from model.decode_attention import DecodeScratch, slot_rows
from model.past import WINDOW_TOKENS

# Repetition damping over a sliding window of committed tokens.  A thinking
# loop is a high-confidence attractor: its top-1 sits near certainty, so
# temperature alone never leaves it, and damping what was just said is what
# breaks the loop.
_REP_PEN = 0.5
_REP_WIN = 128
# How often a token may recur in that window before it is damped at all.
# Ordinary prose repeats articles and punctuation constantly, so charging
# from the first sight would tax every step, and the drafter -- which
# proposes undamped -- would lose its guesses to the mismatch.
_REP_MIN = 4


def _starts(starts, device):
    """Row cursors as the [B] device vector every drafter entry point takes."""
    return torch.tensor([int(x) for x in starts], dtype=torch.long, device=device)


@dataclass
class SpecState:
    """The accepted token at past.pos[slot], its hidden, and its pending draft."""
    token: torch.Tensor
    hidden: torch.Tensor
    draft: torch.Tensor
    score: torch.Tensor
    # The committed-token tail this request keys Engram by.  A batch has one
    # history per row, so it travels with the row state, not on the decoder.
    hist: tuple = ()
    # The temperature this request samples at.  It rides the row state for
    # the same reason ``hist`` does: a batch mixes requests, so a decoder-wide
    # scalar would hand every row the temperature of whoever opened last.
    temp: float = 0.0


class SpecDecoder:
    """Drives draft/verify/commit rounds; the caller owns Past and the slot."""

    def __init__(self, model, drafter, *, window=None):
        self.model, self.drafter = model, drafter
        self.block = int(drafter.block_size)
        self.window = self.block + 1 if window is None else int(window)
        # One graph per batch width, keyed by it.  A capture bakes in the
        # address of the row map, never the slots in it, and a runner refills
        # that map before each replay, so B rows replay on any B slots.
        self._g: dict[int, dict] = {}
        if not 2 <= self.window <= self.block + 1:
            raise ValueError('a window holds the accepted token plus its draft')
        self.ngram = int(model.c['engram_max_ngram_size']) - 1
        self.pen = _REP_PEN
        self.pwin = _REP_WIN
        self.pmin = _REP_MIN
        # Engram keys off its own short tail, the penalty needs a longer one,
        # so the row state carries whichever of the two reaches further back.
        self.keep = max(self.ngram, self.pwin if self.pen > 0 else 0)

    def open(self, main_hidden, token, *, past, slot, history, temperature=0.0):
        """Seed the drafter windows from the prefill tail and draft one block.

        ``history`` is this row's own committed prefix.  Every row keys Engram
        by its own tail, so the prefix arrives per row and lives on the state
        the row carries -- the decoder holds no history of its own.
        """
        start = past.pos[slot]
        hist = tuple(history)
        if len(hist) != start:
            raise ValueError('token history must cover exactly the slot prefix')
        # Ring row p carries the main hidden that produced token p, so a tail of
        # T hidden rows seeds rows start-T+1 .. start.  Seeding only the last row
        # leaves the drafter's whole sliding window on stale rows and the first
        # blocks draft against a hole.
        tail = main_hidden[-WINDOW_TOKENS:]
        # The drafter's projection scratch is a fixed row budget, so a long tail
        # is seeded in workspace-sized chunks instead of one oversized call.
        space = getattr(self.drafter.linear, 'projection_workspace', None)
        cap = getattr(space, 'capacity', 0) or len(tail)
        base = start - len(tail) + 1
        # One request is the B = 1 batch: the drafter takes a slot vector and a
        # start vector here exactly as it does under the captured batch.
        row = slot_rows((slot,), main_hidden.device)
        for i in range(0, len(tail), cap):
            self.drafter.seed(tail[i:i + cap], past=past, slot=row,
                              start=_starts((base + i,), main_hidden.device))
        state = self._draft(main_hidden[-1:], token, past=past, slot=row, start=start,
                            temperature=temperature)
        state.temp = float(temperature)
        state.hist = hist[len(hist) - min(len(hist), self.keep):]
        return state

    # ---- batched verify ------------------------------------------------
    # One graph per batch width: the window is fixed, so a batch of B requests
    # is one [B*W] row block through every operator and the only per-request
    # state is the cursor the attention reads off ``past``.

    def _qrows(self, states):
        """The verify window of every request, packed [B, W]."""
        w = self.window
        # ``draft[0]`` restates the token the round already holds, exactly as
        # the single path drops it: the window is the accepted token followed
        # by the *proposals*, so the draft is read from index 1.  Taking it
        # from 0 shifts every proposal one slot and no draft can ever match.
        out = torch.empty((len(states), w), dtype=states[0].token.dtype,
                          device=states[0].token.device)
        return self._fill_qrows(states, out)

    def _fill_qrows(self, states, out):
        """Write that same window into rows already taken.

        A served round replays on one window buffer, so packing is a
        refill of it rather than a fresh pack handed over and dropped.
        """
        w = self.window
        for r, st in enumerate(states):
            out[r, 0:1].copy_(st.token.reshape(1))
            out[r, 1:w].copy_(st.draft[1:w].reshape(-1))
        return out

    def _hists(self, states, starts):
        """Per-row n-gram keys: row r is keyed by *its own* committed tail.

        Slicing one decoder-wide history by each row's start would hand every
        row the same (wrong) n-gram, so the tail rides on the row state.
        """
        out = []
        for st, s in zip(states, starts):
            n = min(int(s), self.ngram)
            h = st.hist
            out.append(tuple(h[len(h) - n:]) if n else ())
        return tuple(out)

    def capture_b(self, states, *, past, slots):
        """Capture the verify body for a batch of this width.

        The slots only say which pool rows this capture happens to start on;
        the row map is refilled per replay, so the graph belongs to the width.
        """
        slots = tuple(int(x) for x in slots)
        b, w = len(slots), self.window
        starts = [int(past.pos[x]) for x in slots]
        qin = self._qrows(states)
        hists = self._hists(states, starts)
        scratch = DecodeScratch()
        qinh = torch.empty_like(qin, device='cpu').pin_memory()
        qinh.copy_(qin)
        rows = tuple(qin[r] for r in range(b))
        rowsh = tuple(qinh[r] for r in range(b))
        # One temperature per verify row, refilled per replay.  A python
        # float would be burned into the capture as a constant and every
        # later round would sample at whatever the capture happened to see.
        tempv = temp_rows(b * w, 0.0, qin.device)
        temph = torch.empty(b * w, dtype=torch.float32).pin_memory()
        # Which ids to damp on each verify row, and by how much.  A weight of
        # zero is a no-op, so unused columns carry weight zero and any id.
        pid = torch.zeros(b * w, self.pwin, dtype=torch.int64,
                          device=qin.device)
        pwt = torch.zeros(b * w, self.pwin, dtype=torch.float32,
                          device=qin.device)
        pidh = torch.empty_like(pid, device='cpu').pin_memory()
        pidh.zero_()
        pwth = torch.empty_like(pwt, device='cpu').pin_memory()
        pwth.zero_()

        def body():
            out = self.model.forward(rows, past=past, slot=slots,
                                     scratch=scratch, history_tokens=hists)
            if self.pen > 0:
                # These rows may hold only this rank's slice of the
                # vocabulary, so ids that live on another rank are dropped
                # here rather than indexed out of bounds.
                keep = pid < out.logits.shape[1]
                # Repeats land on the same columns, so a duplicated id damps
                # twice -- the window counts frequency for free.
                out.logits.scatter_add_(
                    1, torch.where(keep, pid, torch.zeros_like(pid)),
                    (pwt * keep).to(out.logits.dtype))
            greedy = sample_rows(out.logits, tempv).view(b, w)
            # Accepted length per request: the draft token at q+1 must equal
            # what the model itself would emit at q.
            keep = (qin[:, 1:] == greedy[:, :-1]).cumprod(1).sum(1)
            return out, greedy, keep + 1

        self.model.stage_engram(rowsh, past=past, slot=slots,
                                history_tokens=hists)
        for _ in range(2):
            body()
        torch.cuda.synchronize()
        scratch.pos_t = None
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), \
                torch.cuda.graph(graph, capture_error_mode='thread_local'):
            out, greedy, keep = body()
        torch.cuda.synchronize()
        g = dict(
            slots=slots, graph=graph, scratch=scratch, qin=qin, qinh=qinh,
            rows=rows, rowsh=rowsh, greedy=greedy, keep=keep,
            tempv=tempv, temph=temph,
            pid=pid, pwt=pwt, pidh=pidh, pwth=pwth,
            hidden=out.main_hidden,
            hkeep=torch.empty(b, dtype=keep.dtype, device='cpu').pin_memory(),
            hgre=torch.empty(b, w, dtype=greedy.dtype,
                             device='cpu').pin_memory(),
        )
        # The drafter takes the whole batch in one pass: the draft rows of
        # every request share the row axis of each GEMM and only attention
        # splits them by slot, so a round costs two replays, not two per slot.
        dev = past.pos_dev.device
        # The row map the whole graph reads slots through: one buffer per
        # width, refilled per replay, so the capture holds no slot identity.
        slots_t = slot_rows(slots, dev)
        dh = torch.cat([states[r].hidden.reshape(1, -1) for r in range(b)]).clone()
        dtok = torch.stack([states[r].token.reshape(()) for r in range(b)]).clone()
        sstart = past.pos_dev.index_select(0, slots_t).clone()
        accd = torch.empty(b, dtype=torch.long, device=dev)
        rowsd = torch.arange(b, device=dev) * w
        # The draft samples one token per request per block position, so its
        # row axis is the batch, not the verify window.
        tempd = temp_rows(b, 0.0, dev)
        tempdh = torch.empty(b, dtype=torch.float32).pin_memory()
        # Row offsets of the seed block, one per verify row below the last.
        soff = torch.arange(w - 1, device=dev)
        # The accepted rows and their host mirrors: a round refills this
        # index space rather than minting one.
        rowsel = torch.empty(b, dtype=torch.long, device=dev)
        acch = torch.empty(b, dtype=torch.long, pin_memory=True)
        # Mirrors the cursor's own dtype: a mismatch here would make the
        # copy allocate a cast buffer every round.
        ssth = torch.empty(b, dtype=past.pos_dev.dtype, pin_memory=True)

        def dbody(dh=dh, dtok=dtok):
            # The cursor read must happen inside the body: an index_select done
            # at capture time would freeze the positions this capture saw.
            return self.drafter(dh, dtok, past=past, slot=slots_t,
                                start=past.pos_dev.index_select(0, slots_t),
                                temperature=tempd)

        def sbody(sstart=sstart, hid=out.main_hidden):
            # The seed rows are each request's accepted block of the verify
            # output, re-read on every replay from the buffer it just wrote.
            # ``hid`` is bound to *this* capture's buffer: one graph per width
            # tuple is kept alive, and a stale lookup would seed from another
            # batch's verify output.
            rows = hid.view(b, w, -1)[:, :w - 1]
            # Only the rows this round accepted may enter the drafter's ring.
            # A rejected row would turn into history as soon as the cursor
            # moves past it, and the next draft would attend a token nobody
            # emitted -- the single path seeds main_hidden[:accepted-1] for
            # exactly this reason.  The unaccepted rows keep their true rotary
            # position and land in the window's void row.
            pos = (sstart.reshape(-1, 1) + soff).masked_fill(
                soff.reshape(1, -1) >= accd.reshape(-1, 1), -1)
            return self.drafter.seed(rows.reshape(b * (w - 1), -1), past=past,
                                     slot=slots_t, start=sstart, pos=pos)

        for _ in range(2):
            sbody()
            dbody()
        torch.cuda.synchronize()
        # Seed and draft share the drafter's routing and projection scratch,
        # so they belong in one graph: two graphs each capture those shared
        # buffers and the second capture rewrites the row indices the first one
        # replays with, which walks off the end as soon as the batch grows.
        dgraph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), \
                torch.cuda.graph(dgraph, capture_error_mode='thread_local'):
            sbody()
            ids, _, score = dbody()
        torch.cuda.synchronize()
        g['d'] = dict(dh=dh, dtok=dtok, ids=ids, score=score, sstart=sstart,
                            tempd=tempd, tempdh=tempdh,
                            graph=dgraph, accd=accd, rowsd=rowsd,
                            rowsel=rowsel, acch=acch, ssth=ssth,
                            # Anchors: the closures own the scratch the graph
                            # captured, so dropping them frees that memory and
                            # the replay then reads recycled rows.
                            sbody=sbody, dbody=dbody)
        # One graph per batch width, kept for the process lifetime: a request
        # leaving the batch must not force a re-capture, which would both
        # stall the round and run its warmup passes over live KV pages.
        self._g[b] = g
        return g

    def _slot_state(self, past, d):
        """Per-slot home for one round's draft result.

        The draft graph's outputs live in the graph's own space, which the
        next replay writes over, so a state handed back as a view of them
        decays as soon as any request steps. Each slot owns a row here,
        shaped once from the graph's buffers, so a state stays good until
        that slot steps again and no round allocates.
        """
        sb = self.__dict__.get('_sb')
        if sb is None:
            n = past.pos_dev.numel()
            sb = {k: torch.empty((n,) + tuple(d[k].shape[1:]),
                                 dtype=d[k].dtype, device=d[k].device)
                  for k in ('dtok', 'dh', 'ids', 'score')}
            sb['idx'] = torch.empty(n, dtype=torch.long,
                                    device=d['dtok'].device)
            # Pinned mirror of the same map: the slots of a round are written
            # here and copied down, so naming the batch costs no allocation.
            sb['idxh'] = torch.empty(n, dtype=torch.long).pin_memory()
            self._sb = sb
        return sb

    def _fill_temps(self, states, g):
        """Point both graphs at this round's temperatures.

        Verify samples one token per window row and the drafter one per
        request, so the same per-request temperature is written at two
        different row pitches.
        """
        w = self.window
        th, td = g['temph'], g['d']['tempdh']
        for r, st in enumerate(states):
            t = float(st.temp)
            td[r] = t
            th[r * w:(r + 1) * w] = t
        g['tempv'].copy_(th, non_blocking=True)
        g['d']['tempd'].copy_(td, non_blocking=True)

    def _fill_pen(self, states, g):
        """Point the graph at the tokens a row has overused.

        Only the count above the free allowance is charged, so a run that
        is not repeating itself hands the graph an all-zero weight and the
        drafter's guesses stand untouched.

        Verify samples one token per window row, so a request's charge is
        written across its own window span, at the pitch the temperatures
        use.  A batch mixes requests and each row carries its own history,
        so no row can ever be charged for what another one said.
        """
        if self.pen <= 0:
            return
        w, k, t = self.window, self.pwin, self.pmin
        pidh, pwth = g['pidh'], g['pwth']
        pwth.zero_()
        for r, st in enumerate(states):
            tail = st.hist[-k:]
            if len(tail) <= t:
                continue
            hot = [(i, c - t) for i, c in Counter(tail).items() if c > t]
            if not hot:
                continue
            hot = hot[:k]
            n = len(hot)
            ids = torch.tensor([h[0] for h in hot], dtype=torch.int64)
            wts = torch.tensor([-self.pen * h[1] for h in hot],
                               dtype=torch.float32)
            pidh[r * w:(r + 1) * w, :n] = ids
            pwth[r * w:(r + 1) * w, :n] = wts
        g['pid'].copy_(pidh, non_blocking=True)
        g['pwt'].copy_(pwth, non_blocking=True)

    def step_gb(self, states, *, past, slots):
        """One batched verify round: replay, commit, redraft every request.

        The graph is captured for the batch's width; the caller keeps the
        batch stable for the lease so the staged rows keep meaning one row.
        Setting ``_gbprof`` splits the round into phases with a sync between
        them: the phase totals land in ``_gbt`` for the caller to report.
        """
        from time import perf_counter
        slots = tuple(int(x) for x in slots)
        g = self._g.get(len(slots))
        if g is None:
            # Capturing here would stall every request aboard, so a missing
            # width is a setup error, not something to paper over at runtime.
            raise RuntimeError(
                'no verify graph for width %d: widths %s were captured at '
                'startup' % (len(slots), sorted(self._g)))
        prof = self.__dict__.get('_gbprof')
        acct = self.__dict__.setdefault('_gbt', {})

        def mark(name, t0):
            if not prof:
                return t0
            torch.cuda.current_stream().synchronize()
            t1 = perf_counter()
            acct[name] = acct.get(name, 0.0) + (t1 - t0) * 1e3
            acct['n'] = acct.get('n', 0) + (1 if name == 'verify' else 0)
            return t1

        b, w = len(slots), self.window
        t0 = perf_counter()
        starts = [int(past.pos[x]) for x in slots]
        for i, s_ in enumerate(slots):
            past.ensure(s_, starts[i] + w)
        # Refill the row map the whole graph reads slots through: the capture
        # bakes in its address only, so this is what points it at this batch.
        slot_rows(slots, g['qin'].device)
        self._fill_qrows(states, g['qin'])
        g['qinh'].copy_(g['qin'])
        self._fill_temps(states, g)
        self._fill_pen(states, g)
        t0 = mark('host_in', t0)
        # Staging feeds rows the leading layers read, so it has to be done
        # and visible before the replay: overlapping it under a mid-graph gate
        # lets those layers read a row still being written, which shows up as
        # another request's token.
        try:
            self.model.stage_engram(g['rowsh'], past=past, slot=slots,
                                    history_tokens=self._hists(states, starts),
                                    gated=True)
        except BaseException:
            self.model.release_engram_gates()
            raise
        torch.cuda.current_stream().synchronize()
        g['graph'].replay()
        g['hkeep'].copy_(g['keep'], non_blocking=True)
        g['hgre'].copy_(g['greedy'], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        t0 = mark('verify', t0)
        acc = [int(x) for x in g['hkeep'].tolist()]
        g['scratch'].rebase(tuple(starts))
        self.model.commit(tuple(acc), past=past, slot=slots,
                          scratch=g['scratch'])
        t0 = mark('commit', t0)
        d = g['d']
        for r in range(b):
            past.pos_dev[slots[r]] = starts[r] + acc[r]
        # The accepted token and its hidden row feed the draft graph through the
        # two buffers it was captured on, one row per request.
        ah = d['acch'][:b]
        for r in range(b):
            ah[r] = acc[r] - 1
        d['accd'].copy_(ah, non_blocking=True)
        torch.gather(g['greedy'], 1, d['accd'][:, None], out=d['dtok'][:, None])
        torch.add(d['rowsd'], d['accd'], out=d['rowsel'])
        torch.index_select(g['hidden'], 0, d['rowsel'], out=d['dh'])
        # Same gap as the single path: the intermediate accepted rows must
        # reach the drafter ring before the draft graph reads its window.
        sh = d['ssth'][:b]
        for r in range(b):
            sh[r] = starts[r] + 1
        d['sstart'].copy_(sh, non_blocking=True)
        d['graph'].replay()
        # Two different rows of the same round, exactly as the single path:
        # the n-gram tail advances over the accepted *window* (row 0 of which
        # is the token the last round already emitted), while what the caller
        # receives is the accepted *greedy* output -- window shifted by one,
        # ending on the token that seeds the next round.  Handing back the
        # window instead would repeat the previous token and drop the newest.
        # Both are already on the host from the copies above, so neither costs
        # a device sync.
        ng, qh, gre = self.keep, g['qinh'], g['hgre']
        out, toks = [], []
        # Take the round out of the graph's space and into each slot's row
        # before another replay can write over it.
        sb = self._slot_state(past, d)
        ix, ixh = sb['idx'][:b], sb['idxh'][:b]
        for r in range(b):
            ixh[r] = slots[r]
        ix.copy_(ixh)
        for k in ('dtok', 'dh', 'ids', 'score'):
            sb[k].index_copy_(0, ix, d[k])
        for r in range(b):
            h = states[r].hist + tuple(int(t) for t in qh[r][:acc[r]].tolist())
            toks.append(tuple(int(t) for t in gre[r][:acc[r]].tolist()))
            s_ = slots[r]
            out.append(SpecState(token=sb['dtok'][s_],
                                 hidden=sb['dh'][s_:s_ + 1],
                                 draft=sb['ids'][s_], score=sb['score'][s_],
                                 hist=h[len(h) - min(len(h), ng):]))
        t0 = mark('draft', t0)
        # Each row's committed ids, in order: the caller needs the tokens, and
        # the count it used to get is just their length.
        return out, toks

    def _draft(self, hidden, token, *, past, slot, start, temperature=0.0):
        """The pending draft of one request, taken out of the B = 1 batch."""
        ids, _, score = self.drafter(hidden, token.reshape(1), past=past, slot=slot,
                                     start=_starts((start,), hidden.device),
                                     temperature=temperature)
        return SpecState(token=token, hidden=hidden, draft=ids[0], score=score[0],
                         temp=float(temperature))
