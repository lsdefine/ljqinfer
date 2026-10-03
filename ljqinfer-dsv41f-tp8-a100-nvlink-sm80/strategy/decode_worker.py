"""TP8 generation worker behind server/engine_server.py.

torchrun --standalone --nproc-per-node=8 -m strategy.decode_worker

Rank 0 serves the localhost RPC (port 62001) on a uvicorn thread and drives
the ranks from its main thread; every rank runs the same prefill -> spec
decode rounds, so the emitted tokens are replicated and only the request
header travels over a dedicated CPU Gloo group.  Rank 0 is the only decision
maker: each round it publishes one opcode (OPEN / STEP / CLOSE / STOP) naming
the rows that take part, and every other rank is an interpreter that obeys it.
Because a STEP names its rows, the set of rows moving together can change from
round to round -- that is what lets requests board and leave a running batch.
Greedy only (spec decode verifies argmax).
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist
from tokenizers import Tokenizer

from ops.decode.argmax import sample_rows

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.checkpoint_weights import CheckpointWeights  # noqa: E402
from model.cold import fields_for  # noqa: E402
from model.decode_build import build_decode  # noqa: E402
from model.dspark_build import build_drafter  # noqa: E402
from model.engram import EngramHash  # noqa: E402
from model.model_api import ModelExecution  # noqa: E402
from model.past import SlotPool  # noqa: E402
from model.prefill_build import build_prefill, PrefillParallel  # noqa: E402
from model.prefill_config import released_config  # noqa: E402
from model.spec_decode import SpecDecoder  # noqa: E402
from server.encoding_dsv41 import eos_token  # noqa: E402
from strategy.cold_kv import ColdCache  # noqa: E402
from strategy.strategy import Strategy as PrefixFlow  # noqa: E402
from strategy.host_arena import HostArena  # noqa: E402

# Control plane.  One header per collective action: [op, arg, n_rows, *row_ids].
# `arg` carries the prompt length for OPEN and is unused elsewhere; the row ids
# name which rows the action applies to, so a STEP is free to move a different
# subset every round.
OP_OPEN, OP_STEP, OP_CLOSE, OP_STOP, OP_IMAGE = 1, 2, 3, 4, 5
HEAD = 4           # op, arg, row count, temperature in milli-units


# The one deployment this engine is generated for.  A specialised engine has
# no knobs: every number below is a fact about the model and the machine it
# was built for, and a constant the code can read beats a flag whose default
# disagrees with how the thing actually gets launched.
WEIGHTS = '/mnt/data/kw/models/DeepSeek-V4.1-Flash'
TOKENIZER = WEIGHTS + '/tokenizer.json'
MAX_BATCH = 4        # rows aboard at once: KV slots and the widest verify graph
WINDOW = 6           # speculative window
CHUNK = 12288        # prefill tokens per chunk: the engine's own ceiling,
                     # so a prompt that fits is one chunk and pays the
                     # per-chunk setup once
MAX_SEQ = 1 << 20    # positions the model can address
POOL_TOKENS = 4 << 20  # the paged KV budget, and the real ceiling: a row
                     # can grow to POOL_TOKENS // MAX_BATCH tokens
BOARD_GRACE_S = 0.10 # an empty bus waits this long at the stop, so requests
                     # that arrive together board together instead of each one
                     # stopping the bus for its own prefill
BOARD_EVERY = 128    # a rolling bus opens its door once every this many rounds:
                     # everybody else queues, and nobody aboard gets preempted
COLD_BYTES = 80 << 30 # host prefix cache, one arena shared by every rank
COLD_PIN = True      # the arena page-locks its pages once and reuses them, so the
                     # copy keeps the DMA fast path.  Pinning per entry instead
                     # puts a cudaHostAlloc on every write and measured slower
                     # than pageable staging: 0.0202 against 0.0128 s per chunk
COLD_NS = 'cold'     # one cache, one namespace: no rank appears in the key

class ImageRejected(ValueError):
    """All ranks rejected an image before entering GPU collectives."""


class CancelHandle:
    def __init__(self):
        self.flag = threading.Event()

    def cancel(self):
        was = self.flag.is_set()
        self.flag.set()
        return not was

    def stop_at_semantic_eos(self):
        # Generation already terminates on the model EOS; nothing to change.
        return False


class ResultQueue(queue.Queue):
    """Event stream one request; attributes engine_server reads."""

    def __init__(self, request_id, depth):
        super().__init__()
        self.cancel_handle = CancelHandle()
        self.metrics = {}
        self.request_id = request_id
        self.queue_depth_on_submit = depth


@dataclass
class Job:
    tokens: list
    max_new: int
    out: ResultQueue
    temperature: float = 1.0
    submitted_at: float = field(default_factory=time.perf_counter)
    image_payload: object = None


@dataclass
class Row:
    """Everything one in-flight request owns.

    The engine keeps no per-request state of its own: a row carries its slot,
    its cold-KV lease, its speculative state and its prefill phase clock, so
    any subset of rows can be advanced together by `step_rows`.
    """
    tokens: tuple
    # A slot is borrowed from the pool by open_row and returned by close_row.
    slot: int = -1
    state: object = None
    session: object = None
    hit_tokens: int = 0
    first: int = 0
    temp: float = 0.0
    phases: dict = field(default_factory=dict)


class RequestTooLong(ValueError):
    """One request asked for more context than the engine has; only it fails."""


@dataclass
class Lane:
    """A boarded request: the row it rides on, plus what rank 0 owes its caller.

    A ride is always the same three moves -- `admit`, `take` each round,
    `dismiss` -- whether the bus carries one passenger or four, so the single
    request path and the scheduler keep one set of books instead of two.
    """
    rid: int
    row: object
    eos_id: int
    limit: int
    emit: object = None
    cancel: object = None
    queue_seconds: float = 0.0
    prompt: int = 0
    # The prefill argmax already counts as produced (same convention as
    # bench/spec_generate.py): spec rounds only emit what follows it.
    produced: int = 1
    rounds: int = 0
    reason: str = 'length'
    t0: float = 0.0
    t1: float = 0.0
    control_seconds: float = 0.0
    step_seconds: float = 0.0
    # Time this lane sat still while somebody else's prompt was being prefilled.
    # Without it `wall_ms_per_step` looks like the engine got slower, when all
    # that happened is the bus stopped to pick somebody up.
    stall_seconds: float = 0.0
    done: bool = False

    def take(self, ids):
        """Book one round's tokens for this lane and decide if it gets off."""
        self.rounds += 1
        ids = list(ids)[:self.limit - self.produced]
        if self.eos_id in ids:
            ids = ids[:ids.index(self.eos_id) + 1]
            self.reason = 'eos'
        self.produced += len(ids)
        if self.emit and ids:
            self.emit({'type': 'token', 'token_ids': ids})
        if self.reason == 'eos' or self.produced >= self.limit:
            self.done = True
        return ids

    def check_cancel(self):
        """Cancelling is a rank-0 decision: stop asking for STEPs, then CLOSE."""
        if not self.done and self.cancel is not None and self.cancel.is_set():
            self.reason = 'cancelled'
            self.done = True
        return self.done


@dataclass
class Engine:
    rank: int
    device: torch.device
    past: object
    engine: object
    spec: object
    eos_id: int
    row_cap: int      # longest a row can grow; the KV pool decides it
    chunk: int
    hdr: torch.Tensor = field(default=None)
    # Rows currently boarded, keyed by the id rank 0 gave them.  Every rank
    # builds the same table because every rank obeys the same header stream.
    rows: dict = field(default_factory=dict)
    next_id: int = 0
    ctrl: object = None
    # Cross-request prefix reuse.  `flow` owns the host-side cold KV cache, so
    # a retired row can hand its slot straight back to the pool: what the next
    # request reuses is the cold copy, not the slot it happened to live on.
    flow: object = None
    vision: object = None

    def encode_images(self, payload):
        from server.image_input import patchify
        phases = {'image_preprocess_seconds': 0.0, 'image_broadcast_seconds': 0.0,
                  'image_encode_seconds': 0.0}
        meta = [None]
        patches = []
        if self.rank == 0:
            t = time.perf_counter()
            try:
                patches = [patchify(im) for im in payload['images']]
                meta[0] = {'images': [{k: v for k, v in im.items() if k not in ('url', 'data')}
                                     for im in payload['images']]}
            except Exception as exc:
                meta[0] = {'error': str(exc)}
            phases['image_preprocess_seconds'] = time.perf_counter()-t
        dist.broadcast_object_list(meta, src=0, group=self.ctrl)
        if 'error' in meta[0]:
            raise ImageRejected(meta[0]['error'])
        spans = []
        for i, im in enumerate(meta[0]['images']):
            t = time.perf_counter()
            shape = (im['grid'][0]*im['grid'][1], 588)
            x, error = None, None
            try:
                x = (torch.from_numpy(patches[i]).to(self.device, dtype=torch.bfloat16)
                     if self.rank == 0 else torch.empty(shape, device=self.device, dtype=torch.bfloat16))
            except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                error = f'rank {self.rank}: {exc}'
            errors = [None] * dist.get_world_size(self.ctrl)
            dist.all_gather_object(errors, error, group=self.ctrl)
            if any(errors):
                del x
                raise ImageRejected('; '.join(e for e in errors if e))
            dist.broadcast(x, src=0)
            torch.cuda.synchronize(self.device)
            phases['image_broadcast_seconds'] += time.perf_counter()-t
            t = time.perf_counter()
            spans.append((im['start'], self.vision.encode(x, im)))
            torch.cuda.synchronize(self.device)
            phases['image_encode_seconds'] += time.perf_counter()-t
        if self.rank == 0:
            print('IMAGE_PHASES '+json.dumps(phases), flush=True)
        return spans, phases

    def open_row(self, tokens, temp=0.0, image_spans=(), image_phases=None):
        """Prefill one request and hand back the row that owns its decode state."""
        past = self.past
        row = Row(tokens=tuple(int(x) for x in tokens), temp=float(temp))
        # Phase clock (V4-style): each mark syncs the device so the phase
        # metrics are real wall time, not launch time.  Four syncs per request.
        ph, t = row.phases, time.perf_counter()

        def mark(name):
            nonlocal t
            torch.cuda.synchronize()
            now = time.perf_counter()
            ph[name] = ph.get(name, 0.0) + (now - t)
            t = now
        # The row borrows a slot from the pool and hands it back in close_row.
        # `alloc` only returns slots that `release` has already wiped (it
        # asserts the slot sits at position 0), so nothing -- GDN kv/score
        # state, prefill tails, replay bookkeeping -- survives from whoever
        # used it before.  Every rank runs the same open/close sequence over an
        # identically ordered free list, so all ranks pick the same slot for a
        # row without exchanging a single byte.
        if self.flow is None or image_spans:
            row.slot = past.alloc()
            start = 0
        else:
            # open() allocates the slot, restores the longest cached prefix
            # minus the sliding window and replays the hot rings; the chunk
            # loop below computes only what is left.
            session = self.flow.open(row.tokens)
            row.slot, row.session = session.slot, session
            row.hit_tokens = session.hit_tokens
            start = session.position
            mark('cache_load_seconds')
        slot = row.slot
        if image_spans:
            past.image_spans[slot] = image_spans
            ph.update(image_phases or {})
            ph["image_tokens"] = sum(len(x) for _, x in image_spans)
        while start < len(tokens):
            if row.session is not None:
                # step() publishes each chunk endpoint into the cold cache
                # and may restart from 0 when the ring is partially filled.
                row.session.step()
                start = row.session.position
                continue
            end = min(len(tokens), start + self.chunk)
            self.engine.prefill_chunk(slot, tuple(tokens[start:end]),
                                      history_tokens=tuple(tokens[max(0, start - 3):start]))
            start = end
        mark('prefill_compute_seconds')
        if row.session is not None:
            # The store cost hides inside session.step(), so prefill_compute
            # silently carries it.  Surface what the session already counted
            # to make the cold write path as observable as its read path.
            ph['cache_store_seconds'] = row.session.metrics.cache_store_seconds
            ph['cache_stored_blocks'] = row.session.metrics.cache_stored_blocks
        out = self.engine.finish_prefill(slot)
        mark('finish_prefill_seconds')
        ph['cache_hit_tokens'] = row.hit_tokens
        # The first generated token comes out of prefill and is sampled at
        # the request's temperature like every token after it.
        token = sample_rows(out.logits[-1:], float(temp))[0]
        row.first = int(token)
        row.state = self.spec.open(out.main_hidden, token, past=past, slot=slot,
                                   history=row.tokens, temperature=temp)
        return row

    def step_rows(self, rows):
        """Advance any subset of rows by one speculative round.

        The batch replays the graph captured for its width, and one row is
        that same graph at width one.  Returns, per row and in the order
        given, the token ids that row committed this round.
        """
        past = self.past
        states, toks = self.spec.step_gb([r.state for r in rows], past=past,
                                         slots=[r.slot for r in rows])
        for row, state in zip(rows, states):
            row.state = state
        return [list(ids) for ids in toks]

    def close_row(self, row):
        """Retire `row` and hand its slot back to the pool.

        Closing the cold session publishes this row's KV to the host cache and
        releases the slot; without a session the slot is released directly.
        Either way the slot leaves wiped, so a later row can take it while this
        row's prefix stays reusable from the cold copy."""
        row.state = None
        if row.session is not None:
            row.session.close()
            row.session = None
        elif row.slot >= 0:
            self.past.release(row.slot)
        row.slot = -1

    # ---- control plane -------------------------------------------------
    # Rank 0 decides, everyone obeys.  `publish` and `receive` are the two
    # ends of the same broadcast, `obey` is the action both ends then run.

    def publish(self, op, arg=0, ids=(), temp_milli=0):
        """Rank 0: announce the next action and the rows it applies to.

        The temperature travels as milli-units in the header because the
        header is the only thing a follower sees: every rank draws its own
        Gumbel noise, so a temperature known to rank 0 alone would have the
        ranks commit different tokens and the batch would diverge.
        """
        hdr = self.hdr
        hdr[0], hdr[1], hdr[2], hdr[3] = op, arg, len(ids), temp_milli
        for i, rid in enumerate(ids):
            hdr[HEAD + i] = rid
        dist.broadcast(hdr, 0, group=self.ctrl)
        return op, arg, list(ids), temp_milli

    def receive(self):
        """Followers: block until rank 0 says what happens next."""
        dist.broadcast(self.hdr, 0, group=self.ctrl)
        hdr = self.hdr
        return (int(hdr[0]), int(hdr[1]),
                [int(hdr[HEAD + i]) for i in range(int(hdr[2]))],
                int(hdr[3]))

    def obey(self, op, arg, ids, temp_milli=0, tokens=None, image_payload=None):
        """Run one published action.  Identical on every rank: the prompt for
        an OPEN is broadcast here, so followers need nothing but the header."""
        if op == OP_IMAGE:
            # This watchdog covers *all* post-OPEN allocations/collectives/prefill.
            # A poisoned CUDA context cannot be recovered by a Python catch;
            # torchrun tears down the group when any rank exits. Text is untouched.
            def expired():
                print('FATAL image OPEN exceeded 600s; terminating rank', self.rank, flush=True)
                os._exit(70)
            timer = threading.Timer(600, expired)
            timer.daemon = True
            timer.start()
            try:
                return self._open_image(arg, ids, temp_milli, tokens, image_payload)
            finally:
                timer.cancel()
        if op == OP_OPEN:
            buf = (torch.tensor(list(tokens), dtype=torch.int64)
                   if tokens is not None else torch.zeros(arg, dtype=torch.int64))
            dist.broadcast(buf, 0, group=self.ctrl)
            row = self.open_row(buf.tolist(), temp=temp_milli / 1000.0)
            self.rows[ids[0]] = row
            return row
        if op == OP_STEP:
            return self.step_rows([self.rows[i] for i in ids])
        if op == OP_CLOSE:
            for i in ids:
                self.close_row(self.rows.pop(i))
            return None
        raise AssertionError('unknown opcode %d' % op)

    def _open_image(self, arg, ids, temp_milli, tokens, payload):
        buf = (torch.tensor(list(tokens), dtype=torch.int64) if tokens is not None
               else torch.empty(arg, dtype=torch.int64))
        dist.broadcast(buf, 0, group=self.ctrl)
        spans, phases = self.encode_images(payload)
        row = self.open_row(buf.tolist(), temp=temp_milli / 1000.0,
                            image_spans=spans, image_phases=phases)
        self.rows[ids[0]] = row
        return row

    def board(self, tokens, temperature=1.0, image_payload=None):
        """Rank 0: put a new row on the bus and return (row_id, row)."""
        rid, self.next_id = self.next_id, self.next_id + 1
        tm = int(round(max(float(temperature), 0.0) * 1000))
        op, arg, ids, tm = self.publish(OP_OPEN if image_payload is None else OP_IMAGE, len(tokens), (rid,), tm)
        return rid, self.obey(op, arg, ids, tm, tokens, image_payload)

    def advance(self, ids):
        """Rank 0: move exactly these rows one speculative step."""
        return self.obey(*self.publish(OP_STEP, 0, ids))

    def retire(self, ids):
        """Rank 0: take these rows off the bus and free their slots."""
        self.obey(*self.publish(OP_CLOSE, 0, ids))

    def follow(self):
        """Non-zero ranks: obey headers until rank 0 says stop."""
        while True:
            op, arg, ids, tm = self.receive()
            if op == OP_STOP:
                return
            try:
                self.obey(op, arg, ids, tm)
            except ImageRejected:
                continue  # rank 0 reports the same rejection; no slot was allocated

    def admit(self, tokens, max_new, *, emit=None, cancel=None, queue_seconds=0.0,
              temperature=1.0, image_payload=None):
        """Open a row for one request and hand back the lane that rides it."""
        # Refuse an oversized prompt before it touches the cache: `board` writes
        # KV straight into the slot's pages, so a prompt that does not fit walks
        # off the end of them and takes every rank down with an illegal access.
        room = self.row_cap - len(tokens) - self.spec.window - 8
        if room <= 0:
            raise RequestTooLong('prompt of %d tokens leaves no room under row_cap=%d'
                                 % (len(tokens), self.row_cap))
        t0 = time.perf_counter()
        rid, row = self.board(tokens, temperature, image_payload)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lane = Lane(rid=rid, row=row, eos_id=self.eos_id, limit=min(max_new, room),
                    emit=emit, cancel=cancel, queue_seconds=queue_seconds,
                    prompt=len(tokens), t0=t0, t1=t1)
        if emit:
            emit({'type': 'prefill', 'metrics': {
                'input_tokens': len(tokens),
                'prefill_tokens': len(tokens) - row.hit_tokens,
                'model_prefill_seconds': t1 - t0, 'prefill_seconds': t1 - t0,
                'queue_wait_seconds': queue_seconds,
                'chunks': (len(tokens) - row.hit_tokens + self.chunk - 1) // self.chunk,
                **row.phases,
                # Counts are not durations; cache_store is nested in prefill.
                'open_s': (t1 - t0) - sum(v for k, v in row.phases.items()
                                         if k.endswith('_seconds')
                                         and k != 'cache_store_seconds')}})
            emit({'type': 'token', 'token_ids': [row.first]})
        if row.first == self.eos_id:
            lane.reason = 'eos'
            lane.done = True
        elif lane.produced >= lane.limit:
            lane.done = True
        return lane

    def roll(self, lanes):
        """Advance every lane aboard by one round, booking the time to each."""
        ts = time.perf_counter()
        head = self.publish(OP_STEP, 0, [lane.rid for lane in lanes])
        te = time.perf_counter()
        out = self.obey(*head)
        tf = time.perf_counter()
        for lane, ids in zip(lanes, out):
            lane.control_seconds += te - ts
            lane.step_seconds += tf - te
            lane.take(ids)

    def dismiss(self, lane):
        """Close the row, hand its slot back, and settle the lane's books."""
        t2 = time.perf_counter()
        self.retire([lane.rid])
        r = lane.rounds
        stats = dict(decode_steps=r, accepted_tokens=max(lane.produced - 1 - r, 0),
                     decode_tps=lane.produced / (t2 - lane.t1) if r else 0.0,
                     prefill_seconds=lane.t1 - lane.t0,
                     prefill_stall_seconds=lane.stall_seconds,
                     wall_ms_per_step=1000 * (t2 - lane.t1) / r if r else 0.0,
                     control_ms_per_step=1000 * lane.control_seconds / r if r else 0.0,
                     model_step_host_ms=1000 * lane.step_seconds / r if r else 0.0,
                     queue_wait_seconds=lane.queue_seconds, stop_reason=lane.reason)
        if lane.emit:
            print('[strategy] ' + json.dumps(dict(input_tokens=lane.prompt, output_tokens=lane.produced, **stats, **{k: round(v, 4) for k, v in lane.row.phases.items()}), sort_keys=True), flush=True)
            lane.emit(dict(type='end', reason=lane.reason, **stats))
        return stats

    def generate(self, tokens, max_new, cancel=None, emit=None, queue_seconds=0.0):
        """Run one request as a bus of one: same three moves as a full batch."""
        lane = self.admit(tokens, max_new, emit=emit, cancel=cancel,
                          queue_seconds=queue_seconds)
        while not (lane.done or lane.check_cancel()):
            self.roll([lane])
        self.dismiss(lane)
        return lane.produced, lane.rounds


def build(rank, device, ctrl):
    c = released_config()

    class TokenizerAdapter:
        backend_tokenizer = Tokenizer.from_file(TOKENIZER)

        def __len__(self):
            return self.backend_tokenizer.get_vocab_size()

    tok = TokenizerAdapter.backend_tokenizer
    hasher = EngramHash(c, TokenizerAdapter())
    weights = CheckpointWeights(WEIGHTS, device, rank)
    tables = weights.host_tables(c['engram_layer_ids'])
    weights.load_dense()
    weights.load_experts()
    weights.load_dspark()
    torch.cuda.synchronize()
    past = SlotPool(MAX_BATCH, MAX_SEQ, POOL_TOKENS,
                    device=device).configure_default(
        window_dtype=torch.bfloat16, ckv_dtype=torch.bfloat16,
        index_dtype=torch.bfloat16)
    # No row can pass the pool's per-row share, so that share -- not the
    # addressable span -- sizes the rope tables and the index scan.
    span = past.row_cap
    hasher.image_spans = past.image_spans
    trunk = build_prefill(c, weights, hasher, tables, device=device,
                          length=CHUNK, parallel=PrefillParallel(),
                          max_position=span)
    engine = ModelExecution(trunk, past, prefill_chunk_tokens=CHUNK)
    spec_model = build_decode(c, weights, hasher, tables, device=device,
                              window=WINDOW, parallel=PrefillParallel(),
                              max_position=span, batch=MAX_BATCH)
    drafter = build_drafter(c, weights, device=device,
                            parallel=PrefillParallel(), max_position=span,
                            batch=MAX_BATCH)
    spec = SpecDecoder(spec_model, drafter, window=WINDOW)
    eos_id = tok.token_to_id(eos_token)
    # Cross-request prefix cache: plain host memory, owned by rank 0 alone.
    # An MLA latent has no head dimension to shard and wkv/wk are replicated,
    # so all eight ranks would otherwise hold byte-identical copies and drag
    # each one across PCIe on every hit.  Rank 0 instead stages a hit into its
    # own GPU once and broadcasts it over NVLink, so a restore costs one PCIe
    # crossing rather than eight.  That also retires an unenforceable
    # invariant: nothing now requires eight independent allocators to agree on
    # an offset, because only one of them exists.
    # The staging pages are page-locked and pooled.  One arena locks them
    # once and hands out byte offsets, so a write takes the DMA fast path
    # without an allocator in front of it, and the arena itself stays
    # ignorant of tensor geometry: it only ever sees a count of bytes.
    cache = (ColdCache(fields_for(past), COLD_BYTES, namespace=COLD_NS,
                       shm=HostArena(COLD_BYTES + (COLD_BYTES >> 4),
                                     pin_memory=COLD_PIN))
             if rank == 0 and COLD_BYTES > 0 else None)
    flow = PrefixFlow(engine, cache, namespace=COLD_NS, rank=rank,
                      world=dist.get_world_size() if dist.is_initialized() else 1,
                      device=device, ctrl=ctrl, cold=COLD_BYTES > 0)
    from model.vision_runtime import load_vision
    vision = load_vision(WEIGHTS, device, world=8, rank=rank)
    eng = Engine(rank, device, past, engine, spec, eos_id, span, CHUNK,
                 # One header carries [op, arg, n_rows] plus one id per row that
                 # could possibly be aboard, so its length is fixed for the run.
                 hdr=torch.zeros(HEAD + MAX_BATCH, dtype=torch.int64),
                 ctrl=ctrl, flow=flow, vision=vision)
    # Warm prompt: prefill, open, and capture *every* width this run can ever
    # replay.  A capture costs hundreds of milliseconds and stalls whatever is
    # already aboard, so taking one mid-service is what made the first batched
    # request slower than running the same requests one at a time.  The graphs
    # are keyed by width and read their slots through a row map refilled per
    # replay, so the warm rows only decide where the capture happens to start.
    warm = tuple(tok.encode('The capital of France is Paris. The capital of Italy is').ids)
    widest = MAX_BATCH
    warm_rows = [eng.open_row(warm) for _ in range(widest)]
    for b in range(1, widest + 1):
        spec.capture_b([r.state for r in warm_rows[:b]], past=past,
                       slots=tuple(r.slot for r in warm_rows[:b]))
    for r in warm_rows:
        eng.close_row(r)
    n, _ = eng.generate(warm, 16)
    torch.cuda.synchronize()
    if rank == 0:
        print('ENGINE_READY warm_tokens=%d' % n, flush=True)
    return eng


def serve_rank0(eng, jobs):
    import server.engine_server as es
    from server import engine_server

    class Strategy:
        def query(self, input_ids, max_new_tokens, temperature=1.0, image_payload=None):
            ids = [int(t) for t in input_ids]
            if not ids or max_new_tokens <= 0:
                raise ValueError('empty prompt or non-positive max_new_tokens')
            if len(ids) + 8 + eng.spec.window >= eng.row_cap:
                raise ValueError('prompt exceeds row_cap %d' % eng.row_cap)
            if image_payload is not None:
                from server.image_input import validate_payload
                validate_payload(ids, image_payload)
            out = ResultQueue(None, jobs.qsize())
            jobs.put(Job(ids, int(max_new_tokens), out,
                         temperature=float(temperature), image_payload=image_payload))
            return out

    engine_server.strategy = Strategy()
    es._state['ready'] = True

    def run():
        import uvicorn
        uvicorn.run(es.app, host=es.HOST, port=es.PORT, log_level='warning')

    threading.Thread(target=run, name='rpc', daemon=True).start()


def _bind_cpu_affinity():
    # Pin each rank to a private CPU slice on the NUMA node that owns its GPU
    # (GPU0-3 -> node0, GPU4-7 -> node1).  Free-floating workers drift across
    # nodes and their launch jitter shows up as all-reduce wait on peers.
    try:
        rank = int(os.environ.get('LOCAL_RANK', '0'))
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 64:
            return
        half = len(cpus) // 2
        pool = cpus[:half] if rank < 4 else cpus[half:]
        per = len(pool) // 4
        sl = pool[(rank % 4) * per:(rank % 4 + 1) * per]
        os.sched_setaffinity(0, set(sl))
        print('CPUBIND rank=%d cpus=%d-%d' % (rank, sl[0], sl[-1]), flush=True)
    except Exception as exc:
        print('CPUBIND failed %r' % (exc,), flush=True)


def pickup(jobs, room, *, block):
    """Whoever is at the stop, up to `room` of them.

    An empty bus blocks for its first passenger and then holds the door open
    for BOARD_GRACE_S, so a burst of requests that arrive together ride
    together.  A rolling bus takes only the ones already in line.
    """
    riders = []
    if room <= 0:
        return riders
    if block:
        riders.append(jobs.get())
        time.sleep(BOARD_GRACE_S)
    while len(riders) < room:
        try:
            riders.append(jobs.get_nowait())
        except queue.Empty:
            break
    return riders


def drive(eng, jobs, max_batch):
    """Rank 0's bus route: pick up whoever is waiting, step everyone aboard one
    round, then drop off the rows that finished.

    Rows board and leave independently.  A request that shows up while four
    others are decoding joins the very next round instead of waiting for the
    bus to empty, and a finished row hands its slot straight back to the pool.
    All the model layer sees change is the batch width, and it keeps one
    captured graph per width, so a new width costs one capture and nothing
    afterwards.
    """
    lanes, since_board = [], 0
    while True:
        # The door opens for an empty bus, and after that only once every
        # BOARD_EVERY rounds.  Between openings a new request queues: boarding
        # costs a prefill, and a prefill run mid-ride freezes every row aboard.
        if not lanes or since_board >= BOARD_EVERY:
            t0 = time.perf_counter()
            aboard = list(lanes)
            for job in pickup(jobs, max_batch - len(lanes), block=not lanes):
                try:
                    lanes.append(eng.admit(job.tokens, job.max_new,
                                           temperature=job.temperature,
                                           image_payload=job.image_payload,
                                           emit=job.out.put,
                                           cancel=job.out.cancel_handle.flag,
                                           queue_seconds=time.perf_counter() - job.submitted_at))
                except (RequestTooLong, ImageRejected) as exc:
                    # The passenger is the problem, not the bus: turn this one
                    # away and keep driving for everybody already aboard.
                    job.out.put({'type': 'error', 'error': repr(exc)})
                    print('REJECT %s' % exc, flush=True)
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    job.out.put({'type': 'error', 'error': repr(exc)})
                    raise
            if len(lanes) > len(aboard):
                # Rows already aboard sat out the whole boarding; a passenger
                # who boarded early still waited for the ones behind it.
                shut = time.perf_counter()
                for lane in lanes:
                    lane.stall_seconds += shut - (t0 if lane in aboard else lane.t1)
                since_board = 0
        riding = [lane for lane in lanes if not (lane.done or lane.check_cancel())]
        if riding:
            try:
                eng.roll(riding)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                for lane in lanes:
                    lane.emit({'type': 'error', 'error': repr(exc)})
                raise
            since_board += 1
        for lane in [lane for lane in lanes if lane.done or lane.check_cancel()]:
            eng.dismiss(lane)
            lanes.remove(lane)
            print('REQ prompt=%d new=%d rounds=%d %.1fms batch=%d' % (
                lane.prompt, lane.produced, lane.rounds,
                (time.perf_counter() - lane.t0) * 1e3, len(riding)), flush=True)


def bootstrap():
    """Bring this rank up: bind CPUs, join the groups, build and warm.

    The served worker and the debug harness under tests/ both enter here, so
    what a probe measures is the engine that serves.
    """
    _bind_cpu_affinity()
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30),
                            device_id=torch.device('cuda', rank))
    # V4 control plane: idle workers must not wait in the GPU collective group.
    ctrl = dist.new_group(backend='gloo', timeout=timedelta(days=365))
    return build(rank, torch.device('cuda', rank), ctrl)


def main():
    with torch.inference_mode():
        eng = bootstrap()
        jobs = queue.Queue()
        if eng.rank == 0:
            serve_rank0(eng, jobs)
            drive(eng, jobs, MAX_BATCH)
        else:
            # Nothing to decide here: obey rank 0's headers until it stops.
            eng.follow()


if __name__ == '__main__':
    main()
