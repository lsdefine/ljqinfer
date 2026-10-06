"""TP8 generation worker behind server/engine_server.py.

Mirrors the S-side strategy.decode_worker boundary: this module owns Engine
and build(); NativeDecode owns the single target/NativeDSpark execution path.
"""
import queue
import threading
from math import lcm
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from model.checkpoint_weights import DeviceWeights
from model.capacity import allocate_past


@dataclass(frozen=True)
class EngineConfig:
    cache_root: str
    rank: int
    pool_tokens: int = 2097152
    slots: int = 4
    max_seq: int = 1048576
    snapshot_root: str | None = None


class Engine:
    def __init__(self, config):
        if config.rank not in range(8):
            raise ValueError('TP8 rank required')
        self.config = config
        self.weights = self.past = None

    def load(self, *, verify_copy=False):
        import torch
        import torch_npu
        if self.weights is not None:
            raise RuntimeError('engine already loaded')
        # Avoid eager-prefill allocator flushes beside the persistent decode graphs.
        torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
        torch.npu.set_device(self.config.rank)
        weights = DeviceWeights(self.config.cache_root, self.config.rank,
                                torch.device('npu', self.config.rank))
        if self.config.snapshot_root is not None:
            from model.snapshot import load_snapshot
            self.weights = load_snapshot(weights, self.config.snapshot_root,
                                         verify_copy=verify_copy)
        else:
            self.weights = weights.load(verify_copy=verify_copy)
        from model.checkpoint_weights import HostWeights
        from pathlib import Path
        host_root = (Path(self.config.snapshot_root) / 'host'
                     if self.config.snapshot_root is not None else self.config.cache_root)
        self.host = HostWeights(host_root, self.weights.manifest['source_sha256'])
        c = self.config
        self.past = allocate_past(f'npu:{c.rank}', pool_tokens=c.pool_tokens,
                                  slots=c.slots, max_seq=c.max_seq)
        return self

    def initialize_prefill(self, hasher, host_tables, *, chunk, group=None,
                           capacity=None, library=None):
        import torch
        import torch.distributed as dist
        from model.prefill_build import PrefillParallel, build_prefill
        from model.prefill_config import released_config
        if self.weights is None or self.past is None:
            raise RuntimeError('load weights and Past before prefill')
        if getattr(self, 'compute', None) is not None:
            raise RuntimeError('prefill already initialized')
        if dist.get_rank(group) != self.config.rank:
            raise ValueError('process-group rank differs from weight rank')
        if str(dist.get_backend(group)).lower() != 'hccl':
            raise ValueError('NPU prefill requires HCCL')
        capacity = self.past.max_seq if capacity is None else capacity
        if (type(chunk) is not int or not 1 <= chunk <= 12288
                or type(capacity) is not int
                or not max(chunk, 2) <= capacity <= self.past.max_seq):
            raise ValueError('invalid chunk or context capacity')
        owned = tuple(self.host.engrams.values())
        if any(not any(table is value for value in owned)
               for table in host_tables.values()):
            raise ValueError('host tables must belong to loaded weights')
        torch.npu.set_device(self.config.rank)
        parallel = PrefillParallel(group)
        try:
            compute = build_prefill(released_config(), self.weights.data,
                hasher, host_tables, device=f'npu:{self.config.rank}',
                length=chunk, parallel=parallel, past=self.past,
                capacity=capacity, library=library)
        except BaseException:
            parallel.close()
            raise
        self.compute, self.prefill_parallel = compute, parallel
        self.prefill_chunk = chunk
        return self

    def close_prefill(self):
        if getattr(self, 'decode_compute', None) is not None:
            raise RuntimeError('close decode before releasing shared prefill resources')
        for name in ('compute', 'prefill_parallel'):
            owner = getattr(self, name, None)
            if owner is not None:
                owner.close()
                setattr(self, name, None)
        scratch = getattr(self, 'prefill_workspace', None)
        if scratch is not None:
            import torch
            from ops.prefill.native import ops
            torch.npu.synchronize(self.weights.device)
            ops.unbind_workspace(scratch)
            self.prefill_workspace = None

    # Eager prefill has no persistent score arena to lend to decode.
    # NativeDecode owns and budgets its graph storage and weight conversions.
    def _bound_decode(self, past):
        if past is not self.past:
            raise ValueError('decode must use engine-owned Past')
        compute = getattr(self, 'decode_compute', None)
        if compute is None or compute.state == 'closed':
            raise RuntimeError('bind_decode must succeed before execution')
        return compute

    def close_decode(self):
        compute = getattr(self, 'decode_compute', None)
        if compute is not None:
            compute.close()
            self.decode_compute = None

    def _bound_compute(self, past):
        decode = getattr(self, 'decode_compute', None)
        if decode is not None and decode.state not in ('idle', 'closed'):
            raise RuntimeError('drain and discard decode before mutating shared Past')
        if past is not self.past:
            raise ValueError('execution must use engine-owned Past')
        if self.compute is None:
            raise RuntimeError('initialize prefill before execution')
        return self.compute

    def _prefill(self, tokens, *, slot, start, history_tokens=(), replay=False):
        compute = self._bound_compute(self.past)
        if start < 0 or start + len(tokens) > compute.capacity:
            phase = 'replay' if replay else 'prefill'
            raise ValueError(f'{phase} exceeds bound rotary capacity')
        execute = compute.replay if replay else compute.forward
        return execute(tokens, past=self.past, slot=slot, start=start,
                       history_tokens=history_tokens)

    def bind_cold_cache(self, *, budget_bytes, namespace):
        """Rank zero owns pooled host KV; all ranks restore via TP broadcast."""
        if getattr(self, 'cold', None) is not None:
            raise RuntimeError('cold cache already bound')
        self.cold = None
        self.cold_enabled = budget_bytes > 0
        align = 1
        for source in self.past.sources.values():
            align = lcm(align, int(source.ratio))
        self.cold_align = align
        if self.cold_enabled and self.rank == 0:
            from model.cold import fields_for
            from strategy.cold_kv import ColdCache
            from strategy.host_arena import HostArena
            arena = HostArena(budget_bytes)
            self.cold = ColdCache(fields_for(self.past), budget_bytes,
                                  namespace=namespace, pin_memory=True, shm=arena)
        return self.cold

    def _prefill_spans(self, start, total):
        # The first cold-hit chunk also carries a bounded replay window.
        first = min(start + self.chunk - min(start, self.past.window), total)
        return ((start, first),) + tuple(
            (offset, min(offset + self.chunk, total))
            for offset in range(first, total, self.chunk))

    def _cold_lookup(self, tokens):
        cache = getattr(self, 'cold', None)
        align = getattr(self, 'cold_align', 1)
        limit = max(0, (len(tokens) - 1) // align * align)
        lease = None
        if cache is not None:
            lease = cache.lookup(tokens[:limit], namespace=cache.namespace)
            keep = min(lease.token_count, limit) // align * align
            if keep != lease.token_count:
                lease.close()
                lease = cache.lookup(tokens[:keep], namespace=cache.namespace)
                if lease.token_count != keep:
                    lease.close()
                    lease = cache.lookup((), namespace=cache.namespace)
        return lease

    def uncached_tokens(self, tokens, images=None):
        if images or not getattr(self, 'cold_enabled', False):
            return len(tokens)
        lease = self._cold_lookup(tokens)
        try:
            return len(tokens) - (lease.token_count if lease is not None else 0)
        finally:
            if lease is not None:
                lease.close()

    def _cold_restore(self, row, tokens):
        """Import the longest chunk-aligned cached prefix and replay its window.

        Restore compressed sources; the first append joins the missing sliding
        window replay with the uncached suffix in one encoder call.
        """
        cache = getattr(self, 'cold', None)
        if not getattr(self, 'cold_enabled', False) or self.past.image_spans.get(row.slot):
            return 0
        from model.cold import restore_prefix_tp
        past = self.past
        t_look = time.perf_counter()
        lease = self._cold_lookup(tokens)
        row.lease = lease
        row.cold_lookup_seconds = time.perf_counter() - t_look
        t_load = time.perf_counter()
        import torch.distributed as dist
        hit = restore_prefix_tp(cache, past, row.slot, lease, rank=self.rank,
                                world=dist.get_world_size(group=self.ctrl),
                                device=self.weights.device, group=self.ctrl,
                                workspace=self.restore_workspace)
        if not hit:
            return 0
        row.cold_load_seconds = time.perf_counter() - t_load
        row.hit = hit
        return hit

    def _cold_store(self, row, tokens, end):
        """Publish one committed chunk; the lease advances to the new endpoint."""
        cache = getattr(self, 'cold', None)
        if cache is None or row.lease is None or self.past.image_spans.get(row.slot):
            return
        from model.cold import store_chunk
        align = getattr(self, 'cold_align', 1)
        end = end // align * align
        if end <= row.lease.token_count:
            return
        t_store = time.perf_counter()
        # A prefill only queues its kernels.  store_chunk's first read of the
        # slot blocks until they retire, so without an explicit drain that wait
        # is billed to the store and silently subtracted from model prefill
        # (see the metrics assembly below).  Drain first and split the two.
        self._sync_rows()
        t_drained = time.perf_counter()
        stored = end - row.lease.token_count
        try:
            new = store_chunk(cache, self.past, row.slot, row.lease,
                              tuple(tokens[row.lease.token_count:end]),
                              namespace=cache.namespace)
        except MemoryError:
            # Stop caching this request: a later suffix may exceed one chunk.
            # Retain committed cache entries, but release this request's pins.
            self._cold_drop(row)
            row.cold_drain_seconds += t_drained - t_store
            row.cold_store_seconds += time.perf_counter() - t_drained
            return
        old, row.lease = row.lease, new
        old.close()
        row.cold_drain_seconds += t_drained - t_store
        row.cold_store_seconds += time.perf_counter() - t_drained
        row.cold_stored_blocks += 1
        row.cold_stored_tokens += stored

    def _cold_drop(self, row):
        lease, row.lease = getattr(row, 'lease', None), None
        if lease is not None:
            lease.close()

    def bind_scheduler(self, *, eos_id, ctrl, chunk=None):
        """Startup only, after prefill binding and B1..B4 decode capture.

        ctrl is the existing CPU/Gloo rank-0 control group (not an HCCL
        compute owner). No tokenizer, weights or second engine is constructed.
        Temperature is per row: 0 is greedy, >0 samples in the window.
        """
        import torch
        import torch.distributed as dist
        dec = self._bound_decode(self.past)
        if getattr(self, 'rows', None) is not None:
            raise RuntimeError('scheduler already bound')
        if (dec.state != 'idle' or self.past.n_slots < 4
                or any(b not in dec.plans or any(getattr(dec.plans[b], g) is None
                       for g in ('graph', 'seed_graph', 'draft_graph'))
                       for b in range(1, 5))):
            raise RuntimeError('capture B1..B4 before attaching the scheduler')
        if (ctrl is None or str(dist.get_backend(ctrl)).lower() != 'gloo'
                or dist.get_world_size(ctrl) != 8
                or dist.get_rank(ctrl) != self.config.rank):
            raise ValueError('ordered TP8 Gloo control group required')
        self._bound_compute(self.past)
        chunk = self.prefill_chunk if chunk is None else chunk
        if type(chunk) is not int or not 1 <= chunk <= self.prefill_chunk:
            raise ValueError('chunk exceeds prefill workspace capacity')
        if type(eos_id) is not int or not 0 <= eos_id < 129280:
            raise ValueError('valid EOS token required')
        self.rank, self.ctrl, self.chunk = self.config.rank, ctrl, chunk
        self.eos_id, self.max_batch = eos_id, 4
        # Per-request logical capacity; physical pages are reserved at admission.
        self.row_cap = min(self.past.max_seq, self.compute.capacity, dec.capacity,
                           self.past.pt.n_pages * self.past.page_tokens)
        self.reservations = {}
        if self.row_cap <= 7:
            raise ValueError('no Q6 headroom')
        self.hdr = torch.zeros(HEAD + self.max_batch, dtype=torch.int64)
        self.prompt_buf = torch.empty(self.row_cap, dtype=torch.int64)
        self.prefill_logits = torch.empty((1, self.compute.c['vocab_size']),
                                          dtype=torch.float32, device='cpu', pin_memory=True)
        self.rows, self.next_id, self.failed = {}, 0, None
        self.pending_spans = []
        return self

    def _scheduler_live(self):
        if not hasattr(self, 'rows'):
            raise RuntimeError('bind_scheduler before generation')
        if self.failed is not None:
            raise RuntimeError('worker failed; restart before slot reuse') from self.failed

    def _sync_rows(self):
        import torch
        torch.npu.synchronize(self.weights.device)

    def _first_token(self, output):
        # Startup CPU staging: no device argmax output allocation per request.
        self.prefill_logits.copy_(output.logits, non_blocking=False)
        import torch
        if not bool(torch.isfinite(self.prefill_logits).all()):
            raise RuntimeError('non-finite prefill logits')
        return int(self.prefill_logits.reshape(-1).argmax())

    def _seed_row(self, row, output):
        import torch
        self._bound_decode(self.past).seed_prefill(
            slot=row.slot, anchor=row.first, main_hidden=output.main_hidden,
            producer_stream=torch.npu.current_stream(self.weights.device))

    def validate_request(self, tokens, max_new, temperature=0.0):
        self._scheduler_live()
        if type(temperature) not in (int, float) or temperature not in (0, 1):
            raise ValueError('temperature must be 0 or 1')
        if not isinstance(tokens, (list, tuple)) or not tokens or any(
                type(t) is not int or not 0 <= t < 129280 for t in tokens):
            raise ValueError('nonempty valid token IDs required')
        if type(max_new) is not int or max_new <= 0:
            raise ValueError('positive integer max_new_tokens required')
        room = self.row_cap - len(tokens) - Q
        if room < 1:
            raise RequestTooLong('prompt leaves no Q6 headroom under row_cap=%d' % self.row_cap)
        return min(max_new, room)

    def open_row(self, tokens, temp=0.0):
        import torch
        self.validate_request(tokens, 1, temp)
        dec, past = self._bound_decode(self.past), self.past
        if dec.state != 'idle' or not past.free_slots:
            raise RuntimeError('admission requires idle decode and a free slot')
        from model import prefill_trace
        prefill_trace.begin(self.rank, tokens)
        row = Row(tokens=list(tokens), temp=float(temp))
        # Rank-0 stream spans, read after the existing final synchronization.
        # They include launch starvation, but exclude inter-chunk cold KV work.
        spans = []
        def mark():
            event = torch.npu.Event(enable_timing=True)
            event.record()
            return event
        row.slot = past.alloc()
        if self.pending_spans:
            past.image_spans[row.slot] = tuple(self.pending_spans)
            self.pending_spans = []
        try:
            for start, end in self._prefill_spans(
                    self._cold_restore(row, tokens), len(tokens)):
                past.ensure(row.slot, end)
                begin = max(0, start-past.window) if row.slot in past.replay_pending else start
                begin_event = mark() if self.rank == 0 else None
                self._prefill(tuple(tokens[begin:end]), slot=row.slot,
                             start=begin, history_tokens=tuple(tokens[max(0, begin-3):begin]))
                if self.rank == 0:
                    spans.append((begin_event, mark()))
                past.replay_pending.discard(row.slot)
                if past.pos[row.slot] != start:
                    raise RuntimeError('prefill model must not commit position')
                past.set_pos(row.slot, end)
                self._cold_store(row, tokens, end)
            ced_begin = mark() if self.rank == 0 else None
            output = self._bound_compute(past).finish_prefill(past=past, slot=row.slot)
            ced_end = mark() if self.rank == 0 else None
            self._sync_rows()  # CED logits belong to the prefill producer lane
            prefill_trace.finish()
            if self.rank == 0:
                row.encoder_seconds = sum(a.elapsed_time(b) for a, b in spans) / 1000
                row.ced_seconds = ced_begin.elapsed_time(ced_end) / 1000
            row.first = row.anchor = self._first_token(output)
            self._seed_row(row, output)
            return row
        except BaseException as exc:
            self.failed = exc
            self._cold_drop(row)
            # An uncertain seed/commit is quarantined, never put back on the
            # free list. Idle prefill failures can be drained and wiped safely.
            if dec.state == 'idle':
                self._sync_rows()
                dec.release(row.slot)
                past.release(row.slot)
            raise

    def step_rows(self, rows):
        self._scheduler_live()
        dec, past = self._bound_decode(self.past), self.past
        slots = tuple(row.slot for row in rows)
        if not 1 <= len(rows) <= self.max_batch or len(set(slots)) != len(slots):
            raise ValueError('one to four distinct live rows required')
        if dec.state != 'idle' or any(s < 0 or s in past.free_slots for s in slots):
            raise RuntimeError('step requires idle decode and live slots')
        starts = [past.pos[s] for s in slots]
        if any(n + Q > self.row_cap for n in starts):
            raise RequestTooLong('Q6 exceeds reserved row capacity')
        try:
            for slot, start in zip(slots, starts):
                past.ensure(slot, start + Q)
            # Page-table uploads run on the caller lane, not NativeDecode's
            # private stream. Publish them before proposal/verification reads.
            self._sync_rows()
            plan = dec.begin(slots=slots, temps=[r.temp for r in rows],
                             history_tokens=[tuple(r.tokens[-3:]) for r in rows])
            # begin has already synchronized NativeDSpark's startup D2H buffer.
            # Token zero is the previously emitted anchor, NOT a new output.
            proposals = plan.host['tokens'].numpy().tolist()
            dec.verify()
            dec.commit()
            receipts = dec.finish()
            if len(receipts) != len(rows) or len(proposals) != len(rows):
                raise RuntimeError('decode batch receipt width mismatch')
            for row, slot, start, ids, receipt in zip(rows, slots, starts, proposals, receipts):
                if (len(receipt) != 8 or receipt[0:2] != [slot, start]
                        or receipt[4] != 1 or receipt[5] or not 1 <= receipt[2] <= Q
                        or len(ids) != Q or ids[0] != row.anchor
                        or any(type(t) is not int or not 0 <= t < 129280 for t in ids)
                        or not 0 <= receipt[3] < 129280
                        or past.pos[slot] != start + receipt[2]):
                    raise RuntimeError('invalid decode output receipt')
            output = []
            for row, ids, receipt in zip(rows, proposals, receipts):
                n, anchor = receipt[2:4]
                row.tokens.extend(ids[:n])  # committed history excludes new anchor
                row.anchor = anchor
                output.append(ids[1:n] + [anchor])
            return output
        except BaseException as exc:
            self.failed = exc
            # No discard after commit submission, no cursor rollback, no reuse.
            raise

    def close_row(self, row):
        if row.slot < 0:
            return
        self._scheduler_live()
        dec = self._bound_decode(self.past)
        if dec.state != 'idle':
            raise RuntimeError('cannot release an in-flight or failed decode row')
        try:
            self._sync_rows()
            dec.release(row.slot)
            self.past.release(row.slot)
            self._sync_rows()  # release zeroing must precede a different stream's reuse
            row.slot = -1
            self._cold_drop(row)
        except BaseException as exc:
            self.failed = exc
            raise

    def exchange_images(self, count, images=None):
        """Validated patches only; transport and output storage is preallocated."""
        import torch
        import torch.distributed as dist
        from server.image_input import MAX_IMAGES
        if not 0 < count <= MAX_IMAGES:
            raise RuntimeError('invalid image count')
        vision, spans, offset = self.vision, [], 0
        slot = self.past.free_slots[-1]
        for i in range(count):
            meta = vision.meta
            if self.rank == 0:
                image = images[i]
                meta.numpy()[:] = (image['start'], image['length'], *image['grid'],
                                   *image['llm_grid'], *image['patches'].shape)
            dist.broadcast(meta, 0, group=self.ctrl)
            start, length, gh, gw, lh, lw, rows, cols = meta.tolist()
            if (rows != gh*gw or cols != 588 or rows > vision.max_patches or
                    offset + length > vision.outputs.shape[1]):
                raise RuntimeError('invalid validated vision metadata')
            patches = vision.patch_host[:rows]
            if self.rank == 0:
                patches.copy_(torch.from_numpy(images[i]['patches']))
            dist.broadcast(patches, 0, group=self.ctrl)
            device_patches = vision.patch_device[:rows]
            device_patches.copy_(patches)
            image = dict(start=start, length=length, grid=(gh,gw), llm_grid=(lh,lw))
            features = vision.encode(device_patches, image,
                                     vision.outputs[slot, offset:offset+length])
            spans.append((start, features))
            offset += length
        self.pending_spans = spans

    def command(self, op=None, arg=0, ids=(), temp_milli=0, *, tokens=None, images=None):
        """One header collective, followed by the operation's ordered payload/work."""
        import torch.distributed as dist
        ts = time.perf_counter()
        if op is not None:
            if self.rank != 0 or len(ids) > self.max_batch:
                raise ValueError('rank zero publishes at most four rows')
            self.hdr.zero_()
            self.hdr[:HEAD] = self.hdr.new_tensor((op, arg, len(ids), temp_milli))
            for i, rid in enumerate(ids):
                self.hdr[HEAD+i] = rid
        dist.broadcast(self.hdr, 0, group=self.ctrl)
        te = time.perf_counter()
        op, arg, n, temp_milli = self.hdr[:HEAD].tolist()
        if not 0 <= n <= self.max_batch:
            raise RuntimeError('invalid control header')
        ids = self.hdr[HEAD:HEAD+n].tolist()
        self.control_seconds = te-ts
        if op == OP_STOP:
            return False
        self._scheduler_live()
        if op == OP_IMAGE:
            return self.exchange_images(arg, images)
        if op == OP_OPEN:
            if n != 1 or not 0 < arg < self.row_cap or ids[0] in self.rows:
                raise RuntimeError('invalid OPEN header')
            buf = self.prompt_buf[:arg]
            if tokens is not None:
                buf.numpy()[:] = tokens
            dist.broadcast(buf, 0, group=self.ctrl)
            row = self.open_row(buf.tolist(), temp=temp_milli/1000.0)
            self.rows[ids[0]] = row
            return row
        if len(set(ids)) != len(ids) or any(i not in self.rows for i in ids):
            raise RuntimeError('invalid live row IDs')
        if op == OP_STEP:
            return self.step_rows([self.rows[rid] for rid in ids])
        if op != OP_CLOSE:
            raise RuntimeError('unknown control opcode')
        for rid in ids:
            self.close_row(self.rows[rid])
            del self.rows[rid]
            self.reservations.pop(rid, None)

    def admit(self, tokens, max_new, *, emit=None, cancel=None,
              queue_seconds=0.0, temperature=0.0, images=None):
        # Only this CPU-only preflight may reject a request and keep serving.
        try:
            limit = self.validate_request(tokens, max_new, temperature)
            images = prepare_images(tokens, images)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise RequestRejected(str(exc)) from exc
        page = self.past.page_tokens
        pages = (len(tokens) + limit + Q + page - 1) // page
        if sum(self.reservations.values()) + pages > self.past.pt.n_pages:
            raise RequestBusy('waiting for KV page capacity')
        t0 = time.perf_counter()
        rid, self.next_id = self.next_id, self.next_id + 1
        self.reservations[rid] = pages
        if images:
            self.command(OP_IMAGE, len(images.records), images=images.records)
        row = self.command(OP_OPEN, len(tokens), (rid,),
                           int(round(float(temperature)*1000)), tokens=tokens)
        lane = Lane(rid, row, self.eos_id, limit, emit=emit, cancel=cancel,
                    queue_seconds=queue_seconds, prompt=len(tokens), t0=t0,
                    t1=time.perf_counter())
        try:
            if emit:
                cold = row.cold_lookup_seconds + row.cold_load_seconds + row.cold_store_seconds
                metrics = {f'cache_{name}': getattr(row, f'cold_{attr}')
                           for name, attr in COLD_METRICS.items()}
                metrics.update(input_tokens=len(tokens), prefill_tokens=len(tokens)-row.hit,
                               cache_hit_tokens=row.hit, cache_hit_rate=row.hit/max(1, len(tokens)),
                               model_prefill_seconds=max(0.0, lane.t1-t0-cold),
                               encoder_seconds=row.encoder_seconds, ced_seconds=row.ced_seconds,
                               prefill_seconds=lane.t1-t0, queue_wait_seconds=queue_seconds,
                               chunks=(len(tokens)-row.hit+self.chunk-1)//self.chunk)
                emit({'type': 'prefill', 'metrics': metrics})
            if not lane.check_cancel():
                lane.take([row.first], initial=True)
            return lane
        except BaseException:
            self.command(OP_CLOSE, ids=[rid])
            raise



def build(rank, cache_root, *, snapshot_root=None, verify_copy=False, **limits):
    """S-side worker entry: assemble one rank's engine from resident weights."""
    config = EngineConfig(cache_root=str(cache_root), rank=int(rank),
                          snapshot_root=None if snapshot_root is None else str(snapshot_root),
                          **limits)
    return Engine(config).load(verify_copy=verify_copy)



class CancelHandle(threading.Event):
    @property
    def flag(self):
        return self

    def cancel(self):
        was = self.is_set()
        self.set()
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

    def put(self, event, *args, **kwargs):
        if event.get("type") == "prefill":
            self.metrics.update(event.get("metrics", {}))
        if event.get("type") in ("end", "error"):
            self.cancel_handle.state = "done"
        return super().put(event, *args, **kwargs)


@dataclass
class Job:
    tokens: list
    max_new: int
    out: ResultQueue
    temperature: float = 0.0
    images: object = None
    submitted_at: float = field(default_factory=time.perf_counter)


class RequestRejected(ValueError):
    """CPU preflight failed; no device work or collective was published."""


class RequestBusy(Exception):
    """Valid request waits until live rows release reserved physical pages."""


@dataclass(frozen=True)
class PreparedImages:
    records: tuple


def prepare_images(tokens, payload):
    from server.image_input import IMAGE_ID, validate_payload, patchify
    if isinstance(payload, PreparedImages):
        return payload
    if payload is None:
        if IMAGE_ID in tokens:
            raise ValueError('image placeholder without an image')
        return None
    import copy
    images = validate_payload(tokens, copy.deepcopy(payload))['images']
    return PreparedImages(tuple(dict(im, patches=patchify(im)) for im in images))


class RequestTooLong(ValueError):
    """Request exceeds fixed row capacity; reject before publishing OPEN."""


# Same control vocabulary and boarding policy as S:/strategy/decode_worker.py.
VISION_ROOT = '/data/models/DeepSeek-V4.1-Flash'
OP_STOP, OP_OPEN, OP_STEP, OP_CLOSE, OP_IMAGE = 0, 1, 2, 3, 4
COLD_METRICS = dict(stored_blocks='stored_blocks', stored_tokens='stored_tokens',
                    lookup_seconds='lookup_seconds', load_seconds='load_seconds',
                    store_seconds='store_seconds', store_drain_seconds='drain_seconds')
HEAD, Q = 4, 6
BOARD_EVERY, BOARD_GRACE_S = 4, 0.003


@dataclass
class Row:
    tokens: list
    slot: int = -1
    first: int = -1
    anchor: int = -1
    temp: float = 0.0
    hit: int = 0
    lease: object = None
    encoder_seconds: float = 0.0
    ced_seconds: float = 0.0
    cold_lookup_seconds: float = 0.0
    cold_load_seconds: float = 0.0
    cold_store_seconds: float = 0.0
    cold_drain_seconds: float = 0.0
    cold_stored_blocks: int = 0
    cold_stored_tokens: int = 0


@dataclass(eq=False)
class Lane:
    rid: int
    row: Row
    eos_id: int
    limit: int
    emit: object = None
    cancel: object = None
    queue_seconds: float = 0.0
    prompt: int = 0
    t0: float = 0.0
    t1: float = 0.0
    produced: int = 0
    rounds: int = 0
    accepted: int = 0
    control_seconds: float = 0.0
    step_seconds: float = 0.0
    stall_seconds: float = 0.0
    done: bool = False
    closed: bool = False
    reason: str = 'length'
    stats: dict = field(default_factory=dict)
    batch_steps: dict = field(default_factory=dict)

    def check_cancel(self):
        if not self.done and self.cancel is not None and self.cancel.is_set():
            self.reason, self.done = 'cancelled', True
        return self.done

    def take(self, ids, *, initial=False):
        self.rounds += not initial
        if self.check_cancel():
            return
        accepted = []
        for token in ids:
            if self.produced < self.limit:
                accepted.append(int(token))
                self.produced += 1
                if token == self.eos_id:
                    self.reason, self.done = 'eos', True
            self.done = self.done or self.produced >= self.limit
            if self.done:
                break
        if not initial:
            self.accepted += max(len(accepted)-1, 0)
        if self.emit and accepted:
            self.emit({'type': 'token', 'token_ids': accepted})


    def finish(self, eng):
        if self.closed:
            return self.stats
        eng.command(OP_CLOSE, ids=[self.rid])
        self.closed = True
        elapsed, r = time.perf_counter()-self.t1, self.rounds
        self.stats = dict(decode_steps=r, accepted_tokens=self.accepted,
                          decode_batch_steps=dict(self.batch_steps),
                          decode_tps=self.produced/elapsed if elapsed > 0 else 0.0,
                          prefill_seconds=self.t1-self.t0,
                          prefill_stall_seconds=self.stall_seconds,
                          wall_ms_per_step=1000*elapsed/r if r else 0.0,
                          control_ms_per_step=1000*self.control_seconds/r if r else 0.0,
                          model_step_host_ms=1000*self.step_seconds/r if r else 0.0,
                          queue_wait_seconds=self.queue_seconds, stop_reason=self.reason)
        if self.emit:
            self.emit(dict(type='end', reason=self.reason, **self.stats))
        return self.stats


class QueueStrategy:
    """Existing engine_server.query -> ResultQueue contract; CPU threads only."""
    def __init__(self, eng, jobs):
        self.eng, self.jobs = eng, jobs
        self.lock, self.failure = threading.Lock(), None

    def query(self, input_ids, max_new_tokens, temperature=1.0, images=None):
        if not isinstance(input_ids, (list, tuple)):
            raise ValueError('input_ids must be a list')
        ids = list(input_ids)
        self.eng.validate_request(ids, max_new_tokens, temperature)
        images = prepare_images(ids, images)
        with self.lock:
            if self.failure is not None:
                raise RuntimeError('generation worker failed') from self.failure
            out = ResultQueue(None, self.jobs.qsize())
            self.jobs.put(Job(ids, max_new_tokens, out,
                              temperature=temperature, images=images))
        return out

    def fail(self, exc):
        # Serialize against query so no request can enqueue after the drain.
        with self.lock:
            self.failure = exc
            fail_queued(self.jobs, exc)


def fail_queued(jobs, exc):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        job.out.put({'type': 'error', 'error': repr(exc)})


def serve_rank0(eng, jobs):
    """Attach only a startup-bound Engine; does not load/capture or own weights."""
    import server.engine_server as es
    eng._scheduler_live()
    if eng.rank != 0:
        raise ValueError('only rank zero serves requests')
    backend = QueueStrategy(eng, jobs)
    eng.service = backend
    es.strategy = backend
    es._state['ready'] = True

    def run():
        import uvicorn
        uvicorn.run(es.app, host=es.HOST, port=es.PORT, log_level='warning')

    threading.Thread(target=run, name='rpc', daemon=True).start()
    return backend


def pickup(jobs, room, *, block):
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


def drive(eng, jobs, max_batch=4):
    """S-side continuous batching: one main-thread owner for all device work.

    Fatal compute/collective failures propagate to the launch supervisor. There
    is deliberately no rank-local recovery, slot reuse or second decode engine.
    """
    if eng.rank != 0 or not 1 <= max_batch <= eng.max_batch:
        raise ValueError('rank-zero batch width must be between one and four')
    lanes, boarding, since_board = [], [], BOARD_EVERY
    try:
        while True:
            if not lanes or (not boarding and since_board >= BOARD_EVERY):
                t0, aboard = time.perf_counter(), list(lanes)
                boarding = boarding or pickup(jobs, max_batch-len(lanes), block=not lanes)
                while boarding:
                    job = boarding[0]
                    if job.out.cancel_handle.flag.is_set():
                        # Maintain prefill -> end ordering without allocating a slot.
                        job.out.put({'type': 'prefill', 'metrics': dict(
                            input_tokens=len(job.tokens), prefill_tokens=0,
                            prefill_seconds=0.0, model_prefill_seconds=0.0,
                            queue_wait_seconds=time.perf_counter()-job.submitted_at,
                            chunks=0)})
                        job.out.put({'type': 'end', 'reason': 'cancelled',
                                     'stop_reason': 'cancelled', 'decode_steps': 0,
                                     'accepted_tokens': 0})
                    else:
                        # Request errors are rejected BEFORE publishing an OPEN.
                        # ValueError during model execution must remain fatal.
                        try:
                            eng.validate_request(job.tokens, job.max_new, job.temperature)
                        except ValueError as exc:
                            job.out.put({'type': 'error', 'error': repr(exc)})
                            boarding.pop(0)
                            continue
                        # Keep FIFO head until active decode drains; no cache DMA yet.
                        if lanes and eng.uncached_tokens(job.tokens, job.images) > 128*1024:
                            break
                        try:
                            lane = eng.admit(job.tokens, job.max_new,
                                temperature=job.temperature, images=job.images,
                                emit=job.out.put, cancel=job.out.cancel_handle.flag,
                                queue_seconds=time.perf_counter()-job.submitted_at)
                        except RequestBusy:
                            break
                        except RequestRejected as exc:
                            job.out.put({'type': 'error', 'error': str(exc)})
                        else:
                            lanes.append(lane)
                    boarding.pop(0)
                if len(lanes) > len(aboard):
                    shut = time.perf_counter()
                    for lane in lanes:
                        lane.stall_seconds += shut-(t0 if lane in aboard else lane.t1)
                    since_board = 0
            riding = [lane for lane in lanes if not lane.check_cancel()]
            if riding:
                ts = time.perf_counter()
                outputs = eng.command(OP_STEP, ids=[lane.rid for lane in riding])
                elapsed = time.perf_counter()-ts
                for lane, ids in zip(riding, outputs):
                    lane.batch_steps[len(riding)] = lane.batch_steps.get(len(riding), 0) + 1
                    lane.control_seconds += eng.control_seconds
                    lane.step_seconds += elapsed-eng.control_seconds
                    lane.take(ids)
                since_board += 1
            for lane in lanes:
                if lane.check_cancel():
                    lane.finish(eng)
            lanes = [lane for lane in lanes if not lane.closed]
    except BaseException as exc:
        eng.failed = exc
        for lane in lanes:
            if not lane.closed and lane.emit:
                lane.emit({'type': 'error', 'error': repr(exc)})
        for job in boarding:
            job.out.put({'type': 'error', 'error': repr(exc)})
        backend = getattr(eng, 'service', None)
        if backend is not None:
            backend.fail(exc)
            import server.engine_server as es
            es._state['ready'] = False
        else:
            fail_queued(jobs, exc)
        raise


# Fixed deployment geometry: 8K chunks, 1M context, 2M physical KV pool.
CACHE_ROOT = '/dev/shm/ljqinfer_dsv41f_tp8/wcache_nz_v3'
MODEL_ROOT = '/data/models/DeepSeek-V4.1-Flash'
POOL_TOKENS, SLOTS, MAX_SEQ = 2 << 20, 4, 1 << 20
CHUNK, REPLAY, MAX_BATCH = 8192, 128, 4
COLD_NAMESPACE = 'dsv41f-w4a8-tp8'
COLD_BYTES_DEFAULT = 50 << 30


def cold_budget_bytes():
    """Total rank-zero host budget for the cold prefix store; 0 disables caching."""
    import os
    return int(os.environ.get('LJQINFER_COLD_BYTES', COLD_BYTES_DEFAULT))
# NZ TP8 full-geometry decode uses 1933579664 caller-owned tensor bytes.
# Keep 3 GiB outside the ledger for native/graph pools.
DECODE_RESERVE_BYTES = (3 << 30) - (96 << 20)


def _shutdown(eng, ctrl):
    """No shutdown collective: a failed peer may no longer be participating."""
    import torch.distributed as dist
    try:
        if eng is not None:
            service = getattr(eng, 'service', None)
            if service is not None:
                import server.engine_server as es
                es._state['ready'] = False
                service.fail(RuntimeError('generation worker stopped'))
            # Decode borrows prefill_parallel: reset all graphs before transport.
            eng.close_decode()
            eng.close_prefill()
            eng.past = eng.weights = eng.host = None
            eng.prefill_logits = eng.prompt_buf = eng.hdr = None
            # Prefill callables retain cyclic references to their tensor owners.
            import gc
            gc.collect()
    finally:
        try:
            if ctrl is not None:
                dist.destroy_process_group(ctrl)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


@contextmanager
def worker_scope(owners, label, *, close=False):
    """Preserve the primary failure; successful startup transfers ownership."""
    try:
        yield owners
    except BaseException:
        try:
            _shutdown(*owners)
        except BaseException as cleanup:
            print(f'{label}_CLEANUP_ERROR: {cleanup!r}', flush=True)
        raise
    else:
        if close:
            _shutdown(*owners)


def bootstrap():
    """Own one TP8 rank from load through capture; return only when all are ready.

    Like the S-side worker, the service and probes share this entry. Model paths
    and geometry are fixed here, not runtime algorithm switches. Only torchrun's
    rank/rendezvous environment is consumed. Failure propagates to torchrun;
    no retry, smaller shape, second engine or eager fallback is attempted.
    """
    import os
    from datetime import timedelta
    from pathlib import Path
    import torch
    import torch_npu
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from model.engram import EngramHash
    from model.decode import NativeDecode
    from model.prefill_config import released_config

    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 8 or rank != local or rank not in range(8):
        raise ValueError('single-node TP8 torchrun required')
    if dist.is_initialized():
        raise RuntimeError('bootstrap owns its process groups; already initialized')
    # Do not silently borrow the temporary leaf-test .so paths.
    root = Path(__file__).resolve().parents[1] / 'ops/decode'
    libraries = {n: str(root / ('libdecode_' + n + '.so'))
                 for n in ('norm', 'attention', 'gemm', 'window')}
    for path in libraries.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    torch.npu.set_device(local)
    torch.set_num_threads(2)
    eng = ctrl = None
    with worker_scope([eng, ctrl], 'BOOT') as owners:
        # B1 decode fires ~110 tiny TP8 collectives per speculative cycle
        # (~14KB each); they are pure latency and never overlap with compute.
        # AIV expansion issues them from vector cores: 37.2ms -> 33.5ms measured.
        # Set the variable yourself to opt out; we only supply the default.
        os.environ.setdefault('HCCL_OP_EXPANSION_MODE', 'AIV')
        dist.init_process_group('hccl', timeout=timedelta(minutes=30))
        # Idle followers block only on this CPU group, never on HCCL.
        ctrl = dist.new_group(backend='gloo', timeout=timedelta(days=365))
        owners[1] = ctrl
        with torch.inference_mode():
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT,
                trust_remote_code=True, local_files_only=True)
            config = released_config()
            eos = tokenizer.eos_token_id
            if type(eos) is not int or not 0 <= eos < config['vocab_size']:
                raise ValueError('tokenizer must declare one valid EOS token')
            metadata = EngramHash(config, tokenizer)
            # Retain the owner even if load() fails partway through.
            eng = Engine(EngineConfig(CACHE_ROOT, rank, pool_tokens=POOL_TOKENS,
                                      slots=SLOTS, max_seq=MAX_SEQ))
            owners[0] = eng
            eng.load()
            tables = {}
            for layer in config['engram_layer_ids']:
                keys = [k for k in eng.host.engrams if '.%d.' % layer in k]
                if len(keys) != 1:
                    raise ValueError(f'expected one host Engram table for layer {layer}: {keys}')
                tables[layer] = eng.host.engrams[keys[0]]
            eng.initialize_prefill(metadata, tables, chunk=CHUNK,
                                   capacity=MAX_SEQ, group=dist.group.WORLD)
            # One serialized compute lane: eager encoder and CED reuse scratch.
            # Allocate before decode budgeting; never grow during a request.
            from ops.prefill.native import ops
            scratch = torch.empty(1024**3, dtype=torch.uint8,
                                  device=eng.compute.device)
            torch.npu.synchronize()
            ops.bind_workspace(scratch)
            eng.prefill_workspace = scratch
            from model.cold import RestoreWorkspace
            eng.restore_workspace = RestoreWorkspace(eng.past, eng.weights.device)
            from model.vision_runtime import load_vision
            eng.vision = load_vision(VISION_ROOT, eng.weights.device, world=dist.get_world_size(),
                                     rank=rank).allocate_buffers(SLOTS)
            del scratch
            # Measure eager vision while all persistent input/output buffers exist.
            # Release only cached allocations here, before any graph capture.
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            vision_base = torch.npu.memory_allocated()
            eng.vision.warmup()
            vision_peak = torch.npu.max_memory_allocated() - vision_base
            vision_reserve = int(vision_peak) + (256 << 20)
            torch.npu.empty_cache()
            free, total = torch.npu.mem_get_info()
            # Account for vision separately from native/graph safety headroom.
            reserve = DECODE_RESERVE_BYTES + vision_reserve
            budget = torch.tensor([int(free) - reserve],
                                  dtype=torch.int64, device='cpu')
            dist.all_reduce(budget, op=dist.ReduceOp.MIN, group=ctrl)
            budget_bytes = int(budget.item())
            if budget_bytes <= 0:
                raise MemoryError('no decode tensor budget after prefill and explicit reserve')
            print(f'BOOT_MEMORY rank={rank} total={total} free_after_prefill={free} '
                  f'decode_tensor_limit={budget_bytes} reserve={reserve} vision_peak={vision_peak} '
                  f'pool_tokens={POOL_TOKENS} slots={SLOTS} max_seq={MAX_SEQ} chunk={CHUNK}',
                  flush=True)
            eng.decode_compute = NativeDecode(eng.weights, eng.past,
                parallel=eng.prefill_parallel, metadata=metadata, host_tables=tables,
                capacity=MAX_SEQ, budget_bytes=budget_bytes,
                batches=tuple(range(1, MAX_BATCH + 1)), libraries=libraries)
            eng._bound_decode(eng.past).capture()
            eng.bind_scheduler(eos_id=eos, ctrl=ctrl, chunk=CHUNK)
            # All ranks execute the same local path before the command loop.
            # Capture CED and exercise a full encoder chunk with decode resident.
            # Cold cache is not bound yet, so warmup cannot publish fake entries.
            for length in (128, CHUNK, CHUNK):
                row = eng.open_row([eos] * length)
                eng.close_row(row)
            if len(eng.past.free_slots) != SLOTS or eng.past.prefill_tails:
                raise RuntimeError('startup prefill did not release its request state')
            torch.npu.synchronize()
            print(f'BOOT_PREFILL_READY rank={rank} '
                  f'allocated={torch.npu.memory_allocated()} '
                  f'reserved={torch.npu.memory_reserved()}', flush=True)
            eng.vision.warmup()
            print(f'BOOT_VISION_READY rank={rank}', flush=True)
            eng.bind_cold_cache(budget_bytes=cold_budget_bytes(), namespace=COLD_NAMESPACE)
            torch.npu.synchronize()
            free_after, _ = torch.npu.mem_get_info()
            print(f'BOOT_CAPTURED rank={rank} free={free_after} '
                  f'decode_tensor_bytes={eng.decode_compute.used_bytes}', flush=True)
            # Bounded startup readiness; no HCCL idle wait or ready-before-peers.
            dist.monitored_barrier(group=ctrl, timeout=timedelta(minutes=30))
        return eng


def main():
    import torch
    eng = bootstrap()
    with worker_scope([eng, eng.ctrl], 'WORKER', close=True):
        with torch.inference_mode():
            jobs = queue.Queue()
            if eng.rank == 0:
                serve_rank0(eng, jobs)
                print('ENGINE_READY', flush=True)
                drive(eng, jobs, MAX_BATCH)
                # Only normal completion can promise followers are receiving.
                eng.command(OP_STOP)
            else:
                while eng.command() is not False:
                    pass
    # Do not broadcast STOP into a possibly failed collective. torchrun
    # terminates the remaining ranks after this process reports failure.


if __name__ == '__main__':
    main()
