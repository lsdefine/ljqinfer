"""Global-only prefix cache; any matched token prefix is replay-restorable."""
import torch
import torch.distributed as dist
from strategy.cold_kv import Field


def fields_for(pool):
    if not pool.sources or not pool.windows:
        raise ValueError('adapter requires configured past')
    fields = []
    for layer, source in pool.sources.items():
        for name, paged in [('main_ckv', source.ckv_pool), ('index_k', source.index_pool)]:
            fields.append(Field(('sources', layer, name), (None, paged.data.shape[-1]),
                                paged.data.dtype))
    return tuple(fields)


def _check(cache, pool, slot):
    if not 0 <= slot < pool.n_slots or slot in pool.free_slots:
        raise ValueError('slot must be allocated')
    if cache.fields != {f.key: f for f in fields_for(pool)}:
        raise ValueError('cache/past layout mismatch')


def store_prefix(cache, pool, slot, tokens, *, namespace):
    _check(cache, pool, slot)
    end = pool.pos[slot]
    tokens = tuple(tokens)
    if len(tokens) != end or end == 0:
        raise ValueError('save requires tokens matching current nonempty committed past')
    with cache.lookup(tokens, namespace=namespace) as parent:
        if parent.token_count == end:
            return False
        _store_suffix(cache, pool, slot, parent, tokens[parent.token_count:])
    return True


def _store_suffix(cache, pool, slot, parent, tokens):
    start = parent.token_count
    end = start + len(tokens)
    if end > pool.pos[slot]:
        raise ValueError('suffix exceeds the committed past')
    blob = pool.export_cold(slot, start, end)
    with cache.prepare(parent, tokens) as store:
        for key in cache.fields:
            _, layer, name = key
            value = blob['sources'][layer][name]
            store.write(key, value)
        return store.commit()


def store_chunk(cache, pool, slot, parent, tokens, *, namespace):
    """Append <=12Ki committed tokens and return a pinned endpoint.

    Parent is the caller's lease for this slot's previously stored prefix;
    the caller must preserve that semantic association (same as restore).
    Only suffix tokens are supplied. Old lease remains valid on failure.
    Caller closes the old lease AFTER receiving the new one.
    """
    if namespace != cache.namespace:
        raise ValueError('cache model/layout namespace mismatch')
    _check(cache, pool, slot)
    tokens = tuple(tokens)
    with cache.lock:
        cache._validate(parent)
        if not 0 < len(tokens) <= 12288:
            raise ValueError('chunk must contain 1..12288 tokens')
        if parent.token_count + len(tokens) > pool.pos[slot]:
            raise ValueError('chunk must immediately follow parent endpoint')
        existing = cache.lookup_suffix(parent, tokens, namespace=namespace)
        if existing is not None:
            return existing
        i = _store_suffix(cache, pool, slot, parent, tokens)
        return cache.pin_endpoint(i, namespace=namespace)


def restore_prefix(cache, pool, slot, lease):
    _check(cache, pool, slot)
    cache._validate(lease)
    if pool.pos[slot] != 0:
        raise ValueError('restore requires empty destination position')
    if not lease.ids:
        return 0
    spans = cache.spans(lease)
    # Validate all backing geometry before allocation/copies; clip only complete
    # compression groups. A row crossing the hit boundary is never loaded.
    for i, start, end in spans:
        entry = cache.entries[i]
        for (_, layer, name), values in cache.storage.items():
            r = pool.sources[layer].ratio
            if values[i].shape[0] != entry.end // r - entry.start // r:
                raise ValueError('cache/past segment geometry mismatch')
    pool.ensure(slot, lease.token_count)
    for i, start, end in spans:
        for layer, source in pool.sources.items():
            n = end // source.ratio - start // source.ratio
            source.import_cold(slot, start, end, {
                name: cache.storage[('sources', layer, name)][i][:n]
                for name in ('main_ckv', 'index_k')})
    pool.mark_cold(slot, lease.token_count)
    return lease.token_count


def _geometry(pool):
    """The (layer, name, dim, dtype, ratio) tuple order shared by every rank.

    Layer ratios differ, so rows-per-layer differ; the packed layout therefore
    follows this sequence rather than any fixed row stride.
    """
    return tuple((layer, name, paged.data.shape[-1], paged.data.dtype,
                  source.ratio)
                 for layer, source in pool.sources.items()
                 for name, paged in (('main_ckv', source.ckv_pool),
                                     ('index_k', source.index_pool)))


def _span_bytes(pool, start, end):
    return sum((end // r - start // r) * dim * dt.itemsize
               for _, _, dim, dt, r in _geometry(pool))


def pack_spans(cache, pool, spans):
    """Flatten every (span, layer, field) payload into one host byte buffer.

    The order is span-major, then `_geometry`.  A follower reproduces it from
    the broadcast header alone, so no offset ever crosses the wire.
    """
    parts = []
    for i, start, end in spans:
        for layer, source in pool.sources.items():
            n = end // source.ratio - start // source.ratio
            for name in ('main_ckv', 'index_k'):
                row = cache.storage[('sources', layer, name)][i][:n]
                parts.append(row.contiguous().view(torch.uint8).reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.uint8)
    return torch.cat(parts)


def unpack_into(pool, slot, bounds, packet):
    """Slice a packed payload back into the paged pools of this rank."""
    off = 0
    for start, end in bounds:
        for layer, source in pool.sources.items():
            n = end // source.ratio - start // source.ratio
            blob = {}
            for name, paged in (('main_ckv', source.ckv_pool),
                                ('index_k', source.index_pool)):
                dim, dt = paged.data.shape[-1], paged.data.dtype
                size = n * dim * dt.itemsize
                blob[name] = packet.narrow(0, off, size).view(dt).view(n, dim)
                off += size
            source.import_cold(slot, start, end, blob)
    return off


def restore_prefix_tp(cache, pool, slot, lease, *, rank, world, device,
                      group=None):
    """Restore a cache hit on every rank from a single host->device0 transfer.

    Rank 0 alone owns the cold cache, which is plain host memory and knows
    nothing of ranks.  It stages the whole hit into device 0 once and broadcasts
    it over the device collective, so a restore costs one PCIe crossing rather than `world`
    of them.  Followers recover the geometry from a small integer header and
    rebuild byte-identical slices.  The import itself stays per-rank: `ensure`
    and `mark_cold` touch local page state and must run everywhere.
    """
    if pool.pos[slot] != 0:
        raise ValueError('restore requires an empty destination position')
    head = torch.zeros(2, dtype=torch.int64)
    spans = ()
    if rank == 0:
        _check(cache, pool, slot)
        cache._validate(lease)
        spans = tuple(cache.spans(lease)) if lease.ids else ()
        head[0], head[1] = lease.token_count, len(spans)
    if world > 1:
        dist.broadcast(head, 0, group=group)     # gloo: host counts
    hit, n_spans = int(head[0]), int(head[1])
    if hit == 0:
        return 0
    # Cache chains grow with appended turns, not just prompt length. Send the
    # exact host-side geometry instead of imposing a fixed segment ceiling.
    edges = torch.empty((n_spans, 2), dtype=torch.int64)
    if rank == 0:
        for j, (_, start, end) in enumerate(spans):
            edges[j, 0], edges[j, 1] = start, end
    if world > 1:
        dist.broadcast(edges, 0, group=group)
    bounds = edges.tolist()
    nbytes = sum(_span_bytes(pool, s, e) for s, e in bounds)
    if rank == 0:
        host = pack_spans(cache, pool, spans)
        if host.numel() != nbytes:
            raise ValueError('packed payload contradicts pool geometry')
        packet = host.to(device)
    else:
        packet = torch.empty(nbytes, dtype=torch.uint8, device=device)
    if world > 1:
        dist.broadcast(packet, 0)                # HCCL: replicated compressed KV payload
    pool.ensure(slot, hit)
    if unpack_into(pool, slot, bounds, packet) != nbytes:
        raise ValueError('unpacked payload contradicts pool geometry')
    pool.mark_cold(slot, hit)
    return hit
