"""Page-locked host arena: allocate the pages once, reuse them forever.

Pinned staging only pays off when the page-locked pages outlive the copy
that fills them.  Calling ``torch.empty(..., pin_memory=True)`` per write
puts a ``cudaHostAlloc`` on the hot path, and that allocation costs more
than the faster DMA returns: measured over 72 chunks of 47 MiB, per-entry
pinning ran at 0.0202 s/chunk against 0.0128 s/chunk for plain pageable
staging.  The same pages handed out of an arena keep the DMA fast path
without an allocator in front of it.

The arena is deliberately ignorant: it hands out byte offsets and views
over them.  It knows nothing about attention geometry, tensor parallel
rank, or which model produced the bytes, so one arena serves any cold
cache whose payload can be described as "so many bytes of this dtype".

Growth is incremental.  A cold budget is an upper bound, not a promise,
and page-locked memory cannot be swapped, so segments are locked only as
the cache actually fills.
"""

import threading

import torch

# Offsets are aligned so that any view() re-interpretation of the backing
# byte storage starts on a boundary every supported dtype is happy with.
_ALIGN = 512


def _aligned(nbytes):
    return (int(nbytes) + _ALIGN - 1) // _ALIGN * _ALIGN


class HostArena:
    """Byte-addressed pool of page-locked host memory.

    ``alloc``/``free``/``view`` is the whole contract.  Allocations never
    straddle a segment, which keeps every view a contiguous slice of one
    backing tensor.
    """

    def __init__(self, capacity_bytes, *, segment_bytes=1 << 30,
                 pin_memory=True):
        capacity_bytes = int(capacity_bytes)
        if capacity_bytes <= 0:
            raise ValueError('arena capacity must be positive')
        segment_bytes = _aligned(min(int(segment_bytes), capacity_bytes))
        if segment_bytes <= 0:
            raise ValueError('arena segment must be positive')
        self.capacity = capacity_bytes
        self.segment_bytes = segment_bytes
        self.pin_memory = bool(pin_memory)
        self.used_bytes = 0
        self._segments = []
        self._free = []
        self._lock = threading.Lock()

    @property
    def reserved_bytes(self):
        return sum(int(s.numel()) for s in self._segments)

    def _grow(self):
        base = self.reserved_bytes
        remain = self.capacity - base
        if remain <= 0:
            return False
        size = min(self.segment_bytes, remain)
        segment = torch.empty(size, dtype=torch.uint8, device='cpu',
                              pin_memory=self.pin_memory)
        self._segments.append(segment)
        self._free.append([(0, size)])
        return True

    def alloc(self, nbytes):
        want = _aligned(nbytes)
        if want <= 0:
            raise ValueError('allocation must be positive')
        if want > self.segment_bytes:
            raise ValueError(
                'allocation of %d bytes exceeds arena segment of %d bytes'
                % (want, self.segment_bytes))
        with self._lock:
            while True:
                for index, holes in enumerate(self._free):
                    for slot, (offset, size) in enumerate(holes):
                        if size < want:
                            continue
                        if size == want:
                            holes.pop(slot)
                        else:
                            holes[slot] = (offset + want, size - want)
                        self.used_bytes += want
                        return index * self.segment_bytes + offset
                if not self._grow():
                    raise MemoryError(
                        'host arena exhausted: %d of %d bytes in use'
                        % (self.used_bytes, self.capacity))

    def free(self, offset, nbytes):
        want = _aligned(nbytes)
        index, start = divmod(int(offset), self.segment_bytes)
        if not 0 <= index < len(self._segments):
            raise ValueError('offset %d is outside the arena' % offset)
        with self._lock:
            holes = self._free[index]
            slot = 0
            while slot < len(holes) and holes[slot][0] < start:
                slot += 1
            holes.insert(slot, (start, want))
            self.used_bytes -= want
            # Coalesce with the neighbours so a long-lived arena does not
            # decay into a list of unusable crumbs.
            if slot + 1 < len(holes):
                nxt_offset, nxt_size = holes[slot + 1]
                if start + want == nxt_offset:
                    holes[slot] = (start, want + nxt_size)
                    holes.pop(slot + 1)
            if slot:
                prev_offset, prev_size = holes[slot - 1]
                cur_offset, cur_size = holes[slot]
                if prev_offset + prev_size == cur_offset:
                    holes[slot - 1] = (prev_offset, prev_size + cur_size)
                    holes.pop(slot)

    def view(self, offset, shape, dtype):
        index, start = divmod(int(offset), self.segment_bytes)
        if not 0 <= index < len(self._segments):
            raise ValueError('offset %d is outside the arena' % offset)
        count = 1
        for dim in shape:
            count *= int(dim)
        nbytes = count * torch.empty((), dtype=dtype).element_size()
        segment = self._segments[index]
        if start + nbytes > int(segment.numel()):
            raise ValueError('view runs past the end of its segment')
        return segment[start:start + nbytes].view(dtype).view(tuple(shape))
