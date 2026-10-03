"""Small explicit page allocator for the V4.1 state reference.

Allocation is host-side and precedes execution. No dependency on another engine.
This is a correctness reference, not a graph-captured paging kernel.
"""
import torch

PAGE_TOKENS = 2048


class PageTable:
    def __init__(self, n_slots, max_seq, n_pages, page_tokens=PAGE_TOKENS,
                 device=None):
        if min(n_slots, max_seq, n_pages, page_tokens) <= 0 or page_tokens % 2:
            raise ValueError("positive geometry and even page_tokens required")
        self.n_slots, self.max_seq = n_slots, max_seq
        self.n_pages, self.page_tokens = n_pages, page_tokens
        self.max_pages = (max_seq + page_tokens - 1) // page_tokens
        # What one row can really hold: its equal share of the pool, in
        # whole pages.  Static shapes size themselves off this, never off
        # max_seq, so the addressable span stays free to be huge.
        self.row_cap = min(max_seq, n_pages // n_slots * page_tokens)
        self.table = torch.full((n_slots, self.max_pages), -1,
                                dtype=torch.int64, device=device)
        self._owned = [[] for _ in range(n_slots)]
        self._free = list(range(n_pages - 1, -1, -1))
        # physical_rows() is called once per layer per write with identical
        # arguments (40 layers share one page table): each call costs ~6 tiny
        # CUDA launches, so memoize per allocation generation.
        self._rows_ver = 0
        self._rows_cache = {}

    def _check_slot(self, slot):
        if not 0 <= slot < self.n_slots:
            raise IndexError("slot out of range")

    def to_(self, device):
        self.table = self.table.to(device)
        self._rows_ver += 1
        self._rows_cache.clear()
        return self

    def n_alloc(self, slot):
        self._check_slot(slot)
        return len(self._owned[slot])

    def ensure(self, slot, pos_after):
        self._check_slot(slot)
        if not 0 <= pos_after <= self.max_seq:
            raise ValueError("position out of range")
        required = (pos_after + self.page_tokens - 1) // self.page_tokens
        have = self.n_alloc(slot)
        extra = max(0, required - have)
        if extra > len(self._free):
            raise MemoryError("KV page budget exhausted")
        if extra:
            self._rows_ver += 1
            self._rows_cache.clear()
            pages = self._free[-extra:][::-1]
            self.table[slot, have:required] = torch.tensor(
                pages, dtype=self.table.dtype, device=self.table.device)
            del self._free[-extra:]
            self._owned[slot].extend(pages)

    def release(self, slot):
        self._check_slot(slot)
        self._rows_ver += 1
        self._rows_cache.clear()
        self.table[slot].fill_(-1)
        self._free.extend(reversed(self._owned[slot]))
        self._owned[slot].clear()

    def physical_rows(self, slot, start, end, ratio):
        self._check_slot(slot)
        if ratio not in (1, 2) or self.page_tokens % ratio:
            raise ValueError("unsupported ratio")
        if not 0 <= start <= end <= self.max_seq // ratio:
            raise ValueError("row interval out of range")
        rows_per_page = self.page_tokens // ratio
        needed = (end + rows_per_page - 1) // rows_per_page
        if end > start and needed > self.n_alloc(slot):
            raise RuntimeError("rows accessed before page allocation")
        key = (slot, start, end, ratio)
        hit = self._rows_cache.get(key)
        if hit is not None and hit[0] == self._rows_ver:
            return hit[1]
        rows = torch.arange(start, end, device=self.table.device)
        phys = (self.table[slot, rows // rows_per_page] * rows_per_page
                + rows % rows_per_page)
        if len(self._rows_cache) > 64:
            self._rows_cache.clear()
        self._rows_cache[key] = (self._rows_ver, phys)
        return phys



class PagedPool:
    def __init__(self, pt, ratio, dim, device=None, dtype=None):
        if ratio not in (1, 2) or dim <= 0:
            raise ValueError("invalid source geometry")
        self.pt, self.ratio = pt, ratio
        self.rpp = pt.page_tokens // ratio
        self.data = torch.zeros(pt.n_pages, self.rpp, dim,
                                device=device, dtype=dtype)

    def to_(self, device):
        self.data = self.data.to(device)
        return self

    def read(self, slot, start, end):
        rows = self.pt.physical_rows(slot, start, end, self.ratio)
        return self.data.flatten(0, 1).index_select(0, rows)

    def write(self, slot, start, value):
        if value.ndim != 2 or value.shape[1] != self.data.shape[2]:
            raise ValueError("invalid row payload shape")
        rows = self.pt.physical_rows(slot, start, start + value.shape[0], self.ratio)
        self.data.flatten(0, 1).index_copy_(
            0, rows, value.to(device=self.data.device, dtype=self.data.dtype))
