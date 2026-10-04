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
        self.table = torch.full((n_slots, self.max_pages), -1,
                                dtype=torch.int64, device=device)
        # Host-only staging survives table migration; no hot-path tensor allocation.
        self._page_ids = torch.empty(self.max_pages, dtype=torch.int64, device="cpu")
        self._page_ids_np = self._page_ids.numpy()
        self._owned = [[] for _ in range(n_slots)]
        self._free = list(range(n_pages - 1, -1, -1))

    def _check_slot(self, slot):
        if not 0 <= slot < self.n_slots:
            raise IndexError("slot out of range")

    def to_(self, device):
        self.table = self.table.to(device)
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
            pages = self._free[-extra:][::-1]
            self._page_ids_np[:extra] = pages
            # Complete H2D before staging reuse (also safe across to_ / destruction).
            self.table[slot, have:required].copy_(
                self._page_ids[:extra], non_blocking=False)
            del self._free[-extra:]
            self._owned[slot].extend(pages)

    def release(self, slot):
        self._check_slot(slot)
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
        rows = torch.arange(start, end, device=self.table.device)
        return self.table[slot, rows // rows_per_page] * rows_per_page + rows % rows_per_page


class PagedPool:
    def __init__(self, pt, ratio, dim, device=None, dtype=None, reserve=0):
        if ratio not in (1, 2) or dim <= 0:
            raise ValueError("invalid source geometry")
        if reserve < 0:
            raise ValueError("invalid reserved page count")
        # Reserved pages sit above every page the table can hand out. They are
        # scratch for readers that must see uncommitted rows in page form
        # without ever touching canonical history.
        self.pt, self.ratio, self.reserve = pt, ratio, reserve
        self.rpp = pt.page_tokens // ratio
        self.data = torch.zeros(pt.n_pages + reserve, self.rpp, dim,
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

    def bind_read(self, out, *, library=None):
        """Fixed-output NPU prefix gather; host reserves pages before replay."""
        from ops.prefill.attention import build_paged_read
        if not 0 < len(out) <= self.pt.max_seq // self.ratio:
            raise ValueError("prefix capacity out of range")
        return build_paged_read(self.data, self.pt.table, out, library=library)
