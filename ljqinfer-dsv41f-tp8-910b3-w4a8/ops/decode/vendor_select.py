"""One fused dispatch in place of the decode index score/reduce/top-k chain.

The hand written chain scores the whole index bank on four local heads, sums
the partial scores across the eight ranks, then runs a top-k pass.  The vendor
indexer does selection in a single dispatch over the paged bank, so the chain
collapses to three cheap pieces: gather the four local query heads into the
full thirty two (six kilobytes on the wire, against the four hundred kilobyte
score reduction it replaces), point one block table at the bank, dispatch.

Two invariants shape the code.  The canonical bank is never mutated, yet the
six uncommitted rows of this step must take part in selection: the two bank
blocks that the new rows fall into are copied into a reserved scratch page and
the new rows are written there, so the operator sees an exact causal key
sequence of start+6 rows while the bank stays untouched.  And the bank and the
pending rows are shared by every index layer, so the per-step preparation runs
once for the canonical source and the reindex layers only dispatch.

Selected ids come back in the convention decode attention already uses: an id
below start addresses a committed bank row, an id at or above start addresses
pending row id-start.  No remapping, no post pass.
"""

import ctypes as C
from ops.queued import queued

import torch

_HCCL_BF16 = 11  # hccl_types.h, matches the FP32=4 constant used by the source


class VendorSelect:
    """Per-step preparation plus one dispatch per index layer.

    Owns its device buffers: they are allocated here, before capture, and
    their addresses stay fixed for the life of the object, which is what the
    bound operand set and the captured graph both require.
    """

    def __init__(self, *, bank, table, slots, start, pending, batch,
                 heads_local=4, world=8, dim=128, rows=6, block=128,
                 sparse_count=512, hccl_library=None, hccl_comm=0, device=None):
        rpp, width = bank.shape[1], bank.shape[2]
        if width != dim or rpp % block:
            raise ValueError('vendor selection needs block aligned pages of index rows')
        if not 0 < batch <= 4 or rows != 6 or table.ndim != 2:
            raise ValueError('vendor selection needs B1..B4 with six query rows')
        if bank.shape[0] <= int(table.max().item()):
            raise ValueError('vendor selection needs one page past the page table')
        dev = device or bank.device
        self.bank, self.table, self.slots, self.start = bank, table, slots, start
        self.pending, self.batch, self.rows, self.block = pending, batch, rows, block
        self.world, self.heads_local, self.dim = world, heads_local, dim
        self.heads = heads_local * world
        self.flat = bank.view(-1, dim)
        self.key = bank.view(-1, block, 1, dim)
        self.bpp = rpp // block
        self.blocks_per_slot = table.shape[1] * self.bpp
        # The reserved page is the last one; two scratch blocks per sequence.
        self.scratch = (bank.shape[0] - 1) * self.bpp
        self.total_blocks = bank.shape[0] * self.bpp
        self.bank_pages = bank.shape[0]
        self.block_table = torch.zeros(batch, self.blocks_per_slot,
                                       dtype=torch.int32, device=dev)
        self.seq_q = torch.full((batch,), rows, dtype=torch.int32, device=dev)
        self.seq_k = torch.zeros(batch, dtype=torch.int32, device=dev)
        self.query = torch.zeros(batch, rows, self.heads, dim,
                                 dtype=torch.bfloat16, device=dev)
        self.weight = torch.zeros(batch, rows, self.heads,
                                  dtype=torch.bfloat16, device=dev)
        self.gather_q = torch.zeros(world, batch * rows, heads_local, dim,
                                    dtype=torch.bfloat16, device=dev)
        self.gather_w = torch.zeros(world, batch * rows, heads_local,
                                    dtype=torch.bfloat16, device=dev)
        self.staging = torch.zeros(batch * 2 * block, dim,
                                   dtype=bank.dtype, device=dev)
        self.indices = torch.zeros(batch, rows, 1, sparse_count,
                                   dtype=torch.int32, device=dev)
        self.sparse_count = sparse_count
        self.device = dev
        self._lane = torch.arange(block, device=dev, dtype=torch.int64)
        self._row = torch.arange(rows, device=dev, dtype=torch.int64)
        self._page_lane = torch.arange(self.bpp, device=dev, dtype=torch.int64)
        self._scratch_pair = (self.scratch
                              + 2 * torch.arange(batch, device=dev, dtype=torch.int64))
        self._dst = ((self._scratch_pair[:, None] * block + self._lane[None, :])[:, None, :]
                     + torch.tensor([0, block], device=dev)[None, :, None]).reshape(-1)
        self.indexer = None
        if hccl_library is not None and hccl_comm:
            gather = hccl_library.HcclAllGather
            gather.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_int,
                               C.c_void_p, C.c_void_p]
            gather.restype = C.c_int
            self._gather = queued(gather, gather.argtypes, check_status=True)
            self._comm = C.c_void_p(hccl_comm)
        else:
            self._gather = self._comm = None

        from ops.decode.vendor_indexer import VendorIndexer
        self.indexer = VendorIndexer(self.query, self.key, self.weight,
                                      self.seq_q, self.seq_k, self.block_table,
                                      self.indices, sparse_count=self.sparse_count)

    def prepare(self):
        """Rebuild block table, scratch blocks and lengths for this step."""
        batch, block, bpp = self.batch, self.block, self.bpp
        # Inactive sequences carry slot -1 and a meaningless length; clamp so the
        # gathers stay in range. Their rows are never read back.
        slots = self.slots.clamp(min=0)
        pages = self.table.index_select(0, slots)                   # [B,M] int64
        # Unallocated table columns hold junk; the vendor kernel tiles over the
        # whole block table, so keep every entry inside the bank.
        pages = pages.clamp(0, self.bank_pages - 1)
        table = (pages[:, :, None] * bpp + self._page_lane[None, None, :])
        self.block_table.copy_(table.reshape(batch, -1))
        start = self.start
        self.seq_k.copy_(start + self.rows)
        first = torch.div(start, block, rounding_mode='floor').clamp_(0, self.blocks_per_slot - 1)
        second = torch.clamp(first + 1, max=self.blocks_per_slot - 1)
        pair = torch.stack((first, second), dim=1)                   # [B,2]
        # A block past the committed rows may sit in a page the slot never
        # got; its table entry is unset, so clamp before reading the bank.
        phys = self.block_table.gather(1, pair).long().clamp_(0, self.total_blocks - 1)
        src = (phys[:, :, None] * block + self._lane[None, None, :]).reshape(-1)
        torch.index_select(self.flat, 0, src, out=self.staging)
        self.flat.index_copy_(0, self._dst, self.staging)
        head = self._scratch_pair * block + (start - first * block)
        self.flat.index_copy_(0, (head[:, None] + self._row[None, :]).reshape(-1),
                              self.pending.reshape(-1, self.dim))
        # A sequence whose last block is the slot's last block clamps the pair to
        # one entry; both halves must then name the block that holds the new rows.
        tail = torch.where(second > first, self._scratch_pair + 1, self._scratch_pair)
        scratch = torch.stack((self._scratch_pair, tail), dim=1)
        self.block_table.scatter_(1, pair, scratch.to(torch.int32))

    def dispatch(self, iq, head_weight, selected):
        """Gather the local heads into the full set, select, publish ids."""
        batch, rows, local = self.batch, self.rows, self.heads_local
        if self._gather is not None:
            stream = C.c_void_p(torch.npu.current_stream(self.device).npu_stream)
            for send, recv in ((iq, self.gather_q), (head_weight, self.gather_w)):
                rc = self._gather(C.c_void_p(send.data_ptr()),
                                  C.c_void_p(recv.data_ptr()), send.numel(),
                                  _HCCL_BF16, self._comm, stream)
                if rc:
                    raise RuntimeError(f'vendor selection HcclAllGather failed: {rc}')
            q = self.gather_q.view(self.world, batch, rows, local, self.dim)
            w = self.gather_w.view(self.world, batch, rows, local)
            self.query.copy_(q.permute(1, 2, 0, 3, 4).reshape(batch, rows, -1, self.dim))
            self.weight.copy_(w.permute(1, 2, 0, 3).reshape(batch, rows, -1))
        else:
            self.query.copy_(iq.view(batch, rows, -1, self.dim))
            self.weight.copy_(head_weight.view(batch, rows, -1))
        self.indexer.run()
        selected.copy_(self.indices.view(batch, rows, self.sparse_count))
        return selected

    def close(self):
        self.indexer.close()
