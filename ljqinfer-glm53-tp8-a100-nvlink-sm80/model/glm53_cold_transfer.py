"""Single-copy public cold KV, restored H2D once then fanned out over NVLink.

GLM52 uses leader H2D + peer copies in one process. Here TP ranks are separate
processes: rotate public-field ownership to use every PCIe link, then use NCCL
broadcast into the final GPU pool. Draft K/V heads are genuine private shards.
No collectives run in the asynchronous writeback worker.
"""
import torch
import torch.distributed as dist


class ColdTransfer:
    def __init__(self, fields, device):
        self.device = device
        self.distributed = dist.is_initialized() and dist.get_world_size() > 1
        self.rank = dist.get_rank() if self.distributed else 0
        self.world = dist.get_world_size() if self.distributed else 1
        public = [k for k in fields if k[0] in ('target', 'index')]
        self.owners = {k: i % self.world for i, k in enumerate(public)}
        self.local_keys = tuple(k for k in fields
                                if self.owners.get(k, self.rank) == self.rank)

    def common_prefix(self, hot, cold):
        if not self.distributed:
            return hot, cold
        # Each local store has its own physical allocator/eviction outcomes.
        # MIN is conservative and safe even when backing span boundaries differ.
        counts = torch.tensor([hot, cold], dtype=torch.int64, device=self.device)
        dist.all_reduce(counts, op=dist.ReduceOp.MIN)
        return tuple(counts.tolist())

    def restore(self, fields, cold, lease, copy_stream):
        count = lease.token_count
        spans = cold.spans(lease)
        main = torch.cuda.current_stream(self.device)
        copy_stream.wait_stream(main)
        ready = {}
        try:
            # Different ranks copy DIFFERENT public fields, plus private draft
            # shards, concurrently. Events pipeline H2D with the NVLink fanout.
            with torch.cuda.stream(copy_stream):
                for key in self.local_keys:
                    for entry_id, start, end in spans:
                        base = cold.entries[entry_id].start
                        fields[key][start:end].copy_(
                            cold.storage[key][entry_id][start-base:end-base],
                            non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(copy_stream)
                    ready[key] = event
            if self.distributed:
                for key, owner in self.owners.items():
                    if owner == self.rank:
                        main.wait_event(ready[key])
                    # A leading token slice is contiguous, including bound
                    # request views. No packing / whole-context GPU temporary.
                    work = dist.broadcast(fields[key][:count], src=owner,
                                          async_op=True)
                    # wait() orders the current CUDA stream; it need not block
                    # the host until the transfer finishes.
                    work.wait()
            main.wait_stream(copy_stream)
        finally:
            # Lease release must not let the CPU allocator reuse H2D sources.
            copy_stream.synchronize()
