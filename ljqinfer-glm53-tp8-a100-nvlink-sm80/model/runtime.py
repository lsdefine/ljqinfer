"""TP8 运行时: 每卡线程/流 + KV cache 页式管理 + NCCL 初始化。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from model.config import *  # noqa: F401,F403 — 本机固定常量, 全部写死


class TPRuntime:
    """Owns the 8 GPU streams and the NCCL communicators. ``run`` fans a
    per-rank closure across all ranks concurrently and joins; ``all_reduce``
    sums a list of 8 rank-local tensors in place (NVLink)."""

    def __init__(self, devices: List[int]):
        assert len(devices) == TP
        self.devices = devices
        self.streams = [torch.cuda.Stream(device=d) for d in devices]
        self.comms = None  # ncclCommInitAll handle, set in Engine.load()
        # Prefill crosses this barrier hundreds of times.  Eight persistent,
        # device-bound workers avoid both thread creation and executor submits.
        # One communicator has one barrier generation. Serialize callers
        # because prefill and decode can enter run() from different threads.
        self._run_lock = threading.Lock()
        self._start = threading.Barrier(TP + 1)
        self._done = threading.Barrier(TP + 1)
        self._fn: Optional[Callable[[int], None]] = None
        self._errs: List[Optional[BaseException]] = [None] * TP
        self._workers = [threading.Thread(
            target=self._worker, args=(r,), daemon=True, name=f"ljq-tp-{r}")
            for r in range(TP)]
        for worker in self._workers:
            worker.start()

    def _worker(self, r: int) -> None:
        with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
            while True:
                self._start.wait()
                fn = self._fn
                if fn is not None:
                    try:
                        fn(r)
                    except BaseException as error:
                        self._errs[r] = error
                    finally:
                        # Persistent workers must not retain a closure that captures
                        # prompt-sized CUDA tensors while idle between generations.
                        fn = None
                self._done.wait()

    def run(self, fn: Callable[[int], None]) -> None:
        """Serialize one communicator generation; ranks still run in parallel."""
        with self._run_lock:
            self._run_serial(fn)

    def _run_serial(self, fn: Callable[[int], None]) -> None:
        """Execute fn(rank) on persistent device-bound workers and join."""
        self._fn = fn
        self._errs[:] = [None] * TP
        try:
            self._start.wait()
            self._done.wait()
        finally:
            # Do not keep the last closure (and its CUDA tensor captures) alive.
            self._fn = None
        for error in self._errs:
            if error is not None:
                raise error

    def all_reduce(self, shards: List[torch.Tensor]) -> None:
        """In-place sum-allreduce across the 8 ranks (one of two per block)."""
        torch.cuda.nccl.all_reduce(
            shards, op=torch.cuda.nccl.SUM,
            streams=self.streams, comms=self.comms)


@dataclass
class KVCache:
    """One logical KV state backed by a persistent pool and an attention workspace.

    TP8 keeps both roles on the same replicated allocation.  Request-local
    caches share those tensors and only own their page tables and length.
    """
    max_len: int
    # Persistent KV storage: [physical_page, KV_PAGE_SIZE, 576] per layer/rank.
    pool: List[List[torch.Tensor]]
    # Hot attention storage.  TP8 aliases ``pool`` exactly (zero allocation/copy).
    workspace: List[List[torch.Tensor]]
    # Workspace logical_page -> physical_page, one CUDA table per rank.
    page_table: List[torch.Tensor]
    length: int = 0

    @property
    def logical_pool_pages(self) -> int:
        """Logical capacity exposed to request admission."""
        return int(self.pool[0][0].shape[0]) if self.pool and self.pool[0] else 0

    @classmethod
    def alloc(cls, rt: TPRuntime, dtype=torch.float16,
              n_layers: int = N_LAYER) -> "KVCache":
        num_pages = EXECUTION_LEN // KV_PAGE_SIZE
        pool = [[torch.empty(num_pages, KV_PAGE_SIZE, CACHE_DIM,
                             dtype=dtype, device=f"cuda:{d}")
                 for d in rt.devices] for _ in range(n_layers)]
        page_table = [torch.arange(num_pages, dtype=torch.int64,
                                   device=f"cuda:{d}") for d in rt.devices]
        # The current TP8 implementation uses one replicated allocation for
        # both persistent storage and the attention-visible hot workspace.
        return cls(max_len=EXECUTION_LEN, pool=pool, workspace=pool,
                   page_table=page_table)

    def bind_request(self, max_len: int,
                     page_indices: List[int]) -> "KVCache":
        """Create a request-local mapping over this TP8 storage owner."""
        tables = [torch.tensor(page_indices, dtype=torch.int64,
                               device=table.device)
                  for table in self.page_table]
        cache = KVCache(max_len=max_len, pool=self.pool,
                        workspace=self.workspace, page_table=tables)
        cache.validate_page_tables()
        return cache

    def pool_tokens(self, layer: int, rank: int) -> torch.Tensor:
        """Flat token view of persistent TP8 storage for cold-KV transfer."""
        return self.pool[layer][rank].view(-1, CACHE_DIM)

    def validate_page_tables(self, *, require_identity: bool = False) -> None:
        """Validate the current TP8 workspace and request page mappings."""
        if self.workspace is not self.pool:
            raise RuntimeError("TP8 KV workspace must alias the persistent pool")
        if (len(self.page_table) != TP or not self.pool or
                len(self.pool[0]) != TP):
            raise ValueError("paged KV must contain exactly one table/pool per TP rank")
        pool_pages = self.logical_pool_pages
        identity = list(range(pool_pages))
        for r, table in enumerate(self.page_table):
            if table.dtype != torch.int64 or table.dim() != 1 or not table.is_cuda:
                raise ValueError(f"rank {r} page table must be CUDA int64 [pages]")
            values = table.tolist()  # synchronizes pending table updates before kernels read it
            if not values:
                raise ValueError(f"rank {r} page table must contain at least one page")
            if len(set(values)) != len(values):
                raise ValueError(f"rank {r} page table contains duplicate physical pages")
            if min(values) < 0 or max(values) >= pool_pages:
                raise ValueError(
                    f"rank {r} page table references outside physical pool [0,{pool_pages})")
            if require_identity and values != identity:
                raise RuntimeError(
                    "identity mapping over the complete physical pool is required")


def _nccl_init_all(devices):
    """Create one explicit 8-rank NCCL communicator group in this process."""
    if len(devices) != TP:
        raise ValueError(f"expected {TP} CUDA devices, got {len(devices)}")
    devs = [torch.device(d) for d in devices]
    if any(d.type != "cuda" or d.index is None for d in devs):
        raise ValueError("NCCL ranks must be explicit CUDA devices")
    if len({d.index for d in devs}) != TP:
        raise ValueError("NCCL ranks must use 8 distinct CUDA devices")

    uid = torch.cuda.nccl.unique_id()
    comms = [None] * TP
    errs: List[Optional[BaseException]] = [None] * TP

    def _init(r: int):
        try:
            with torch.cuda.device(devs[r]):
                comms[r] = torch.cuda.nccl.init_rank(TP, uid, r)
        except BaseException as e:
            errs[r] = e

    threads = [threading.Thread(target=_init, args=(r,)) for r in range(TP)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for error in errs:
        if error is not None:
            raise error
    return comms
