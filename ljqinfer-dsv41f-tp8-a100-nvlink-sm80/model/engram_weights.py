"""Host-only Engram table ABI: FP8 [rows,256], E8M0 bytes [rows,8].

Tables are mapped once from a complete safetensors file (checkpoint or wcache).
No rank copy or full-table dequantization. This is a synchronous CPU reference
collector, NOT the future pinned-buffer/async-H2D prefetch implementation.
"""
import torch
from safetensors import safe_open


class HostEngram:
    def __init__(self, weight, scale):
        if (weight.device.type != 'cpu' or scale.device.type != 'cpu'
                or weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8
                or weight.ndim != 2 or weight.shape[1] != 256
                or scale.shape != (weight.shape[0], 8)
                or not weight.is_contiguous() or not scale.is_contiguous()):
            raise ValueError('expected CPU contiguous FP8 [rows,256], uint8 [rows,8]')
        self.weight, self.scale = weight, scale
        self._np = None
        self._u8 = None

    @classmethod
    def open(cls, path, prefix):
        with safe_open(path, framework='pt', device='cpu') as f:
            return cls(f.get_tensor(prefix + '.weight'), f.get_tensor(prefix + '.scale'))

    def gather(self, ids, *, rank=None):
        """Return ONLY selected raw rows/scales. IDs must already be valid hashes.

        With rank, input [...,24] is split into three hash columns per rank,
        matching weights.placement('layers.1.engram.wkv.weight'). No truncation
        or modulo of invalid IDs; image/padding masks belong to hash generation.
        """
        ids = self._select(ids, rank)
        # CPU advanced indexing support for FP8 varies; index its exact bytes.
        # numpy take: torch CPU index on a mmap'd multi-GB table pays far more
        # per call than the 21 rows it moves (decode hot path, ~0.45ms/round).
        if self._np is None:
            self._np = (self.weight.view(torch.uint8).numpy(), self.scale.numpy())
        w, sc = self._np
        idx = ids.numpy()
        values = torch.from_numpy(w[idx]).view(torch.float8_e4m3fn)
        return values, torch.from_numpy(sc[idx])

    def _select(self, ids, rank):
        """Validate IDs and cut this rank's hash columns; see gather()."""
        if ids.device.type != 'cpu' or ids.dtype != torch.int64:
            raise ValueError('expected CPU int64 IDs')
        if rank is not None:
            if not 0 <= rank < 8 or ids.ndim == 0 or ids.shape[-1] != 24:
                raise ValueError('rank gather expects [...,24], rank 0..7')
            ids = ids[..., rank * 3:(rank + 1) * 3]
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.weight.shape[0]):
            raise ValueError('hash row out of range')
        return ids

    def rows_bf16(self, ids, *, rank=None, out=None):
        """gather() + dequantize in one host pass, straight into bf16 `out`.

        Splitting the two costs ~0.8ms per decode round: numpy advanced
        indexing on a mmap'd multi-GB table plus six small torch ops, all for
        21 rows. Since the host is this loop's critical path, that time is not
        hidden by anything. `out` may be pinned so the H2D can be async; it is
        ignored when too small (prefill gathers thousands of rows).
        """
        from ops.decode import engram_host
        ids = self._select(ids, rank)
        flat = ids.reshape(-1).contiguous()
        n = flat.numel()
        if out is None or out.shape[0] < n:
            out = torch.empty(n, 256, dtype=torch.bfloat16)
        out = out[:n]
        if self._u8 is None:
            self._u8 = self.weight.view(torch.uint8)
        engram_host.mod().engram_rows(self._u8, self.scale, flat, engram_host.lut(), out)
        return out.view(*ids.shape, 256)
