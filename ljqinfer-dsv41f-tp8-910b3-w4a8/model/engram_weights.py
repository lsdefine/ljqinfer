"""Host-only Engram table ABI: INT8 [rows,256], FP32 group scales [rows,8].

Tables arrive already mapped: HostWeights maps the host cache MAP_PRIVATE, so a
gather pages in only the rows it touches and a 98 GB table never becomes a
resident copy. No rank copy or full-table dequantization. This is a synchronous
CPU reference collector, NOT the future pinned-buffer/async-H2D prefetch path.
"""
import torch

ROW_BYTES = 256 + 8 * 4


class HostEngram:
    def __init__(self, weight, scale):
        if (weight.device.type != 'cpu' or scale.device.type != 'cpu'
                or weight.dtype != torch.int8 or scale.dtype != torch.float32
                or weight.ndim != 2 or weight.shape[1] != 256
                or scale.shape != (weight.shape[0], 8)
                or not weight.is_contiguous() or not scale.is_contiguous()):
            raise ValueError('expected CPU contiguous int8 [rows,256], float32 [rows,8]')
        self.weight, self.scale = weight, scale

    def gather(self, ids, *, rank=None):
        """Return ONLY selected raw rows/scales. IDs must already be valid hashes.

        With rank, input [...,24] is split into three hash columns per rank,
        matching weights.placement('layers.1.engram.wkv.weight'). No truncation
        or modulo of invalid IDs; image/padding masks belong to hash generation.
        """
        if ids.device.type != 'cpu' or ids.dtype != torch.int64:
            raise ValueError('expected CPU int64 IDs')
        if rank is not None:
            if not 0 <= rank < 8 or ids.ndim == 0 or ids.shape[-1] != 24:
                raise ValueError('rank gather expects [...,24], rank 0..7')
            ids = ids[..., rank * 3:(rank + 1) * 3]
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.weight.shape[0]):
            raise ValueError('hash row out of range')
        return self.weight[ids], self.scale[ids]