"""Batched compressor-row publish (decode): one launch, no device staging."""
from functools import lru_cache
from pathlib import Path
import os


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    return load(name='ljq_paged_commit',
                sources=[str(Path(__file__).resolve().parent / 'cuda' / 'paged_commit.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


def commit_pair(ckv_pool, index_pool, table, ck, ik, slots, starts, counts):
    """Publish `counts[b]` rows of request b starting at row `starts[b]` of slot
    `slots[b]`.  The plan rides along as kernel arguments, so a step costs one
    launch and no host/device synchronisation."""
    extension().paged_commit_pair(ckv_pool, index_pool, table, ck, ik,
                                  list(slots), list(starts), list(counts))
