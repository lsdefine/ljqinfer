# V4.1 local scoring; collective and stable selection remain external.
import os
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='v41_index_score_native27',
                sources=[str(Path(__file__).parent/'cuda'/'index_score.cu')],
                extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)
