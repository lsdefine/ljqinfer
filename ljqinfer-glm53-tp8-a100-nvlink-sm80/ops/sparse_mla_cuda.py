import os
from pathlib import Path
from functools import lru_cache
@lru_cache(None)
def extension():
 from torch.utils.cpp_extension import load
 os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0');os.environ.setdefault('MAX_JOBS','2')
 return load(name='glm53_sparse_mla_pair_core',sources=[str(Path(__file__).with_suffix('.cu'))],extra_cuda_cflags=['-O3'],verbose=False)
def forward(q,pool,table,idx,pos,ctx,out,ws=None,**kw):
 return extension().forward(q,pool,idx,table,pos,ctx,out,1/16)
