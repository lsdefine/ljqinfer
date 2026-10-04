"""Register Tensor operators once; build products live outside the source tree."""
from pathlib import Path
import os
import torch
import torch_npu
from torch.utils.cpp_extension import load


def _load():
    kernels = Path(__file__).resolve().parents[1] / 'kernels'
    npu = Path(torch_npu.__file__).parent
    cann = Path(os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest'))
    load(name='ljq_prefill_native', sources=[str(kernels / n) for n in ('prefill_torch.cpp', 'prefill_attention_torch.cpp', 'prefill_cann_torch.cpp')],
         extra_include_paths=[str(npu / 'include'), str(cann / 'include')],
         extra_cflags=['-O2', '-std=c++17'],
         extra_ldflags=[str(kernels / n) for n in ('libhc.so', 'libattention.so', 'libdq.so')] + [ f'-L{npu}/lib', '-ltorch_npu',
                        f'-Wl,-rpath,{npu}/lib', f'-Wl,-rpath,{kernels}',
                        f'-L{cann}/lib64', '-lopapi', '-lnnopbase', f'-Wl,-rpath,{cann}/lib64'],
         is_python_module=False, verbose=False)
    return torch.ops.ljq_prefill


ops = _load()
