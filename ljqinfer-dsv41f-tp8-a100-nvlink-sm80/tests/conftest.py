"""Per-test numerical baseline.

model/prefill_build.py enables TF32 process-wide (a deliberate choice for the
released engine). Once any test builds an engine, every later float32 matmul in
the same pytest process silently drops to ~1e-3 accuracy, so the float32
diagnostic comparisons fail depending on collection order. Pin the baseline
before each test; engine builds are free to turn TF32 back on for themselves.
"""
import pytest
import torch


@pytest.fixture(autouse=True)
def _fp32_baseline():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
