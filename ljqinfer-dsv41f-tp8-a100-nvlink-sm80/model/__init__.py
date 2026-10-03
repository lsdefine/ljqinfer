import os
# A100 NV12 all-to-all NVLink: the default Ring AllReduce needs 2(N-1)=14
# serialised hops, which dominates at decode's tiny 61KB payload. Tree needs
# ~2*log2(8)=6, measured 24.01us vs 36.05us per all-reduce (-33%).
os.environ.setdefault('NCCL_ALGO', 'allreduce:tree')

"""V4.1 model state primitives."""
