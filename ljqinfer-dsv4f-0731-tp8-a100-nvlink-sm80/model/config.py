"""DeepSeek-V4-Flash-0731 TP8 fixed model/runtime configuration.

This mirrors ljqinfer_tp8/model/config.py at the architecture boundary: model
geometry and execution-pool constants are explicit code, while checkpoint and
cache locations remain DSV4-specific.
"""
from pathlib import Path

TP = 8
MODEL_DIR = Path("/mnt/data/kw/models/DeepSeek-V4-Flash-0731")
CACHE_DIR = Path("/dev/shm/ljq_dsv4f_tp8")
CACHE_FORMAT = "ljq-dsv4f-tp8-v1"
ALIGNMENT = 256

# Model geometry from the validated checkpoint config.
D = DIM = 4096
N_LAYER = 43
VOCAB = 129280
HC = 4
NORM_EPS = HC_EPS = 1e-6
HC_MULT = HC
HC_SINKHORN_ITERS = 20
N_HEAD = 64
LOCAL_H = N_HEAD // TP
Q_RANK = 1024
KV_LORA = 512
ROPE_DIM = 64
CACHE_DIM = KV_WIDTH = KV_LORA  # 512; RoPE baked into last ROPE_DIM dims at write (DSV4 absorbed MLA)
HEAD_DIM = 512
ROPE_BASE = 10000.0
WINDOW = 128
O_LORA = 1024

# MoE.
N_EXPERT = 256
TOPK = 6
ROUTED_FF = EXPERT_D = 2048
SHARED_FF = 2048
ROUTED_SCALING = 1.5
N_GROUP = 8
GROUP_SIZE = N_EXPERT // N_GROUP

# Three checkpoint blocks form one DSV4 MTP/DSpark draft stack.
N_MTP = 3
MTP_LAYER = N_LAYER
Q_MAX = 8
MAX_BATCH_SIZE = 8

EOS_DEFAULT = 1
KV_PAGE_SIZE = 2048
CKV_PAGE_SIZE = 128      # 压缩池页大小
KV_ALIGN = 128           # 冷 KV 导出/导入对齐粒度 = max(RATIOS)
EXECUTION_LEN = 210 * 1024
DEFAULT_PREFILL_CHUNK_TOKENS = 12 * 1024

# --- ref-skeleton compatibility constants (DSV4 has no dense/special layers) ---
DENSE_LAYERS = ()
SPECIAL_LAYERS = frozenset()
SPECIAL_CFG = {}
LOCAL_ROUTED_FF = ROUTED_FF // TP        # 256 per-rank routed expert width
LOCAL_SHARED_FF = SHARED_FF // TP        # 256 per-rank shared expert width
LOCAL_FF = LOCAL_SHARED_FF               # no dense FFN in DSV4; placeholder for ref ABI
GROUP_TOPK = N_GROUP                     # DSV4 routing is plain top-6 (no group limit)

