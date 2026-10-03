"""ljqinfer model config — all constants hard-coded for this machine/model.

No env vars, no CLI params. Change code, not environment.
"""

from model import weights as W


TP          = 8            # GPU ranks
D           = W.D          # 6144  hidden
N_LAYER     = W.N_LAYER    # 78
VOCAB       = W.VOCAB      # 154880
# Attention (MLA)
N_HEAD      = 64           # total attention heads
LOCAL_H     = N_HEAD // TP # 8    heads per rank
Q_RANK      = 2048         # q low-rank dim
KV_LORA     = 512          # compressed kv latent dim (shared, replicated)
ROPE_DIM    = 64           # decoupled rope dimension
CACHE_DIM   = KV_LORA + ROPE_DIM  # contiguous MLA cache row
ROPE_BASE   = 8e6          # rotary theta base (LLAMA_ROPE_TYPE_NORM)
HEAD_V      = 256          # per-head value/out dim  (LOCAL_H*HEAD_V = o_proj in)
# FFN / MoE
FF              = 12288        # dense FFN intermediate
LOCAL_FF        = FF // TP     # 1536 dense channels per rank
SHARED_FF       = 2048         # shared-expert intermediate
LOCAL_SHARED_FF = SHARED_FF // TP  # 256 shared-expert channels per rank
ROUTED_FF       = 2048         # routed-expert intermediate
LOCAL_ROUTED_FF = ROUTED_FF // TP  # 256 routed channels per rank
N_EXPERT        = 256          # routed experts
N_GROUP     = 1            # GLM53 checkpoint: no grouped filtering
GROUP_SIZE  = N_EXPERT // N_GROUP   # 256
GROUP_TOPK  = 1            # single group
TOPK        = 8            # routed experts kept per token (+1 shared)
ROUTED_SCALING = 2.5       # routed_scaling_factor (weight = probs*scale/sum)

DENSE_LAYERS   = W.DENSE_LAYERS     # {0,1,2}
SPECIAL_LAYERS = W.SPECIAL          # empty: uniform routed INT4
SPECIAL_CFG = {}
MTP_LAYER      = 78

EOS_DEFAULT = 154820  # <|endoftext|>; must match glm52_tokenizer.json/service.py

KV_PAGE_SIZE = 2048
EXECUTION_LEN = 210 * 1024  # fixed TP8 KV pool: 105 pages, 215,040 tokens

# Bound prefill scratch independently of the request/KV capacity.
# TP8 resident graphs + a 12K chunk exceed the measured A100 memory budget.
DEFAULT_PREFILL_CHUNK_TOKENS = 8 * 1024

SPECIAL_LOCAL = 256                 # expert intermediate / rank (gold SpecialMoeTP8.LOCAL)

Q_MAX = 8                      # fixed verify width: anchor + 7 DFlash2 drafts
