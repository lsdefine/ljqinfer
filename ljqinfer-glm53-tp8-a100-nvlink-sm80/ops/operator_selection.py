"""The only operator implementation selection and build recipe table.

Edit SELECTED_OPS to choose implementations.  Both the explicit builder and the
runtime loader consume this table; no runtime fallback or implicit compilation
is allowed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SM80 = "-gencode=arch=compute_80,code=sm_80"
CUTLASS = "third_party/cutlass/include"

# Ordered by required ELF load order.  A value is an implementation id below.
SELECTED_OPS = {
    "q8": "q8_cublas",
    "q8lin": "q8_linear",
    "flash": "flash_mla_tp",
    "prefill_attn": "prefill_attn_v4_batchq2",
    "route": "decode_moe_route_q13",
    "q6": "q6_lookup",
    "decode_attn": "nova_b512_tn_rowbase_staticfrag",
    "residual_add_copy": "residual_add_copy_b2q2",
    "peer_ar": "peer_ar_os_v4",
}

# sources/cflags/cuda/include define a build; abi defines the role contract.
IMPLEMENTATIONS = {
    "q8_cublas": dict(
        sources=["q8_cublas.cpp", "q8_cublas.cu"], cflags=[],
        cuda=["-O3", "--use_fast_math", SM80], include=[],
        abi=["forward", "forward_out", "clear_weight_cache", "pin_weight_cache"],
    ),
    "q8_linear": dict(
        sources=["q8_linear.cpp", "q8_linear.cu"], cflags=[],
        cuda=["-O3", "--use_fast_math", SM80], include=[], abi=["forward"],
    ),
    "flash_mla_tp": dict(
        sources=["prefill_flash_mla_tp.cu"], cflags=[],
        cuda=["-O3", "--use_fast_math", "-lineinfo", SM80], include=[], abi=["forward"],
    ),
    "prefill_attn_v4_batchq2": dict(
        sources=["decode_attn_orch.cpp", "decode_q8_mmvq_core.cu",
                 "decode_rms_rope.cu", "prefill_flash_mla_core.cu",
                 "q8_cublas.cu", "paged_prefill_mla.cu"],
        cflags=["-O3", "-std=c++17", "-DLJQ_PPM_EMBED"],
        cuda=["-O3", "--use_fast_math", "-std=c++17", "-DLJQ_PPM_EMBED", SM80],
        include=[], abi=["forward_rank_paged_inplace_tc", "forward_rank_paged_batch_q2_k0"],
    ),
    "decode_moe_route_q13": dict(
        sources=["decode_moe_route_q13.cu"], cflags=["-O3"],
        cuda=["-O3", "--use_fast_math"], include=[],
        abi=["moe_route_t1_cuda", "decode_moe_route_probs_q13_cuda"],
    ),
    "q6_lookup": dict(
        sources=["q6_lookup.cpp", "q6_lookup.cu"], cflags=["-O3"],
        cuda=["-O3", "--use_fast_math", SM80], include=[],
        abi=["lookup", "matvec", "dequant_fp16_out", "gemm_fp16_out"],
    ),
    "nova_b512_tn_rowbase_staticfrag": dict(
        sources=["nova_decode_attn.cpp", "nova_decode_q8_mmvq_tn.cu",
                 "nova_decode_rms_rope.cu", "nova_flash_mla.cu",
                 "nova_q8_cublas.cu", "nova_b512_paged_rowbase_staticfrag.cu",
                 "paged_prefill_mla.cu"],
        cflags=["-O3", "-std=c++17", "-DLJQ_PPM_EMBED",
                "-Dpaged_prefill_mla=nova_paged_prefill_mla",
                "-Dpaged_prefill_mla_smem=nova_paged_prefill_mla_smem"],
        cuda=["-O3", "--use_fast_math", "-std=c++17", "-DLJQ_PPM_EMBED",
              "-Dpaged_prefill_mla=nova_paged_prefill_mla",
              "-Dpaged_prefill_mla_smem=nova_paged_prefill_mla_smem", SM80],
        include=[], abi=["forward_rank_paged_batch_k0",
                         "forward_rank_paged_batch_k0_projected",
                         "rms_norm_half_out"],
    ),
    "residual_add_copy_b2q2": dict(
        sources=["residual_add_copy_b2q2.cpp", "residual_add_copy_b2q2.cu"],
        cflags=["-O3"], cuda=["-O3", "--use_fast_math", SM80], include=[],
        abi=["forward", "fused_add_rmsnorm", "combine_moe_f32_to_f16"],
    ),
    "peer_ar_os_v4": dict(
        sources=["peer_ar_os.cu"], cflags=[], cuda=["-O3"], include=[],
        abi=["peer_ar_register", "peer_ar_unregister", "peer_ar_run",
             "latent_gather_register", "latent_gather_run",
             "latent_gather_unregister",
             "peer_meta_bcast_register", "peer_meta_bcast_run",
             "peer_gather_bf16_register", "peer_gather_bf16_run",
             "peer_gather_bf16_unregister",
             "peer_bcast_fp16_register", "peer_bcast_fp16_run",
             "peer_bcast_fp16_unregister",
             "peer_argmax_register", "peer_argmax_run",
             "peer_argmax_unregister"],
    ),
}


def selected_specs():
    """Return selected (role, implementation, spec) tuples; reject table drift."""
    unknown = [(role, impl) for role, impl in SELECTED_OPS.items()
               if impl not in IMPLEMENTATIONS]
    if unknown:
        raise RuntimeError(f"unknown selected operator implementations: {unknown}")
    return [(role, impl, IMPLEMENTATIONS[impl])
            for role, impl in SELECTED_OPS.items()]

def recipe_id(implementation: str, spec: dict, root: Path, environment: str) -> str:
    """Fingerprint one exact binary recipe, environment, and source snapshot."""
    h = hashlib.sha256()
    h.update(implementation.encode())
    h.update(json.dumps(spec, sort_keys=True).encode())
    h.update(environment.encode())
    for source in spec["sources"]:
        path = root / source
        if not path.is_file():
            raise FileNotFoundError(path)
        h.update(source.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]
