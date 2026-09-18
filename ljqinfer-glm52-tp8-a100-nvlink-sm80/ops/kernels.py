"""Fixed compiled-kernel binding layer for ljqinfer.

This module owns the one production extension recipe and ABI binding. It does
not import model/decode and contains no model execution order. Operator changes
belong here; model/decode only compose the fixed exported operations.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

# Compiled handles; the authoritative no-fallback selection table is above _EXT_RECIPE.
# Call sites use explicit symbols rather than dynamic kernel-version selection.
class Kernels:
    """Thin namespace over the compiled extension. Bound in Engine.load().

    Every method is a *stub contract* here: it documents in/out shapes so the
    body can be filled by delegating to the real .so entry point. Keeping the
    contract in code (not prose) is the review requirement — a caller can reason
    about shapes locally without opening the CUDA source.
    """
    so = None  # legacy single-handle slot; kept None. Use per-module handles below.
    # Per-module compiled handles, populated by Kernels.bind() in Engine.load().
    # Each corresponds to one entry of _EXT_RECIPE (see loader at bottom).
    iq = None       # iq_moe_rank
    down_cache = None  # selected routed-expert exact-LUT Q8 Down residency
    q8 = None       # ljq_q8_cublas
    q8lin = None    # ljq_q8_linear
    flash = None    # flash_mla_sm80_tp_ext
    prefill_attn = None  # selected prefill attention implementation
    decode_attn = None  # unified Nova SO for Q=1..4 and B2Q2 decode
    special_ext = None  # legacy special dequant/reference extension
    special_quant = None  # packed special MoE decode, explicit-out leaf ABI
    residual_add_copy = None  # prebuilt exact B2Q2 residual add + FP32 copy
    route = None    # decode_moe_route_q13
    q6 = None       # ljq_q6_lookup
    peer_ar = None  # peer_ar_os_v1; loaded lazily by decode graph capture

    # --- primitive matvec/matmul over raw quantized bytes ------------------
    @staticmethod
    def q8_matmul(x: torch.Tensor, w_bytes: torch.Tensor, out_dim: int) -> torch.Tensor:
        """x:[T,D] fp16 @ Q8_0 weight bytes[out_dim, D/32*34] -> [T,out_dim] fp16.

        Backing op: ljq_q8_cublas.forward(x, packed, in_features). The kernel's
        3rd arg is the *contraction* dim (= x.size(-1)); out_dim (= packed.size(0))
        is derived by the kernel and only asserted here.
        """
        y = Kernels.q8.forward(x.contiguous(), w_bytes, x.shape[-1])
        assert y.shape[-1] == out_dim, (y.shape, out_dim)
        return y

    @staticmethod
    def kquant_matmul(x: torch.Tensor, w_bytes: torch.Tensor, out_dim: int, cfg) -> torch.Tensor:
        """x:[T,K] @ Qk_K packed bytes -> [T,out_dim] (same dtype as x).

        cfg = quant type int in {3,4,5,6} (Q3_K..Q6_K). Backing path matches
        decode_special_moe.cpp::linear_q for qt>=3:
            W = dequant_k(packed, K=x.size(-1), qt)   # [M,K] fp32
            y = x_fp32 @ W.t()
        IQ paths (qt 1/2) are only available inside the fused special MoE kernel
        (mmq_iq*_tile, not pybind-exported), so this leaf is Qk_K-only.
        """
        qt = int(cfg)
        assert 3 <= qt <= 6, qt
        k_in = x.shape[-1]
        W = Kernels.special_ext.dequant_k(w_bytes.contiguous(), k_in, qt)
        assert W.shape[0] == out_dim, (W.shape, out_dim)
        y = torch.matmul(x.to(torch.float32), W.t())
        return y.to(dtype=x.dtype)



K = Kernels

# Runtime resolution is intentionally boring: the explicit builder writes one
# immutable lock, and production verifies and loads exactly those artifacts.
# Runtime never imports torch.utils.cpp_extension and never compiles or falls back.
from ops.operator_selection import IMPLEMENTATIONS, SELECTED_OPS, recipe_id

_ROOT = Path(__file__).resolve().parent
_LOCK = _ROOT / "selected_ops.lock.json"
_ENVIRONMENT_ID = (f"python={sys.version_info[:3]};torch={torch.__version__};"
                   f"cuda={torch.version.cuda}")
_ROLE_TO_ATTR = {
    "iq": "iq",
    "down_cache": "down_cache",
    "q8": "q8",
    "q8lin": "q8lin",
    "flash": "flash",
    "prefill_attn": "prefill_attn",
    "special_ext": "special_ext",
    "special_quant": "special_quant",
    "route": "route",
    "q6": "q6",
    "decode_attn": "decode_attn",
    "residual_add_copy": "residual_add_copy",
    "peer_ar": "peer_ar",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_artifact(role: str, record: dict):
    path = (_ROOT / record["artifact"]).resolve()
    if _ROOT not in path.parents or path.suffix != ".so":
        raise RuntimeError(f"invalid {role} artifact path: {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"selected {role} artifact is absent: {path}; "
            "run `python -m ops.build_selected`")
    actual = _sha256(path)
    if actual != record["sha256"]:
        raise RuntimeError(
            f"selected {role} artifact hash mismatch: expected={record['sha256']} "
            f"actual={actual} path={path}")
    name = record["module_name"]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create extension loader for selected {role}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [symbol for symbol in record["abi"] if not hasattr(module, symbol)]
    if missing:
        raise RuntimeError(f"selected {role} ABI missing {missing}: {path}")
    print(f"[ops] role={role} implementation={record['implementation']} "
          f"sha256={actual} artifact={path}", flush=True)
    return module


def load_kernels():
    """Verify and load the exact selected operator set; never build at runtime."""
    if K.prefill_attn is not None:
        return K
    if not _LOCK.is_file():
        raise FileNotFoundError(
            f"operator lock is absent: {_LOCK}; run `python -m ops.build_selected`")
    lock = json.loads(_LOCK.read_text())
    if lock.get("format") != 1:
        raise RuntimeError(f"unsupported operator lock format: {lock.get('format')}")
    if lock.get("selection") != SELECTED_OPS:
        raise RuntimeError(
            "operator selection changed after build; run `python -m ops.build_selected`")
    records = lock.get("operators", {})
    if set(records) != set(_ROLE_TO_ATTR):
        raise RuntimeError(
            f"operator lock roles mismatch: expected={list(_ROLE_TO_ATTR)} "
            f"actual={list(records)}")
    # Load and verify every module before publishing any handle on K.  A late
    # failure must not make a retry observe a half-initialized operator set.
    loaded = {}
    # Dict insertion order is the declared ELF load order in SELECTED_OPS.
    for role in SELECTED_OPS:
        record = records[role]
        implementation = SELECTED_OPS[role]
        if record.get("implementation") != implementation:
            raise RuntimeError(f"selected implementation mismatch for {role}")
        expected_recipe = recipe_id(
            implementation, IMPLEMENTATIONS[implementation], _ROOT, _ENVIRONMENT_ID)
        if record.get("recipe_id") != expected_recipe:
            raise RuntimeError(
                f"selected {role} recipe/source/environment changed after build; "
                "run `python -m ops.build_selected`")
        # NOTE(2026-08-09): the historical RTLD_GLOBAL promotion of the
        # prefill-attention provider caused ELF symbol interposition: both the
        # prefill and Nova decode modules export identically-named q8_mmvq/rms
        # kernels, so Nova's calls could bind to the stale prefill copies
        # (cold-start bistable kv_a zero bug). The Nova decode module resolves
        # all its imports from libtorch alone (verified RTLD_NOW|RTLD_LOCAL),
        # so no promotion is required.
        loaded[_ROLE_TO_ATTR[role]] = _load_artifact(role, record)
    for attr, module in loaded.items():
        setattr(K, attr, module)
    return K


def _require_loaded(attr: str):
    module = getattr(K, attr)
    if module is None:
        raise RuntimeError("operators are not loaded; Engine.load must call load_kernels first")
    return module


def get_decode_attn():
    return _require_loaded("decode_attn")


def get_residual_add_copy():
    return _require_loaded("residual_add_copy")


def get_peer_ar():
    return _require_loaded("peer_ar")
