"""CANN 9 AOT GDN prepare kernels for the production 64-token prefill path."""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).with_suffix("")
_LIB = _ROOT / "lib"
_BIN = _ROOT / "bin"
_EXPECTED_SHA256 = {
    "lib/gdn_kkt.cpython-39-aarch64-linux-gnu.so": "f6173fe0d53148d1170caac1d8910c0e25217c644598afb81f0511451a9c88d5",
    "lib/gdn_solve16.cpython-39-aarch64-linux-gnu.so": "500eab88669f33bc9de7ade2215df552dc817ae985538b80b4d61f1d5c7e12b6",
    "lib/gdn_merge64.cpython-39-aarch64-linux-gnu.so": "d258b5db8f9bb1d677a08c09b30c8a3f3d1769c4074908686683d7cee438d1d9",
    "lib/gdn_recompute.cpython-39-aarch64-linux-gnu.so": "dec592550c8814f5a8b21c9120e79fb9ae4d819d9718dfe8d61b3db46c1be887",
    "lib/gdn_npu_utils.cpython-39-aarch64-linux-gnu.so": "753c7bdcbf4c0850abff9737da9b6fac2bb21add6b2d7fcb86376d8477246f85",
    "bin/00_chunk_scaled_dot_kkt_fwd_kernel_mix.npubin": "290923ab9e946dc1842af834ebd75a52d1f1b2bdc9282c47b42b085863007815",
    "bin/01_solve_tril_16x16_kernel_aiv.npubin": "82084c1a587ec8b66d973a899733b358833e5fabd568a7e977508f81d100c32d",
    "bin/02_merge_16x16_to_64x64_inverse_kernel_mix.npubin": "98430cb341d5570c5ee53a120c7e43ad91bb1ef94cf18f0752f527607564a255",
    "bin/03_recompute_w_u_fwd_kernel_mix.npubin": "088e142dc95ea7f465c35e9ef3340bbd47df20a11b850a81328ef22970d4b4e3",
}
_ASSETS = {
    "kkt": (
        "chunk_scaled_dot_kkt_fwd_kernel", "mix",
        "00_chunk_scaled_dot_kkt_fwd_kernel_mix.npubin",
        {"kernel_name": "chunk_scaled_dot_kkt_fwd_kernel", "tensor_kinds": [0, 0, 0, 1, 0, 0]},
    ),
    "solve16": (
        "solve_tril_16x16_kernel", "aiv",
        "01_solve_tril_16x16_kernel_aiv.npubin",
        {"kernel_name": "solve_tril_16x16_kernel", "tensor_kinds": [0, 1, 0, 0]},
    ),
    "merge64": (
        "merge_16x16_to_64x64_inverse_kernel", "mix",
        "02_merge_16x16_to_64x64_inverse_kernel_mix.npubin",
        {"kernel_name": "merge_16x16_to_64x64_inverse_kernel", "tensor_kinds": [0, 0, 1, 0, 0]},
    ),
    "recompute": (
        "recompute_w_u_fwd_kernel", "mix",
        "03_recompute_w_u_fwd_kernel_mix.npubin",
        {"kernel_name": "recompute_w_u_fwd_kernel", "tensor_kinds": [0, 0, 0, 1, 1, 0, 0, 0, 0]},
    ),
}
_MODULES = None
_KERNELS = {}
_INDEX = {}


def _verify_assets() -> None:
    for relative, expected in _EXPECTED_SHA256.items():
        path = _ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing GDN AOT asset: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"GDN AOT asset hash mismatch: {path} {actual} != {expected}")


def _modules():
    global _MODULES
    if _MODULES is None:
        _verify_assets()
        sys.path.insert(0, str(_LIB))
        try:
            names = (
                "gdn_kkt", "gdn_solve16", "gdn_merge64",
                "gdn_recompute", "gdn_npu_utils",
            )
            loaded = tuple(importlib.import_module(name) for name in names)
        finally:
            if sys.path and sys.path[0] == str(_LIB):
                sys.path.pop(0)
        for module in loaded:
            if Path(module.__file__).resolve().parent != _LIB.resolve():
                raise RuntimeError(f"stale GDN AOT module loaded: {module.__file__}")
        _MODULES = loaded
    return _MODULES


def _device_index(device) -> int:
    index = device.index
    if index is None:
        index = int(torch.npu.current_device())
    return int(index)


def _kernels(device):
    index = _device_index(device)
    cached = _KERNELS.get(index)
    if cached is not None:
        return cached
    kkt, solve16, merge64, recompute, utils = _modules()
    functions = []
    metadata = []
    for key in ("kkt", "solve16", "merge64", "recompute"):
        name, mode, filename, meta = _ASSETS[key]
        module, function, _, _ = utils.load_kernel_binary(
            name, (_BIN / filename).read_bytes(), 1, index, mode)
        if not module or not function:
            raise RuntimeError(
                f"failed to register GDN AOT kernel {key} on npu:{index}")
        functions.append(function)
        metadata.append(meta)
    cached = (kkt, solve16, merge64, recompute,
              tuple(functions), tuple(metadata))
    _KERNELS[index] = cached
    return cached


def _indices(device, tokens: int):
    key = (_device_index(device), int(tokens))
    cached = _INDEX.get(key)
    if cached is not None:
        return cached
    chunks = tokens // 64
    large_chunks = (tokens + 1215) // 1216
    cu = torch.tensor([0, tokens], device=device, dtype=torch.int64)
    small = torch.stack((
        torch.zeros(chunks, device=device, dtype=torch.int64),
        torch.arange(chunks, device=device, dtype=torch.int64),
    ), 1)
    large = torch.stack((
        torch.zeros(large_chunks, device=device, dtype=torch.int64),
        torch.arange(large_chunks, device=device, dtype=torch.int64),
    ), 1)
    cached = (cu, small, large)
    _INDEX[key] = cached
    return cached


def prepare_wy(kg, vc, bc, gc):
    """Return fp32 ``u``/``w`` in [B,H,T,D] for the fixed production shape."""
    if kg.device.type != "npu" or any(
            tensor.device != kg.device for tensor in (vc, bc, gc)):
        raise ValueError("GDN AOT requires all inputs on one NPU")
    if kg.ndim != 5 or vc.ndim != 5 or bc.ndim != 4 or gc.shape != bc.shape:
        raise ValueError("unexpected GDN AOT tensor ranks")
    if any(tensor.dtype != torch.float32 for tensor in (kg, vc, bc, gc)):
        raise ValueError("GDN AOT requires fp32 kg/vc/bc/gc inputs")
    batch, heads, chunks, block, key_dim = kg.shape
    tokens = chunks * block
    if (batch, heads, block, key_dim, vc.shape[-1]) != (1, 12, 64, 128, 128):
        raise ValueError(
            f"unsupported GDN AOT shape kg={tuple(kg.shape)} vc={tuple(vc.shape)}")
    if vc.shape[:4] != kg.shape[:4] or bc.shape != (batch, heads, chunks, block):
        raise ValueError("inconsistent GDN AOT shapes")
    if not 0 < tokens <= 12288:
        raise ValueError(f"unsupported GDN AOT token count {tokens}")

    k = kg.permute(0, 2, 3, 1, 4).reshape(
        batch, tokens, heads, key_dim).contiguous()
    v = vc.permute(0, 2, 3, 1, 4).reshape(
        batch, tokens, heads, vc.shape[-1]).contiguous()
    beta = bc.permute(0, 2, 3, 1).reshape(
        batch, tokens, heads).contiguous()
    gate = gc.permute(0, 2, 3, 1).reshape(
        batch, tokens, heads).contiguous()
    beta_hbt = beta.permute(2, 0, 1).contiguous()
    gate_hbt = gate.permute(2, 0, 1).contiguous()
    cu, chunk_indices, large_indices = _indices(kg.device, tokens)
    stream = torch.npu.current_stream(kg.device).npu_stream

    kkt, solve16, merge64, recompute, functions, metadata = _kernels(kg.device)
    fk, fs, fm, fr = functions
    mk, ms, mm, mr = metadata
    nt = chunks
    nt_large = large_indices.shape[0]
    matrix = torch.empty(
        batch, tokens, heads, 64, device=kg.device, dtype=torch.float32)
    diagonal = torch.empty(
        batch, tokens, heads, 16, device=kg.device, dtype=torch.float32)
    inverse = torch.empty_like(matrix)
    w = torch.empty(
        batch, tokens, heads, key_dim, device=kg.device, dtype=torch.float32)
    u = torch.empty_like(v)

    kkt.launch(
        20, 1, 1, stream, fk, mk, None, None, None,
        k, beta_hbt, gate_hbt, matrix, cu, chunk_indices,
        tokens, batch, batch * heads, nt * batch * heads, 20)
    solve16.launch(
        nt_large, batch * heads, 1, stream, fs, ms, None, None, None,
        matrix, diagonal, cu, large_indices, tokens, heads)
    merge64.launch(
        nt, batch * heads, 1, stream, fm, mm, None, None, None,
        matrix, diagonal, inverse, cu, chunk_indices, tokens, heads)
    recompute.launch(
        nt, batch, 1, stream, fr, mr, None, None, None,
        k, v, beta.transpose(1, 2).contiguous(), w, u, inverse,
        gate.transpose(1, 2).contiguous(), cu, chunk_indices,
        tokens, heads, heads, key_dim, v.shape[-1])
    return (
        u.permute(0, 2, 1, 3).contiguous(),
        w.permute(0, 2, 1, 3).contiguous(),
    )
