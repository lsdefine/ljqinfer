"""ops build: compile broken-lineage kernels via torch cpp_extension."""
from pathlib import Path
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
os.environ.setdefault("MAX_JOBS", "8")
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def _load(name, sources, cuda_flags, verbose=False):
    (ROOT / ".build").mkdir(exist_ok=True)
    return load(
        name=name,
        sources=[str(ROOT / s) for s in sources],
        extra_cuda_cflags=cuda_flags,
        build_directory=str(ROOT / ".build"),
        verbose=verbose,
    )


# Vendored in-tree (ops/third_party/cutlass) so this folder is self-contained;
# it used to point at a sibling checkout, which broke on a plain folder copy.
CUTLASS_INC = str(ROOT / "third_party" / "cutlass" / "include")
if not (Path(CUTLASS_INC) / "cutlass" / "cutlass.h").is_file():
    raise RuntimeError(
        "CUTLASS headers missing at %s -- ops/third_party/cutlass is part of this "
        "repo; restore it before building." % CUTLASS_INC)


def load_wgemm(verbose=False):
    # exact broken recipe: 6 sources, NO_PYBIND for non-main files, cutlass headers
    return _load(
        "dsv4_wgemm",
        ["dsv4_wgemm.cu", "prefill_moe_cutlass_gemm.cu", "sparse_attn_paged.cu",
         "paged_io.cu", "compressor_tail.cu", "index_score.cu",
         "peer_ar_ipc.cu"],
        ["-O3", "--use_fast_math", "-lineinfo", "-DDSV4_NO_PYBIND", "-I" + CUTLASS_INC],
        verbose,
    )
