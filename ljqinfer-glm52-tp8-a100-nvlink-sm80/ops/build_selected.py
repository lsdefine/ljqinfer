"""Explicitly build and atomically publish exactly SELECTED_OPS.

Runtime never calls this module.  Run: python -m ops.build_selected
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MAX_JOBS", "8")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

import torch
from torch.utils.cpp_extension import load

from ops.operator_selection import SELECTED_OPS, recipe_id, selected_specs

ROOT = Path(__file__).resolve().parent
BUILD_ROOT = ROOT / ".operator_build"
ARTIFACT_ROOT = ROOT / "artifacts"
LOCK = ROOT / "selected_ops.lock.json"
ENVIRONMENT_ID = (f"python={sys.version_info[:3]};torch={torch.__version__};"
                  f"cuda={torch.version.cuda}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    BUILD_ROOT.mkdir(exist_ok=True)
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    resolved = {}
    for role, impl, spec in selected_specs():
        build_id = recipe_id(impl, spec, ROOT, ENVIRONMENT_ID)
        module_name = f"ljq_{impl}_{build_id}"
        build_dir = BUILD_ROOT / module_name
        build_dir.mkdir(exist_ok=True)
        print(f"[build] role={role} implementation={impl} id={build_id}", flush=True)
        module = load(
            name=module_name,
            sources=[str(ROOT / s) for s in spec["sources"]],
            build_directory=str(build_dir),
            extra_cflags=spec["cflags"] or None,
            extra_cuda_cflags=spec["cuda"] or None,
            extra_include_paths=[str(ROOT / p) for p in spec["include"]] or None,
            verbose=True,
        )
        missing = [symbol for symbol in spec["abi"] if not hasattr(module, symbol)]
        if missing:
            raise RuntimeError(f"{role}/{impl} ABI missing: {missing}")
        built = Path(module.__file__).resolve()
        digest = _sha256(built)
        artifact = ARTIFACT_ROOT / f"{module_name}-{digest[:16]}.so"
        if not artifact.exists() or _sha256(artifact) != digest:
            tmp = artifact.with_suffix(".so.tmp")
            shutil.copy2(built, tmp)
            os.replace(tmp, artifact)
        resolved[role] = {
            "implementation": impl,
            "module_name": module_name,
            "recipe_id": build_id,
            "artifact": str(artifact.relative_to(ROOT)),
            "sha256": digest,
            "abi": spec["abi"],
        }
        print(f"[built] role={role} artifact={artifact} sha256={digest}", flush=True)

    lock = {
        "format": 1,
        "selection": SELECTED_OPS,
        "environment": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "operators": resolved,
    }
    tmp = LOCK.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lock, indent=2) + "\n")
    os.replace(tmp, LOCK)
    print(f"[published] {LOCK}", flush=True)


if __name__ == "__main__":
    main()
