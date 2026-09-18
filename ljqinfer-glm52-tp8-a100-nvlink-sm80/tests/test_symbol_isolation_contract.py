"""Symbol-isolation contract for ops kernel artifacts.

Guards against the two loader pitfalls that caused the 2026-08-08/09 bugs:
1. RTLD_GLOBAL promotion anywhere in ops/ python code (symbol interposition:
   one .so's stale kernels hijack another's -> bistable cold-start corruption).
2. A selected artifact that cannot load standalone (RTLD_NOW|RTLD_LOCAL),
   i.e. a real cross-.so dependency that would tempt someone to re-add
   RTLD_GLOBAL as a "fix".
3. NEW duplicate exported T symbols across selected artifacts beyond the
   recorded baseline (tests/symbol_overlap_baseline.json). If this fails:
   make the kernel `static`, rename it, or link with -Wl,-Bsymbolic-functions.
   NEVER fix it with RTLD_GLOBAL.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import itertools

REPO = pathlib.Path(__file__).resolve().parent.parent
OPS = REPO / "ops"
BASELINE = pathlib.Path(__file__).resolve().parent / "symbol_overlap_baseline.json"


def _selected_artifacts():
    lock = json.loads((OPS / "selected_ops.lock.json").read_text())
    arts = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "artifact" and isinstance(v, str):
                    arts.append(OPS / v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(lock)
    return sorted(set(a for a in arts if a.exists()))


def test_no_rtld_global_in_ops_code():
    bad = []
    for py in OPS.rglob("*.py"):
        for i, line in enumerate(py.read_text(errors="replace").splitlines(), 1):
            if "RTLD_GLOBAL" in line and not line.lstrip().startswith("#"):
                bad.append(f"{py.relative_to(REPO)}:{i}: {line.strip()}")
    assert not bad, (
        "RTLD_GLOBAL is banned in ops/ (symbol interposition hazard, "
        "see 2026-08-09 cold-start bistable kv_a bug):\n" + "\n".join(bad))


def test_selected_artifacts_load_standalone():
    failures = []
    for so in _selected_artifacts():
        code = (
            "import ctypes, os, sys; import torch; "
            f"ctypes.CDLL({str(so)!r}, mode=os.RTLD_NOW | os.RTLD_LOCAL)"
        )
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            failures.append(f"{so.name}: {r.stderr.strip()[-200:]}")
    assert not failures, (
        "selected artifacts must load with RTLD_NOW|RTLD_LOCAL standalone "
        "(fix the real dependency; do NOT use RTLD_GLOBAL):\n"
        + "\n".join(failures))


def _exports(so):
    r = subprocess.run(["nm", "-D", str(so)], capture_output=True, text=True)
    out = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in ("T", "W") and not parts[2].startswith(("_ZN", "_ZSt", "__")):
            out.add(parts[2])
        elif len(parts) == 3 and parts[1] == "T":
            out.add(parts[2])
    return out


def test_no_new_cross_so_symbol_overlap():
    sos = _selected_artifacts()
    exports = {so.name: _exports(so) for so in sos}
    overlaps = {}
    for a, b in itertools.combinations(sorted(exports), 2):
        inter = exports[a] & exports[b]
        if inter:
            overlaps[f"{a}||{b}"] = sorted(inter)
    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(
            sorted(set(s for v in overlaps.values() for s in v)), indent=1))
        return
    baseline = set(json.loads(BASELINE.read_text()))
    new = {k: [s for s in v if s not in baseline]
           for k, v in overlaps.items()}
    new = {k: v for k, v in new.items() if v}
    assert not new, (
        "NEW duplicate exported symbols across selected .so files. "
        "Under RTLD_LOCAL this is latent, but any global promotion or "
        "future loader change detonates it. Make kernels static/renamed "
        "or link with -Wl,-Bsymbolic-functions:\n"
        + json.dumps(new, indent=1)[:2000])
