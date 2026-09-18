"""Gate 0: functional purity of ops leaves (seconds). Each op: (a) inputs bit-unchanged,
(b) canary buffers around inputs untouched, (c) bit-identical output after heap perturbation,
(d) no pointer-keyed cache (fp8 dequant+gemm), (e) no static mutable state in ops/*.cu."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import time, torch
import ops
from ops import _mod

t0 = time.time()
dev = "cuda"
torch.manual_seed(0)
FAILS = []


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def snap(ts):
    return [t.clone() for t in ts if torch.is_tensor(t)]


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8) if a.dtype.is_floating_point else a, b.view(torch.uint8) if b.dtype.is_floating_point else b)


def outs_list(o):
    return list(o) if isinstance(o, (list, tuple)) else [o]


def check(name, fn, args, n_perturb=3):
    args = list(args)
    ts = [a for a in args if torch.is_tensor(a)]
    # canaries allocated right after inputs
    canaries = [torch.full((1 << 20,), 0x5A, dtype=torch.uint8, device=dev) for _ in range(2)]
    before = snap(ts)
    torch.cuda.synchronize()
    ref = [o.clone() for o in outs_list(fn(*args))]
    torch.cuda.synchronize()
    ok = True
    for i, (a, b) in enumerate(zip(ts, before)):
        if not same(a, b):
            FAILS.append(f"{name}: input#{i} MUTATED"); ok = False
    for i, c in enumerate(canaries):
        if not bool((c == 0x5A).all()):
            FAILS.append(f"{name}: canary#{i} CORRUPTED"); ok = False
    for k in range(n_perturb):
        junk = [torch.empty(int(torch.randint(1 << 16, 1 << 24, (1,))), dtype=torch.uint8, device=dev) for _ in range(5)]
        junk[0].fill_(0xFF)
        out = outs_list(fn(*args))
        torch.cuda.synchronize()
        for j, (o, r) in enumerate(zip(out, ref)):
            if not same(o, r):
                d = (o.float() - r.float()).abs().max().item() if o.dtype != torch.int32 else int((o != r).sum())
                FAILS.append(f"{name}: out#{j} NONDETERMINISTIC after heap perturb {k} (maxdiff={d})"); ok = False
                break
        del junk
    log(f"{name:24s} {'PASS' if ok else 'FAIL'}")
    return ref


# ---------------- fp8 linear leaves
m = _mod()
N, K, M = 1024, 2048, 87
w = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
s = torch.full((N // 128, K // 128), 1.0, device=dev).to(torch.float8_e8m0fnu)
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
wd = check("dequant_fp8_bf16", m.dequant_fp8_bf16, [w, s])[0]
check("bf16_gemm", m.bf16_gemm, [x, wd])
# fp8_linear leaf vs kernels_torch oracle (act_quant ue8m0 + fp8_gemm fp32-accum)
from model.kernels_torch import act_quant as _aq, fp8_gemm as _fg
yl = check("fp8_linear", m.fp8_linear, [x, wd])[0]
_xq, _xs = _aq(x, 128, "ue8m0", torch.float8_e8m0fnu)
_yr = _fg(_xq, _xs, w, s, torch.float8_e8m0fnu)
_err = (yl.float() - _yr.float()).abs().max().item()
if _err > 0.05:
    FAILS.append(f"fp8_linear: oracle mismatch max|d|={_err}")
log(f"{'fp8_linear oracle':24s} {'PASS' if _err <= 0.05 else 'FAIL'} (max|d|={_err:.4g})")

# rms_norm leaf vs arch.RMSNorm oracle (fp32 math, bf16 round)
_wn = (torch.randn(x.shape[-1], device=dev) * 0.1 + 1.0).float()
yn = check("rms_norm", m.rms_norm, [x, _wn, 1e-6])[0]
_xf = x.float()
_yr = (_wn * (_xf * torch.rsqrt(_xf.square().mean(-1, keepdim=True) + 1e-6))).to(torch.bfloat16)
_err = (yn.float() - _yr.float()).abs().max().item()
if _err > 0.02:
    FAILS.append(f"rms_norm: oracle mismatch max|d|={_err}")
log(f"{'rms_norm oracle':24s} {'PASS' if _err <= 0.02 else 'FAIL'} (max|d|={_err:.4g})")

# rope_inplace: documented in-place op -> purity checked as (copy-in, compare) not via check()
_S, _Hh, _rd = 16, 4, 64
_fr = torch.polar(torch.ones(_S, _rd // 2, device=dev), torch.randn(_S, _rd // 2, device=dev))
for _inv in (False, True):
    _q4 = torch.randn(2, _S, _Hh, 192, device=dev, dtype=torch.bfloat16)
    _ref4 = _q4.clone()
    _qf = _fr.clone()
    _frs = torch.polar(torch.ones(_S * 2, _rd // 2, device=dev), torch.randn(_S * 2, _rd // 2, device=dev))[::2]
    _frs.copy_(_fr)  # strided-row view holding the same values
    m.rope_inplace(_q4[..., -_rd:], _frs, _inv)
    _yr = torch.view_as_complex(_ref4[..., -_rd:].float().unflatten(-1, (-1, 2)))
    _yr = torch.view_as_real(_yr * (_fr.conj() if _inv else _fr).view(1, _S, 1, -1)).flatten(-2).to(torch.bfloat16)
    _err = (_q4[..., -_rd:].float() - _yr.float()).abs().max().item()
    _un = (_q4[..., :-_rd] != _ref4[..., :-_rd]).any().item()
    _fm = not torch.equal(_fr, _qf)
    ok = _err <= 0.02 and not _un and not _fm
    if not ok:
        FAILS.append(f"rope_inplace(inv={_inv}): max|d|={_err} untouched-mutated={_un} freqs-mutated={_fm}")
    log(f"{'rope_inplace inv=' + str(_inv)[0]:24s} {'PASS' if ok else 'FAIL'} (max|d|={_err:.4g})")

# wo_a_grouped leaf vs einsum oracle (fp32 math, bf16 round)
_G, _R, _D = 4, 32, 64
_og = torch.randn(8, _G, _D, device=dev).to(torch.bfloat16)
_wg = (torch.randn(_G, _R, _D, device=dev) * 0.1).to(torch.bfloat16)
_yg = check("wo_a_grouped", m.wo_a_grouped, [_og, _wg])[0]
_ygr = torch.einsum("tgd,grd->tgr", _og.float(), _wg.float()).to(torch.bfloat16)
_err = (_yg.float() - _ygr.float()).abs().max().item()
if _err > 0.02:
    FAILS.append(f"wo_a_grouped: oracle mismatch max|d|={_err}")
log(f"{'wo_a_grouped oracle':24s} {'PASS' if _err <= 0.02 else 'FAIL'} (max|d|={_err:.4g})")

# hc_fused_pre leaf vs torch oracle (hc_pre fp32 + rmsnorm). mixes gemm is bf16 in-kernel -> tolerance.
_HC, _Dh, _Th = 4, 64, 8
_xh = torch.randn(_Th, _HC, _Dh, device=dev).to(torch.bfloat16)
_fn = (torch.randn((2 + _HC) * _HC, _HC * _Dh, device=dev) * 0.05).float()
_hs = torch.tensor([1.0, 0.7, 1.3], device=dev)
_hb = (torch.randn((2 + _HC) * _HC, device=dev) * 0.3).float()
_nw = (1 + 0.1 * torch.randn(_Dh, device=dev)).float()
_eps = 1e-6
_post, _comb, _xn = check("hc_fused_pre", m.hc_fused_pre, [_xh, _fn, _hs, _hb, _nw, _eps, False])[:3]
def _hc_oracle(x4, fn, hs, hb, nw, eps, iters=20):
    hc = x4.shape[1]
    xf = x4.flatten(1).float()
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    z = torch.nn.functional.linear(xf, fn) * rs
    pre = torch.sigmoid(z[:, :hc] * hs[0] + hb[:hc]) + eps
    post = 2 * torch.sigmoid(z[:, hc:2 * hc] * hs[1] + hb[hc:2 * hc])
    comb = (z[:, 2 * hc:] * hs[2] + hb[2 * hc:]).reshape(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    y = (pre.unsqueeze(-1) * x4.float()).sum(1).to(torch.bfloat16).float()
    xn = (nw * (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps))).to(torch.bfloat16)
    return post, comb, xn
_po, _co, _xo = _hc_oracle(_xh, _fn, _hs, _hb, _nw, _eps)
_e1 = (_post - _po).abs().max().item(); _e2 = (_comb - _co).abs().max().item()
_e3 = (_xn.float() - _xo.float()).abs().max().item()
_ok = _e1 <= 0.02 and _e2 <= 0.02 and _e3 <= 0.06
if not _ok:
    FAILS.append(f"hc_fused_pre: oracle mismatch post={_e1} comb={_e2} xn={_e3}")
log(f"{'hc_fused_pre oracle':24s} {'PASS' if _ok else 'FAIL'} (post={_e1:.3g} comb={_e2:.3g} xn={_e3:.3g})")

# hc_post_fused_bf16 leaf vs torch oracle
_br = torch.randn(_Th, _Dh, device=dev).to(torch.bfloat16)
_yp = check("hc_post_fused_bf16", m.hc_post_fused_bf16, [_br, _xh, _po.contiguous(), _co.contiguous(), False])[0]
# raw kernel contracts comb's LAST index: y[j] = post[j]*br + sum_i comb[j,i]*res[i]
_ypo = (_po.unsqueeze(-1) * _br.float().unsqueeze(-2) + (_co.unsqueeze(-1) * _xh.float().unsqueeze(-3)).sum(2)).to(torch.bfloat16)
# ops.hc_post wrapper follows arch semantics (comb^T): y[j] = post[j]*br + sum_i comb[i,j]*res[i]
from ops import hc_post as _hc_post_w
_ypw = _hc_post_w(_br, _xh, _po, _co)
_ypwo = (_po.unsqueeze(-1) * _br.float().unsqueeze(-2) + (_co.unsqueeze(-1) * _xh.float().unsqueeze(-2)).sum(1)).to(torch.bfloat16)
_e5 = (_ypw.float() - _ypwo.float()).abs().max().item()
if _e5 > 0.05:
    FAILS.append(f"ops.hc_post wrapper: oracle mismatch max|d|={_e5}")
log(f"{'hc_post wrapper oracle':24s} {'PASS' if _e5 <= 0.05 else 'FAIL'} (max|d|={_e5:.3g})")
_e4 = (_yp.float() - _ypo.float()).abs().max().item()
if _e4 > 0.05:
    FAILS.append(f"hc_post_fused_bf16: oracle mismatch max|d|={_e4}")
log(f"{'hc_post oracle':24s} {'PASS' if _e4 <= 0.05 else 'FAIL'} (max|d|={_e4:.3g})")

# (d) no pointer-keyed state anywhere: free w, allocate w2 at (likely) the same ptr,
# result must equal a from-scratch dequant of w2.
ptr = w.data_ptr(); del w, wd; torch.cuda.synchronize()
w2 = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
alias = w2.data_ptr() == ptr
y2 = m.bf16_gemm(x, m.dequant_fp8_bf16(w2, s))
y2_ref = torch.matmul(x.float(), w2.float().t()).to(torch.bfloat16)
ok = (y2.float() - y2_ref.float()).abs().max().item() < 0.5
if not ok:
    FAILS.append(f"dequant_fp8_bf16/bf16_gemm: wrong after pointer reuse (alias={alias})")
log(f"{'fp8 ptr-alias':24s} {'PASS' if ok else 'FAIL'} (ptr reused={alias})")

# (e) source-level gate: no process-global mutable state in compiled ops
import re, pathlib
_pat = re.compile(r"^\s*(static\s+)?(std::(mutex|map|unordered_map|vector)|thread_local)\b.*;\s*(//.*)?$")
for cu in sorted(pathlib.Path(ops.__file__).parent.glob("*.cu")):
    for ln, line in enumerate(cu.read_text().splitlines(), 1):
        if line.lstrip().startswith("static ") and _pat.match(line) and "(" not in line.split("//")[0]:
            FAILS.append(f"global state: {cu.name}:{ln}: {line.strip()}")
log(f"{'no-global-state grep':24s} {'FAIL' if any(f.startswith('global state') for f in FAILS) else 'PASS'}")

# ---------------- indexer leaves
b, sq, h, n, ratio, off = 1, 87, 64, 87, 1, 0
score = torch.randn(b, sq, h, n, device=dev, dtype=torch.bfloat16)
wts = torch.rand(b, sq, h, device=dev, dtype=torch.bfloat16)
red = check("index_score_reduce", m.index_score_reduce, [score, wts, ratio, 0])[0]
Kt = min(512, n)
check("topk_select_post", m.topk_select_post, [red.float(), Kt, ratio, off, 0])
# tie-heavy score (many -inf / equal values): ordering must still be deterministic
tie = torch.zeros(b, sq, n, device=dev); tie[..., :10] = 1.0
check("topk_select_post(ties)", m.topk_select_post, [tie, Kt, ratio, off, 0])

# ---------------- sparse attention leaf
hq = 8
q = torch.randn(b, sq, hq, 512, device=dev, dtype=torch.bfloat16)
pool = torch.randn(b, n, 512, device=dev, dtype=torch.bfloat16)
table = torch.arange(b, device=dev, dtype=torch.int64).view(b, 1)
sink = torch.zeros(hq, device=dev)
idxs = torch.arange(n, device=dev).repeat(b, sq, 1)
idxs = torch.where(idxs > torch.arange(sq, device=dev).view(1, sq, 1), -1, idxs).contiguous()
check("sparse_attn_paged", m.sparse_attn_paged, [q, pool, table, pool, table, sink, idxs, n, 0.05])
# same set of ids, shuffled order -> must be equal (kernel accumulation order sensitivity)
perm = torch.argsort(torch.rand(idxs.shape, device=dev), dim=-1)
idxs_shuf = torch.gather(idxs, -1, perm).contiguous()
o1 = m.sparse_attn_paged(q, pool, table, pool, table, sink, idxs, n, 0.05)
o2 = m.sparse_attn_paged(q, pool, table, pool, table, sink, idxs_shuf, n, 0.05)
d = (o1.float() - o2.float()).abs().max().item()
log(f"{'sparse_attn idx-order':24s} maxdiff={d:.3e} {'(order-sensitive)' if d > 0 else 'PASS'}")

print("=" * 60)
for f in FAILS:
    print("FAIL", f)
log("GATE0", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
