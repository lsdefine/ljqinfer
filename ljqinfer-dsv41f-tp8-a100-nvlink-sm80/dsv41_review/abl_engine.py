# -*- coding: utf-8 -*-
"""Amdahl ablation harness: patches the ENGINE's own function objects, then
runs the ORIGINAL driver via runpy. No re-implemented forward anywhere."""
import os, sys, runpy
import torch

ABL = os.environ.get('ABL', 'none')
REPO = '/mnt/data/kw/ljqinfer_dsv41f_tp8'
sys.path.insert(0, REPO)
RANK = int(os.environ.get('RANK', '0'))


def log(*a):
    if RANK == 0:
        print('[ABL]', *a, flush=True)


if ABL == 'probe':
    import inspect
    from ops.prefill import moe_workspace as MW
    from model import decode_attention as DA
    from ops.decode import v4k
    log('WorkspaceRouted methods:', [m for m in dir(MW.WorkspaceRouted) if not m.startswith('__')])
    log('moe_workspace funcs:', [n for n, o in vars(MW).items() if inspect.isfunction(o)])
    log('decode_attention names:', [n for n, o in vars(DA).items() if inspect.isfunction(o)])
    log('v4k funcs:', [n for n, o in vars(v4k).items() if inspect.isfunction(o)])
    import model.decode_build as DB
    log('decode_build names:', [n for n in vars(DB) if not n.startswith('_')])
    sys.exit(0)

# ---- ablations: each replaces ONE engine callable with a shape-correct no-op ----
if ABL == 'ar':
    import torch.distributed as dist
    _real = dist.all_reduce
    dist.all_reduce = lambda t, *a, **k: None
    log('dist.all_reduce -> no-op')

elif ABL == 'moe':
    from ops.prefill import moe_workspace as MW
    MW.WorkspaceRouted.__call__ = lambda self, x, ids, probabilities, reduce=True: torch.zeros_like(x)
    log('WorkspaceRouted.__call__ -> zeros_like(x)')

elif ABL == 'moegemm':
    from ops.prefill import moe_workspace as MW
    MW.WorkspaceRouted.gemm = lambda self, stage, a, w, sizes, out: None
    MW.WorkspaceRouted.gemm_device = lambda self, stage, a, w, out: None
    log('WorkspaceRouted.gemm/gemm_device -> no-op (out kept stale)')

elif ABL == 'attn':
    from model import decode_attention as DA
    _ra = DA.decode_attend
    DA.decode_attend = lambda **k: torch.zeros_like(k['q'])
    log('decode_attend -> zeros_like(q)')

elif ABL == 'gemv':
    from ops.decode import v4k
    _g = v4k.grouped_linear_bf16
    v4k.grouped_linear_bf16 = lambda x, w: torch.zeros(
        x.shape[0], x.shape[1], w.shape[1] if w.dim() == 3 else w.shape[0],
        dtype=x.dtype, device=x.device)
    log('v4k.grouped_linear_bf16 -> zeros')

elif ABL == 'draft':
    import model.spec_decode as SD
    def _nodraft(self, hidden, token, *, past, slot, start):
        ids = torch.zeros(self.window, dtype=torch.long, device=hidden.device)
        return SD.SpecState(token=token, hidden=hidden, draft=ids,
                            score=torch.zeros(self.window, device=hidden.device))
    SD.SpecDecoder._draft = _nodraft
    log('SpecDecode._draft -> drafter call removed')

elif ABL == 'commit':
    import model.decode_build as DB
    DB.DecodeModel.commit = lambda self, accepted, **kw: None
    log('DecodeModel.commit -> no-op')

elif ABL in ('proj', 'rms', 'rope'):
    import importlib, sys as _s
    for _m in ('model.decode_layer', 'model.decode_attention', 'model.decode_build',
               'model.dspark_build', 'ops.decode.v4k', 'ops.prefill.gemm'):
        try: importlib.import_module(_m)
        except Exception as e: log('preimport fail', _m, e)
    import ops.decode.v4k as V4K
    from ops.prefill import gemm as PG

    if ABL == 'proj':
        def _noproj(self, name, x):
            w = self.weights[name + '.weight']
            out = w.shape[0] if w.dim() == 2 else w.shape[-1]
            return torch.zeros(x.shape[:-1] + (out,), dtype=x.dtype, device=x.device)
        PG.PrefillLinear.__call__ = _noproj
        log('PrefillLinear.__call__ -> zeros')
    else:
        tgt = 'rms' if ABL == 'rms' else 'rope_'
        orig = getattr(V4K, tgt)
        repl = (lambda x, weight, eps: x) if ABL == 'rms' else (lambda x, freqs, inverse=False: x)
        setattr(V4K, tgt, repl)
        n = 0
        for _mod in list(_s.modules.values()):
            nm = getattr(_mod, '__name__', '') or ''
            if nm.startswith(('model', 'ops')) and getattr(_mod, tgt, None) is orig:
                setattr(_mod, tgt, repl); n += 1
        log(f'v4k.{tgt} -> passthrough, patched {n} extra namespaces')

elif ABL != 'none':
    raise SystemExit(f'unknown ABL={ABL}')

log(f'ABL={ABL} -> running ORIGINAL driver bench/decode_graph.py')
sys.argv = ['bench/decode_graph.py'] + sys.argv[1:]
runpy.run_path(os.path.join(REPO, 'bench/decode_graph.py'), run_name='__main__')
