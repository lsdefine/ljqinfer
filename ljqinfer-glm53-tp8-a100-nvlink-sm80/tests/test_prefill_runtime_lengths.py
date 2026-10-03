"""CPU-only guard: request-dependent scalars must not specialize kernels."""
import importlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SPEC = {'ops/moe_prefill.py': {'_dispatch': ['R'], 'dense_gu': ['R'], 'dense': ['R'], 'activate': ['R']}, 'ops/moe_common.py': {'combine': ['T']}, 'ops/moe_layer.py': {'_router_cast': ['N'], '_shared_act': ['T']}, 'ops/sparse_index_query.py': {'_score_query': ['T', 'N', 'START', 'ROWS']}, 'ops/prefill_elementwise.py': {'square_fp32': ['N', 'S'], 'rope_apply': ['N', 'S0', 'S1', 'S2'], 'pack_rope': ['N', 'L0', 'L1', 'R0', 'R1']}, 'model/glm53_block.py': {'_residual': ['N'], '_rms_apply': ['N', 'S']}}

def test_runtime_lengths():
    for path, functions in SPEC.items():
        module = importlib.import_module(path[:-3].replace('/', '.'))
        for name, args in functions.items():
            fn = getattr(module, name)
            for arg in args:
                p = next(p for p in fn.params if p.name == arg)
                assert not p.is_constexpr and p.do_not_specialize, (path, name, arg)
                assert arg in fn.do_not_specialize_on_alignment, (path, name, arg)

if __name__ == '__main__':
    test_runtime_lengths()
    print('PASS: 13 kernels use runtime request scalars')
