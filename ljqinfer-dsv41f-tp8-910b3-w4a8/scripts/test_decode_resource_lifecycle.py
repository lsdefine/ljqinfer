#!/usr/bin/env python3
"""Standalone CPU-only lifecycle tests; no torch/CANN install or NPU required.

Run: python -B scripts/test_decode_resource_lifecycle.py [report.json]
The real source modules are compiled/executed with stubbed imports. All native
library loads and all NPU access are forbidden, not merely avoided by convention.
"""
import ctypes as C
import inspect
import json
import math
from pathlib import Path
import sys
import traceback
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = []


class Injected(BaseException):
    """Also verifies rollback handles interruptions outside Exception."""


class Forbidden:
    def __getattr__(self, name):
        raise AssertionError('NPU access forbidden: ' + name)


class Tensor:
    def __init__(self, shape=(1,), dtype='bf16', device='cpu'):
        self.shape = tuple(shape)
        self.dtype, self.device = dtype, device
        self.ndim = len(self.shape)
        self.close_calls = 0

    def view(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return Tensor(shape, self.dtype, self.device)

    def to(self, *, dtype):
        return Tensor(self.shape, dtype, self.device)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return 4096

    def numel(self):
        return math.prod(self.shape)

    def __getitem__(self, key):
        return Tensor(self.shape[1:], self.dtype, self.device)

    def close(self):
        self.close_calls += 1
        raise AssertionError('borrowed tensor must not be closed')


class NativeFn:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


class Harness:
    def __init__(self, fault=None, close_faults=(), quantized=False, prepared=False):
        self.fault = fault
        self.original = Injected('injected build: ' + str(fault))
        self.cleanup_error = Injected('injected cleanup')
        self.close_faults = set(close_faults)
        self.quantized, self.prepared = quantized, prepared
        self.counts, self.events, self.plans, self.tensors = {}, [], [], []
        self.created, self.destroyed = [], []
        self.executor_created = False
        self.executor_attempts = 0
        self.workspace_status = self.repeatable_status = 0
        self.return_executor = True
        self.native_raise = None
        self.destroy_status = False
        self.weights = {}
        self.scratch = {'sentinel': Tensor()}
        self.sentinel = self.scratch['sentinel']
        self.parallel = types.SimpleNamespace(world=8, device='cpu')
        self.past = types.SimpleNamespace(main_kv=Tensor(), ring=256, pad=32)
        self.inputs = {k: Tensor() for k in ('x','incoming','output','freqs','slots','start','active')}
        self.torch = module('torch', bfloat16='bf16', float32='fp32', int64='i64',
                            int8='i8', float16='fp16', uint8='u8', npu=Forbidden(),
                            zeros=self.zeros, empty=self.empty)
        self.acl = types.SimpleNamespace(tensor=self.descriptor, workspace=self.workspace,
                    repeatable=self.repeatable, launch=lambda *a: 0,
                    base=types.SimpleNamespace(
                        aclDestroyTensor=NativeFn(self.destroy_tensor),
                        aclDestroyAclOpExecutor=NativeFn(self.destroy_executor)))

    def hit(self, stage):
        n = self.counts[stage] = self.counts.get(stage, 0) + 1
        self.events.append((stage, n))
        if self.fault == (stage, n):
            raise self.original
        return n

    def empty(self, shape, *, dtype, device):
        self.hit('allocation')
        return self.new_tensor(shape, dtype, device)

    def zeros(self, *shape, dtype, device):
        self.hit('zeros')
        return self.new_tensor(shape, dtype, device)

    def new_tensor(self, shape, dtype, device='cpu'):
        if isinstance(shape, int):
            shape = (shape,)
        t = Tensor(shape, dtype, device)
        self.tensors.append(t)
        return t

    def allocator(self, shape, dtype):
        self.hit('allocator')
        return self.new_tensor(shape, dtype)

    def weight(self, key):
        self.hit('weight')
        if key not in self.weights:
            dtype = 'i8' if self.quantized and key.endswith('.weight') and (
                '.attn.wq_' in key or '.attn.wkv.' in key or '.attn.wo_a.' in key or
                '.shared_experts.' in key) else 'bf16'
            self.weights[key] = Tensor(dtype=dtype)
        return self.weights[key]

    def prepare_weight(self, key, dtype):
        self.hit('prepare_weight')
        return Tensor(self.weights[key].shape, dtype)

    def prepare_scale(self, key):
        self.hit('prepare_scale')
        return Tensor(dtype='i64')

    def plan_factory(self, kind):
        def factory(*args, **kwargs):
            n = self.hit('plan')
            h = self
            class Plan:
                workspace_bytes = 17
                storage = args
                closed = False
                close_calls = 0
                def close(self):
                    self.close_calls += 1
                    self.closed = True
                    h.events.append(('close_plan', n))
                    if n in h.close_faults:
                        raise h.cleanup_error
            p = Plan()
            p.kind, p.number = kind, n
            self.plans.append(p)
            return p
        return factory

    def descriptor(self, tensor):
        n = self.hit('descriptor')
        self.created.append(n)
        return C.c_void_p(n)

    def workspace(self, *args):
        self.hit('workspace')
        C.cast(args[-2], C.POINTER(C.c_uint64))[0] = 32
        if self.return_executor:
            C.cast(args[-1], C.POINTER(C.c_void_p))[0] = 999
            self.executor_created = True
        if self.native_raise == 'workspace':
            raise self.original
        return self.workspace_status

    def repeatable(self, executor):
        self.hit('repeatable')
        if self.native_raise == 'repeatable':
            raise self.original
        return self.repeatable_status

    def destroy_executor(self, handle):
        assert handle.value == 999
        self.executor_attempts += 1
        self.events.append(('destroy_executor', 999))
        if 'executor' in self.close_faults:
            if self.destroy_status:
                return 31
            raise self.cleanup_error
        return 0

    def destroy_tensor(self, handle):
        self.destroyed.append(handle.value)
        self.events.append(('destroy_tensor', handle.value))
        if handle.value in self.close_faults:
            if self.destroy_status:
                return 32
            raise self.cleanup_error
        return 0

    def modules(self):
        queued = lambda f, *a, **kw: f
        return {
            'torch': self.torch,
            'ops': module('ops'), 'ops.decode': module('ops.decode'),
            'model': module('model'),
            'ops.queued': module('ops.queued', queued=queued),
            'model.decode_ops': module('model.decode_ops', native=lambda *a: None),
            'ops.decode.hc_finish': module('ops.decode.hc_finish', hc_finish=lambda *a: None),
            'ops.decode.kv_prepare': module('ops.decode.kv_prepare', kv_prepare=lambda *a: None),
            'ops.decode.gemm': module('ops.decode.gemm', Matmul=self.plan_factory('Matmul'),
                                     W8Matmul=self.plan_factory('W8Matmul')),
            'ops.decode.w4a8': module('ops.decode.w4a8', W4A8GroupedMatmul=self.plan_factory('W4')),
            'ops.decode.wo_b': module('ops.decode.wo_b', WoB=self.plan_factory('WoB')),
            'ops.decode.hc_project': module('ops.decode.hc_project', HCProject=self.plan_factory('HC')),
        }

    def load(self, tree, relative):
        assert tree == "candidate"
        path = ROOT / relative
        ns = {'__file__': str(path), '__name__': 'cpu_candidate'}
        exec(compile(path.read_text(encoding='utf8'), str(path), 'exec'), ns)
        return ns

    def vendor_init(self, cls):
        obj = cls.__new__(cls)
        self.obj = obj
        values = [Tensor() for _ in range(7)]
        self.borrowed = values
        cls.__init__(obj, *values, 16)
        return obj

    def target_init(self, cls, **overrides):
        obj = cls.__new__(cls)
        self.obj = obj
        h = self
        class Weights:
            def __getitem__(self, key):
                return h.weight(key)
        kwargs = dict(layer=0, batch=1, parallel=self.parallel, past=self.past,
            **self.inputs, norm_library=object(), attention_library=object(), gemm_library=object(),
            scratch=self.scratch, prepare_weight=self.prepare_weight,
            prepare_routed_scale=self.prepare_scale)
        if self.prepared:
            keys = ['attn.wq_a','attn.wq_b','attn.wkv','attn.wo_a',
                    'ffn.shared_experts.w1','ffn.shared_experts.w3','ffn.shared_experts.w2']
            kwargs['w8_prepared'] = {'layers.0.'+key: Tensor() for key in keys}
        kwargs.update(overrides)
        config = {'n_layers': 1, 'hc_eps': 1e-6, 'compress_ratios': [0]}
        cls.__init__(obj, Weights(), config, **kwargs)
        return obj

    def assert_borrowed(self):
        assert self.scratch['sentinel'] is self.sentinel
        assert all(t.close_calls == 0 for t in [*self.inputs.values(), self.past.main_kv,
                    *self.weights.values(), *self.scratch.values(), *getattr(self, 'borrowed', [])])

    def assert_vendor_closed(self):
        assert self.destroyed == self.created[::-1], (self.created, self.destroyed)
        assert self.executor_attempts == int(self.executor_created)
        if self.executor_created and self.destroyed:
            assert self.events.index(('destroy_executor', 999)) < self.events.index(('destroy_tensor', self.destroyed[0]))
        before = (list(self.destroyed), self.executor_attempts)
        self.obj.close()
        self.obj.close()
        assert before == (self.destroyed, self.executor_attempts)
        assert not self.obj.handles and not self.obj.executor
        assert self.obj.workspace is None and self.obj.values is None
        self.assert_borrowed()

    def assert_target_closed(self):
        assert all(p.close_calls == 1 for p in self.plans)
        attempts = [n for kind, n in self.events if kind == 'close_plan']
        assert attempts == [p.number for p in self.plans][::-1], attempts
        self.obj.close()
        self.obj.close()
        assert all(p.close_calls == 1 for p in self.plans)
        assert self.obj.closed and not self.obj.projections and not self.obj.storage
        assert set(self.obj.__dict__) == {'closed', 'projections', 'storage'}
        self.assert_borrowed()


def caught(call):
    try:
        call()
    except BaseException as exc:
        return exc
    raise AssertionError('expected injected failure')


def run_case(name, test):
    try:
        detail = test()
        RESULTS.append({'name': name, 'passed': True, 'detail': detail})
    except BaseException:
        RESULTS.append({'name': name, 'passed': False, 'error': traceback.format_exc()})


def environment(h):
    return patch.dict(sys.modules, h.modules())


def vendor_case(fault=None, status=None, close_faults=(), explicit=False, tree='candidate',
                no_executor=False, native_raise=None, destroy_status=False):
    h = Harness(fault, close_faults)
    h.return_executor = not no_executor
    h.native_raise, h.destroy_status = native_raise, destroy_status
    if status == 'workspace': h.workspace_status = 7
    if status == 'repeatable': h.repeatable_status = 8
    with environment(h):
        ns = h.load(tree, 'ops/decode/vendor_indexer.py')
        ns['_Acl'] = lambda: h.acl
        cls = ns['VendorIndexer']
        if fault or status or native_raise:
            exc = caught(lambda: h.vendor_init(cls))
            if fault or native_raise:
                assert exc is h.original
                tb_names = []
                tb = exc.__traceback__
                while tb:
                    tb_names.append(tb.tb_frame.f_code.co_name)
                    tb = tb.tb_next
                assert any(n in tb_names for n in ('hit','workspace','repeatable')), tb_names
            else:
                expected = ('aclnnLightningIndexerGetWorkspaceSize failed: 7; '
                    'rerun with ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=1 '
                    'to read the tiling check that rejected the operands') if status == 'workspace' else 'aclSetAclOpExecutorRepeatable failed: 8'
                assert type(exc) is RuntimeError and str(exc) == expected, repr(exc)
        else:
            obj = h.vendor_init(cls)
            assert not h.destroyed and not h.executor_attempts
            assert obj.run(stream=1) is h.borrowed[-1]
            if explicit and close_faults:
                exc = caught(obj.close)
                if destroy_status:
                    assert isinstance(exc, RuntimeError) and 'failed: ' in str(exc)
                else:
                    assert exc is h.cleanup_error
            else:
                obj.close()
        h.assert_vendor_closed()
    return {'descriptors': len(h.created), 'destroyed': len(h.destroyed),
            'executor_attempts': h.executor_attempts}


def target_case(fault=None, close_faults=(), explicit=False, tree='candidate',
                quantized=False, prepared=False, owner_alloc=False, overrides=None, expected=None):
    h = Harness(fault, close_faults, quantized, prepared)
    with environment(h):
        cls = h.load(tree, 'model/decode.py')['TargetLayer']
        kw = dict(overrides or {})
        if owner_alloc: kw['allocator'] = h.allocator
        if fault or expected:
            exc = caught(lambda: h.target_init(cls, **kw))
            if fault: assert exc is h.original, repr(exc)
            else: assert isinstance(exc, expected), repr(exc)
        else:
            obj = h.target_init(cls, **kw)
            assert len(h.plans) == 13 and all(p.close_calls == 0 for p in h.plans)
            assert obj.workspace_bytes == 64
            snapshot = dict(h.scratch)
            if explicit and close_faults:
                assert caught(obj.close) is h.cleanup_error
            else:
                obj.close()
            assert h.scratch == snapshot
        h.assert_target_closed()
    return dict(h.counts)


def half_initialized():
    h = Harness()
    with environment(h):
        for relative, name in [('model/decode.py','TargetLayer'),
                               ('ops/decode/vendor_indexer.py','VendorIndexer')]:
            cls = h.load('candidate', relative)[name]
            obj = cls.__new__(cls)
            obj.close()
            obj.close()
        target = h.load('candidate', 'model/decode.py')['TargetLayer']
        assert isinstance(caught(lambda: h.target_init(target, batch=0)), ValueError)
        h.assert_target_closed()


def singleton_retry():
    h = Harness()
    with environment(h):
        cls = h.load('candidate', 'ops/decode/vendor_indexer.py')['_Acl']
        seen = []
        def bind(obj):
            seen.append(obj)
            if len(seen) == 1:
                raise h.original
        cls._bind = bind
        assert caught(cls) is h.original
        assert cls._instance is None
        obj = cls()
        assert obj is cls() and obj is not seen[0] and len(seen) == 2


def acl_null_tensor():
    h = Harness()
    with environment(h):
        cls = h.load('candidate', 'ops/decode/vendor_indexer.py')['_Acl']
        acl = object.__new__(cls)
        acl.base = types.SimpleNamespace(aclCreateTensor=NativeFn(lambda *a: 0))
        t = types.SimpleNamespace(dim=lambda: 1, shape=(1,), stride=lambda: (1,),
                                 numel=lambda: 1, dtype='torch.bfloat16', data_ptr=lambda: 4096)
        exc = caught(lambda: acl.tensor(t))
        assert type(exc) is RuntimeError and str(exc) == 'aclCreateTensor returned null'






def early_failure_case(stage):
    h = Harness()
    with environment(h):
        ns = h.load('candidate', 'ops/decode/vendor_indexer.py')
        def fail_acl():
            raise h.original
        ns['_Acl'] = fail_acl if stage == 'acl' else lambda: h.acl
        cls = ns['VendorIndexer']
        obj = cls.__new__(cls)
        h.obj = obj
        class BadInt:
            def __int__(self):
                raise h.original
        args = [Tensor() for _ in range(7)]
        sparse_count = BadInt() if stage == 'int' else 16
        if stage == 'torch_import':
            with patch.dict(sys.modules, {'torch': None}):
                exc = caught(lambda: cls.__init__(obj, *args, sparse_count))
            assert isinstance(exc, ModuleNotFoundError)
        else:
            assert caught(lambda: cls.__init__(obj, *args, sparse_count)) is h.original
        h.assert_vendor_closed()


def borrowed_scratch_reuse():
    h = Harness()
    with environment(h):
        cls = h.load('candidate', 'model/decode.py')['TargetLayer']
        obj = h.target_init(cls, query_latent=Tensor((6,1280)))
        retained = dict(h.scratch)
        assert 'qn' not in retained
        obj.close()
        h.assert_target_closed()
        assert h.scratch == retained
        h2 = Harness(('plan', 8))
        h2.scratch = h.scratch
        h2.sentinel = h.sentinel
        with environment(h2):
            cls2 = h2.load('candidate', 'model/decode.py')['TargetLayer']
            assert caught(lambda: h2.target_init(cls2, query_latent=Tensor((6,1280)))) is h2.original
            h2.assert_target_closed()
        assert all(h.scratch[key] is value for key, value in retained.items())


def main():
    with patch.object(C, 'CDLL', side_effect=AssertionError('Native load forbidden')):
        for stage in ('acl', 'int', 'torch_import'):
            run_case('vendor/early_failure_'+stage, lambda stage=stage: early_failure_case(stage))
        run_case('target/reuse_borrowed_scratch_across_failed_build', borrowed_scratch_reuse)
        run_case('half_initialized_and_invalid_input_close', half_initialized)
        run_case('acl/failed_bind_retry', singleton_retry)
        run_case('acl/null_descriptor', acl_null_tensor)
        run_case('vendor/success_repeat_close', vendor_case)
        for stage, count in [('descriptor',8),('zeros',1),('allocation',1),('workspace',1),('repeatable',1)]:
            for n in range(1,count+1):
                run_case(f'vendor/fault_{stage}_{n}', lambda stage=stage,n=n: vendor_case((stage,n)))
        for status in ('workspace','repeatable'):
            run_case('vendor/status_'+status, lambda status=status: vendor_case(status=status))
            run_case('vendor/raise_after_executor_'+status, lambda status=status: vendor_case(native_raise=status))
            run_case('vendor/cleanup_preserves_'+status, lambda status=status: vendor_case(status=status,close_faults=('executor',8,4)))
        run_case('vendor/workspace_error_no_executor', lambda: vendor_case(status='workspace',no_executor=True))
        run_case('vendor/build_identity_cleanup_raises', lambda: vendor_case(('allocation',1),close_faults=('executor',8,3)))
        run_case('vendor/explicit_cleanup_raises_continues', lambda: vendor_case(close_faults=('executor',8,3),explicit=True))
        run_case('vendor/explicit_destroy_status_continues', lambda: vendor_case(close_faults=('executor',8,3),explicit=True,destroy_status=True))
        run_case('vendor/build_preserved_destroy_status', lambda: vendor_case(('allocation',1),close_faults=('executor',8,3),destroy_status=True))
        run_case('target/success_repeat_close', target_case)
        # Discover counts from the unmodified build logic, not from guessed counts.
        counts = target_case()
        for stage in ('plan','allocation','weight','prepare_weight','prepare_scale'):
            for n in range(1,counts[stage]+1):
                run_case(f'target/fault_{stage}_{n}', lambda stage=stage,n=n: target_case((stage,n)))
        counts = target_case(owner_alloc=True)
        for n in range(1,counts['allocator']+1):
            run_case(f'target/owner_allocator_fault_{n}', lambda n=n: target_case(('allocator',n),owner_alloc=True))
        for quantized, prepared, name in [(True,False,'W8'),(True,True,'prepared_W8')]:
            run_case('target/'+name+'_success', lambda q=quantized,p=prepared: target_case(quantized=q,prepared=p))
            for n in range(1,14):
                run_case(f'target/{name}_plan_fault_{n}', lambda n=n,q=quantized,p=prepared: target_case(('plan',n),quantized=q,prepared=p))
        run_case('target/cleanup_preserves_build_identity', lambda: target_case(('plan',13),close_faults=(12,8,1)))
        run_case('target/explicit_close_continues', lambda: target_case(close_faults=(13,8,1),explicit=True))
        run_case('target/missing_scale_callback', lambda: target_case(overrides={'prepare_routed_scale':None},expected=ValueError))
        run_case('target/invalid_compressed_borrowed', lambda: target_case(overrides={'compressed':{'borrowed':Tensor()}},expected=ValueError))
        run_case('target/incompatible_scratch', lambda: target_case(overrides={'scratch':{'qr':Tensor((123,))}},expected=ValueError))
    failed = [r for r in RESULTS if not r['passed']]
    report = {'cpu_only':True,'native_library_loads':'forbidden','npu_access':'forbidden',
              'python':sys.version,'total':len(RESULTS),'passed':len(RESULTS)-len(failed),
              'failed':len(failed),'cases':RESULTS}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    for result in failed:
        print('FAIL',result['name'],result['error'])
    print(f"CPU lifecycle tests: {report['passed']}/{report['total']} passed; {report['failed']} failed")
    return bool(failed)


if __name__ == '__main__':
    sys.exit(main())
