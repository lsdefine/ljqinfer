"""Decode FP16 consumers: independent CPU oracle and optional prior-library A/B.
Run from repository root, with idle NPU:
  PYTHONPATH=. python scripts/test_decode_half_consumers.py
  ... --baseline-lib /path/to/pre-change/libdecode_norm.so
No model weights, service, or sampling-policy changes.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import torch
import torch_npu
from model.decode_ops import native


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-lib', type=Path)
    ap.add_argument('--library', type=Path, default=Path('ops/decode/libdecode_norm.so'))
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261014)
    lib = C.CDLL(str(args.library.resolve()))
    old = C.CDLL(str(args.baseline_lib.resolve())) if args.baseline_lib else None
    finite = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    finite = finite[torch.isfinite(finite)]
    stream = torch.npu.Stream()
    cases = []
    for batch in (1, 2, 3, 4):
        rows = batch * 36
        for kind in ('act', 'combine'):
            cols = 576 if kind == 'act' else 5120
            # Rotate finite patterns through all positions; every finite FP16
            # value reaches both gate and up halves, even for B1.
            width = rows * (288 if kind == 'act' else cols)
            patterns = [None] + list(range(0, len(finite), width))
            for offset in patterns:
                if offset is None:
                    host = (torch.randn(rows, cols) * 4).half()
                else:
                    indices = (torch.arange(width) + offset) % len(finite)
                    values = finite[indices].reshape(rows, -1)
                    host = torch.cat((values, values.flip(1)), dim=1) if kind == 'act' else values
                aux = torch.rand(rows) if kind == 'act' else torch.randperm(rows, dtype=torch.int64)
                x, other = host.npu(), aux.npu()
                shape = (rows, 288) if kind == 'act' else (batch * 6, 5120)
                dtype = torch.bfloat16 if kind == 'act' else torch.float32
                y = torch.empty(shape, dtype=dtype, device='npu')
                symbol = 'dec_routed_' + kind
                h = host.bfloat16().float()
                if kind == 'act':
                    gate = h[:, :288].clamp(max=10)
                    up = h[:, 288:].clamp(-10, 10)
                    ref = ((gate / (torch.exp(-gate) + 1)) * up * aux[:, None]).bfloat16()
                else:
                    ids = aux.reshape(batch * 6, 6).sort(dim=1).values
                    selected = h[ids]
                    ref = selected[:, 0] + 0.0
                    for j in range(1, 6):
                        ref = ref + selected[:, j]
                native(lib, symbol, (x, other, y), batch, check_status=True)
                torch.npu.synchronize()
                if kind == 'act':
                    # Ascend exp/div can round across a BF16 boundary vs CPU;
                    # the optional old native library must remain bit exact.
                    torch.testing.assert_close(y.cpu().float(), ref.float(), rtol=0.008, atol=1e-6)
                else:
                    assert torch.equal(y.cpu(), ref), (batch, kind, offset, 'CPU sum')
                if old:
                    rounded = x.bfloat16()
                    z = torch.empty_like(y)
                    native(old, symbol, (rounded, other, z), batch)
                    torch.npu.synchronize()
                    assert torch.equal(y, z), (batch, kind, offset, 'old native')
                eager = y.clone()
                torch.npu.synchronize()
                with torch.npu.stream(stream):
                    native(lib, symbol, (x, other, y), batch, check_status=True)
                stream.synchronize()
                assert torch.equal(y, eager), 'side stream'
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    native(lib, symbol, (x, other, y), batch, check_status=True)
                for _ in range(3):
                    graph.replay()
                torch.npu.synchronize()
                assert torch.equal(y, eager), 'graph replay'
                del graph
                cases.append(dict(batch=batch, kind=kind, offset=offset))
    print(json.dumps(dict(complete=True, cases=len(cases), batches=[1, 2, 3, 4],
                         finite_fp16_patterns=len(finite), cpu_oracle=True,
                         previous_native_bit_exact=old is not None,
                         side_stream=True, graph_replay=True)))


if __name__ == '__main__':
    main()
