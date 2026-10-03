"""Independent eager/graph oracle for the shared prefill dispatch boundary.

Run on GPU7 separately from full-capacity service warmup.
"""
import argparse
import gc
import json
import runpy
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.cuda.set_device(7)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.00625)
    torch.backends.cuda.matmul.allow_tf32 = False
    test = runpy.run_path(str(Path(__file__).with_name('test_moe_four.py')))
    from ops.moe_common import Workspace, DENSE_PREFILL_MIN_TOKENS
    cutoff = DENSE_PREFILL_MIN_TOKENS
    records = []
    for tokens in (cutoff - 1, cutoff, cutoff + 1):
        ws = Workspace.create(tokens, 3, 5, 256, 128, 'cuda:7')
        assert (ws.weight is not None) == (tokens >= cutoff)
        del ws
        for group in (64, 128):
            for fp8 in (False, True):
                records += test['case']((tokens, 256, 128, 5, 3), group,
                                        fp8, 7300 + tokens, True)
                gc.collect()
                torch.cuda.empty_cache()
    # Wider intermediate dimensions use the same prefill path.
    ws = Workspace.create(cutoff, 3, 5, 128, 128, 'cuda:7')
    assert ws.weight is not None
    del ws
    records += test['case']((cutoff, 128, 128, 5, 3), 64, False, 9101, True)
    report = dict(status='passed', records=records,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', len(records), flush=True)


if __name__ == '__main__':
    main()
