"""Ask the tie-prone served prompt of the engine directly, eight times over.

    torchrun --standalone --nproc-per-node=8 -m tests.tie_probe

determinism_probe asks with synthetic token rows and finds every solo run
identical.  The text the server forks on forks at its third token, where the
top two candidates sit a hair apart.  So put that exact row, encoded the way
the server encodes it, through the same solo channel and see whether the
engine alone forks or whether only the served path does.
"""
import sys

import torch
import torch.distributed as dist
from tokenizers import Tokenizer

sys.path.insert(0, '/mnt/data/kw/ljqinfer_dsv41f_tp8')

from server.encoding_dsv41 import encode_messages  # noqa: E402
from strategy.decode_worker import MAX_BATCH, WEIGHTS, bootstrap  # noqa: E402
from tests.determinism_probe import abreast, solo  # noqa: E402

TEXT = '用一句话说明什么是快速排序。'
N = 12          # tokens collected per run, well past the fork at three
REPEATS = 8


def served_row():
    prompt = encode_messages([{'role': 'user', 'content': TEXT}],
                             thinking_mode='chat')
    tok = Tokenizer.from_file(WEIGHTS + '/tokenizer.json')
    return tuple(tok.encode(prompt, add_special_tokens=False).ids), tok


def main():
    toks, tok = served_row()
    with torch.inference_mode():
        eng = bootstrap()
        say = dist.get_rank() == 0
        if say:
            print('TIE prompt tokens=%d %r' % (len(toks), toks), flush=True)
        runs = [tuple(solo(eng, toks, N)) for _ in range(REPEATS)]
        pool = set(runs)
        if say:
            for i, r in enumerate(runs):
                print('TIE solo run=%d %r' % (i, tok.decode(list(r))),
                      flush=True)
            print('TIE solo distinct=%d' % len(pool), flush=True)
        rows = [tuple(r) for r in abreast(eng, toks, N, MAX_BATCH)]
        if say:
            for i, r in enumerate(rows):
                print('TIE b4 row=%d in_solo=%s %r'
                      % (i, r in pool, tok.decode(list(r))), flush=True)


if __name__ == '__main__':
    main()
