"""Bitwise tests for DFlash RoPE/path fusion; run with python on a CUDA GPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.glm53_dflash import rope
from ops.dflash_fused import frequencies, rope_only, dflash_path_select

@torch.inference_mode()
def main():
    torch.manual_seed(713)
    checks = 0
    for n in [1, 3, 8, 65, 257]:
        for start in [0, 63, 2040, 8191, 131071]:
            positions = torch.arange(start, start + n, device='cuda')
            freq = frequencies(positions, torch.bfloat16)
            for heads in [1, 8]:
                # Include QKV-like non-contiguous token strides.
                storage = torch.randn(n, heads + 2, 128, device='cuda', dtype=torch.bfloat16)
                x = storage[:, :heads]
                actual = rope_only(x, freq)
                expected = rope(x, positions)
                assert torch.equal(actual, expected), (n, start, heads)
                checks += 1
    for dtype in [torch.float32, torch.bfloat16]:
        for batch in [1, 3]:
            candidate = torch.randint(0, 100000, (batch, 7, 16), device='cuda')
            scores = torch.randn(batch, 7, 16, 16, device='cuda', dtype=dtype)
            for ties in [False, True]:
                if ties:
                    scores.zero_()
                previous = torch.zeros(batch, device='cuda', dtype=torch.long)
                rows = torch.arange(batch, device='cuda')
                tokens = []
                for step in range(7):
                    previous = scores[rows, step, previous].argmax(-1)
                    tokens.append(candidate[rows, step, previous])
                expected = torch.stack(tokens, 1)
                actual = dflash_path_select(scores, candidate)
                assert torch.equal(actual, expected), (dtype, batch, ties)
                checks += 1
    torch.cuda.synchronize()
    print(f'DFLASH_FUSED_EXACT checks={checks}', flush=True)

if __name__ == '__main__':
    main()
