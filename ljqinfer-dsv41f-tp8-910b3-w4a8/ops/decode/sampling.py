"""Full-vocabulary p=.95: one sort, inverse CDF, no inverse permutation.

Persistent B1..4/Q6 graph buffers; bias precedes temperature and truncation.
Temperature zero retains lowest-ID argmax ties. RNG state is shared across TP
ranks and advanced once per call. Caller owns stream ordering and input/output.
"""
import ctypes as C
from pathlib import Path
import torch


class TopPSampler:
    def __init__(self, batch, allocate, parallel, temperature, seed, bias, output):
        if not 1 <= batch <= 4:
            raise ValueError('top-p requires B1..4')
        self.batch, self.rows, self.parallel = batch, batch * 6, parallel
        self.temperature, self.seed, self.bias, self.output = temperature, seed, bias, output
        a, r, n = allocate, self.rows, 129280
        self.gathered = a((8, r, 16160), torch.float32)
        self.logits = a((r, n), torch.float32)
        self.values = a((r, n), torch.float32)
        self.ids = a((r, n), torch.int64)
        self.probs = a((r, n), torch.float32)
        self.cdf = a((r, n), torch.float32)
        self.temp = a((batch, 1, 1), torch.float32)
        # Native aligned reads include padding for row counts not divisible by 4.
        self.greedy = a(((r + 3) // 4 * 4,), torch.int64)
        self.library = C.CDLL(str(Path(__file__).with_name('libdecode_sampling.so')))
        self.finish = self.library.dec_top_p_finish
        self.finish.argtypes, self.finish.restype = [C.c_void_p] * 7 + [C.c_uint32] * 2, C.c_int
        self.args = tuple(C.c_void_p(x.data_ptr()) for x in
            (self.cdf, self.ids, temperature, seed, output, self.greedy)) + (r, n)

    def sample(self, local_logits):
        if self.parallel.rank == 7:
            local_logits[:, 128822 - 7 * 16160].add_(self.bias.reshape(-1))
        self.parallel.logits(local_logits, out=self.gathered)
        self.logits.view(self.rows, 8, 16160).copy_(self.gathered.permute(1, 0, 2))
        torch.argmax(self.logits, dim=-1, out=self.greedy[:self.rows])
        torch.clamp(self.temperature.view(self.batch, 1, 1), min=1e-5, out=self.temp)
        self.logits.view(self.batch, 6, 129280).div_(self.temp)
        torch.sort(self.logits, dim=-1, out=(self.values, self.ids))
        torch.softmax(self.values, dim=-1, out=self.probs)
        torch.cumsum(self.probs, dim=-1, out=self.cdf)
        rc = self.finish(C.c_void_p(torch.npu.current_stream().npu_stream), *self.args)
        if rc:
            raise RuntimeError('top-p finish: %d' % rc)
