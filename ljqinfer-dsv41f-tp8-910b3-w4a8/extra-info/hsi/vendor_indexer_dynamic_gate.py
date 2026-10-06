"""Does a captured dispatch honour a sequence length changed after capture?

Decode replays one captured graph for every context length, so the op must read
actual_seq_lengths from its device tensors at kernel time rather than freezing
them into tiling. Capture at 128K, rewrite the length tensors to a short
context, replay, and compare against an eager run at that short length.
"""
import sys, torch, torch_npu

sys.path.insert(0, '/data/ljqinfer_dsv41f_tp8')
from ops.decode.vendor_indexer import VendorIndexerPlan

DEV = 'npu:0'
BATCH, ROWS, HEADS, DIM, SC = 1, 6, 32, 128, 512
BLOCK, LONG, SHORT = 128, 131072, 8192


def operands(blocks):
    torch.manual_seed(0)
    q = torch.randn(BATCH, ROWS, HEADS, DIM, dtype=torch.bfloat16, device=DEV)
    key = torch.randn(blocks, BLOCK, 1, DIM, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(BATCH, ROWS, HEADS, dtype=torch.bfloat16, device=DEV)
    table = torch.arange(blocks, dtype=torch.int32, device=DEV).view(BATCH, blocks)
    sq = torch.full((BATCH,), ROWS, dtype=torch.int32, device=DEV)
    sk = torch.full((BATCH,), LONG, dtype=torch.int32, device=DEV)
    idx = torch.zeros(BATCH, ROWS, 1, SC, dtype=torch.int32, device=DEV)
    return q, key, w, table, sq, sk, idx


def main():
    torch.npu.set_device(DEV)
    blocks = LONG // BLOCK
    q, key, w, table, sq, sk, idx = operands(blocks)
    plan = VendorIndexerPlan(q, key, w, sq, sk, table, idx, SC)
    plan.run()
    torch.npu.synchronize()

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):
            plan.run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        plan.run()
    torch.npu.synchronize()

    sk.fill_(SHORT)
    idx.zero_()
    graph.replay()
    torch.npu.synchronize()
    replayed = idx.clone()

    reference = torch.zeros_like(idx)
    eager = VendorIndexerPlan(q, key, w, sq, sk, table, reference, SC)
    eager.run()
    torch.npu.synchronize()

    beyond = int((replayed >= SHORT).sum())
    print('replay max index', int(replayed.max()), 'out of range', beyond)
    print('eager   max index', int(reference.max()))
    same = torch.equal(replayed.sort(-1).values, reference.sort(-1).values)
    print('replay honours the shortened length:', beyond == 0 and same)
    print('DONE')


main()
