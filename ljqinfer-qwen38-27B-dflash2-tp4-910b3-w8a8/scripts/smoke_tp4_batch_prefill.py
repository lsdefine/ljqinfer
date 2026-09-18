import argparse
import json
import os
import time

import torch
import torch_npu

from model.decode import decode_step
from model.model_api import ModelExecution

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, required=True)
args = parser.parse_args()
os.environ["LJQ_SPMD_RANK"] = str(args.rank)


def page_set(row):
    return {int(page) for page in row if int(page) >= 0}


def assert_disjoint(rows, name):
    sets = [page_set(row) for row in rows]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                raise AssertionError(f"{name} physical-page alias rows {i},{j}")


def hidden_metrics(actual, expected):
    actual = actual.detach().float().cpu().reshape(-1)
    expected = expected.detach().float().cpu().reshape(-1)
    return {
        "max_abs": float((actual - expected).abs().max()),
        "cosine": float(torch.nn.functional.cosine_similarity(
            actual.reshape(1, -1), expected.reshape(1, -1)).item()),
    }


def global_top1(model, rows):
    hidden = torch.cat([row.reshape(1, -1) for row in rows], dim=0)
    return model._global_argmax_rows(model.engine.local_logits(hidden))


t0 = time.perf_counter()
model = ModelExecution.startup(args.rank)
model.rt.barrier()
startup_s = time.perf_counter() - t0

# The partial path is deliberately split identically on both sides:
# 6144-token prefix, then a 12856-token suffix split as 12288+568 by the
# production chunker.  A final teacher-forced token validates continuation
# state rather than only comparing the exported boundary hidden row.
ids_partial = [int((17 + i * 13) % 248000) for i in range(19001)]
ids_exact = [int((29 + i * 7) % 248000) for i in range(12289)]
prefix_lengths = (6144, 12288)
prompt_lengths = (19000, 12288)
capacities = (19001, 12289)

base_t0 = time.perf_counter()
base = model.prefill_batch(
    (ids_partial[:prefix_lengths[0]], ids_exact[:prefix_lengths[1]]),
    sequence_ids=(0, 1), max_lengths=capacities)
partial_records = base.export_prefix_records(0, 0, prefix_lengths[0])
exact_records = base.export_prefix_records(1, 0, prefix_lengths[1])
assert tuple(r["end"] for r in partial_records) == tuple(
    range(1024, prefix_lengths[0] + 1, 1024))
assert tuple(r["end"] for r in exact_records) == tuple(
    range(1024, prefix_lengths[1] + 1, 1024))

base_final = (
    model._stream_prefill(ids_partial[prefix_lengths[0]:prompt_lengths[0]],
                          prefix_lengths[0], 0),
    base.rows[1].last_hidden,
)
model.rt.synchronize()
base_prompt_top1 = global_top1(model, base_final)
base_s = time.perf_counter() - base_t0

hit_t0 = time.perf_counter()
hit = model.prefill_batch(
    (ids_partial[:prompt_lengths[0]], ids_exact[:prompt_lengths[1]]),
    sequence_ids=(2, 3), max_lengths=capacities,
    restored_records=(partial_records, exact_records))
model.rt.synchronize()
hit_final = tuple(row.last_hidden for row in hit.rows)
hit_prompt_top1 = global_top1(model, hit_final)
teacher_logits = decode_step(
    model.engine,
    [ids_partial[prompt_lengths[0]], ids_exact[prompt_lengths[1]],
     ids_partial[prompt_lengths[0]], ids_exact[prompt_lengths[1]]],
    sequence_ids=[0, 1, 2, 3], return_logits=True)
model.rt.synchronize()
teacher_top1 = model._global_argmax_rows(teacher_logits)
hit_s = time.perf_counter() - hit_t0

prompt_metrics = [hidden_metrics(hit_final[i], base_final[i]) for i in range(2)]
teacher_metrics = [hidden_metrics(teacher_logits[i + 2], teacher_logits[i])
                   for i in range(2)]
assert hit.lengths == prompt_lengths
assert hit.restored_lengths == prefix_lengths
assert tuple(int(model.engine.cache.lengths[sid].item())
             for sid in (0, 1, 2, 3)) == (19001, 12289, 19001, 12289)
assert tuple(model.drafter.context_lengths[sid]
             for sid in (0, 1, 2, 3)) == (19000, 12288, 19000, 12288)
assert_disjoint(
    [row.target_page_table for row in base.rows + hit.rows], "target/all")
assert_disjoint(
    [row.dflash_page_table for row in base.rows + hit.rows], "dflash/all")
if base_prompt_top1 != hit_prompt_top1:
    raise AssertionError(
        f"prompt top1 mismatch base={base_prompt_top1} hit={hit_prompt_top1}")
if teacher_top1[:2] != teacher_top1[2:]:
    raise AssertionError(f"teacher top1 mismatch rows={teacher_top1}")
if any(row["max_abs"] != 0.0 for row in prompt_metrics + teacher_metrics):
    raise AssertionError(
        f"equivalent cold path mismatch prompt_hidden={prompt_metrics} "
        f"teacher_logits={teacher_metrics}")
assert tuple(hit.decode_handoff()["last_hidden"].shape) == (2, 5120)

result = {
    "status": "ok", "rank": args.rank,
    "startup_seconds": startup_s, "baseline_seconds": base_s,
    "hit_seconds": hit_s, "prompt_lengths": prompt_lengths,
    "restored_lengths": hit.restored_lengths,
    "partial_suffix_chunks": [12288, 568],
    "prompt_hidden": prompt_metrics, "teacher_logits": teacher_metrics,
    "prompt_top1": hit_prompt_top1, "teacher_top1": teacher_top1,
    "target_pages": [len(row.target_resident_pages) for row in hit.rows],
    "dflash_pages": [len(row.dflash_resident_pages) for row in hit.rows],
}
if args.rank == 0:
    print("BATCH_PREFILL_NPU_OK " + json.dumps(result, sort_keys=True), flush=True)
hit.release()
base.release()
model.rt.barrier()
model.rt.destroy()
