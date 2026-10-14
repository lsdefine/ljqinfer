# GLM53 shared-prefix LRU repair (2026-10-12)

Implementation commit: 259f8a6. Lookup pins only entries contributing backing slices through the actual matched boundary. New forks no longer inherit unrelated suffix entries. LRU policy and allocation budgets are unchanged.

## Validation
- Old implementation fails the new shared-prefix fanout regression, both with and without arena.
- Fixed: 58 pytest checks passed, including async CUDA, protocol and template.
- Eight-rank NCCL: 24 forks per rank, exact public/target/private-draft payload restore; all eight rank reports passed, including divergence/recovery.
- Production restarted from committed code. Ten branches of 32,575 input tokens, each repeated once: 20/20 HTTP requests succeeded; all repeats hit 32,574 tokens and recomputed one. Cold/warm outputs matched exactly.
- Total first-pass input: 325,750 tokens, exceeding the 295,040 token-equivalent budget. Median TTFT: new branches 6.992826 s; repeats 0.2571735 s.
- Revisit branch 1 after pressure: 25,143 tokens retained, 7,432 recomputed; immediate repeat hit 32,574. This demonstrates partial chunk eviction and successful reinsertion. An exploratory assertion expecting hit<1000 was wrong: eviction need not remove the entire branch.
- Zero admission-skip warnings and tracebacks during the 20-request audit.

Scope: GLM53 only. Other repositories not patched. Long-duration production soak and representative workload hit-rate remain unmeasured; 99.997% applies only to controlled immediate repeats.

Unit reproduction: `python -m pytest -q tests/test_cold_branch_lru.py tests/test_cold_capacity.py tests/test_async_writeback.py tests/test_openai_protocol.py tests/test_service_template_contract.py`

Distributed reproduction: `COLD_TEST_TOKENS=128 torchrun --standalone --nproc-per-node=8 tests/test_cold_single_copy.py`

Audit artifacts and current service log: `/mnt/data/kw/glm53_cache_fix_20261012_cj3cad31/`.
