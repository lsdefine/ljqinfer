# GLM53 weights and tmpfs cache

`model.weights.load_tp8(rank=r)` loads one rank. A cache miss builds final-layout tensors into `/dev/shm/ljqinfer_glm53/<fingerprint>/`; hits only mmap/copy to the target device. No holding daemon needed. Reboot removes tmpfs; first subsequent load rebuilds. Keep source FP8 and INT4 shards on disk.

Prebuild eight ranks: `python -m model.wcache build --wait 3600`. The wait permits concurrent INT4 conversion. `build.status.json`, per-rank logs and per-fingerprint `rankN.complete.json` expose completion.

Includes 78 layers, vocabulary-sharded BF16 embedding/head, FP16 attention/indexer, dense and shared FFN, G64 routed INT4. No native MTP, CP delta, old GGUF or alternate FP8 down copy. Source/index/config/code fingerprint + shard receipts invalidate stale entries. Locks serialize same-entry builders. Temporary writes are finite-checked and roundtrip-verified before atomic publication. Cold loads verify source INT4 SHA256; warm loads check stat/header/receipt, not full hashes. Privileged adversarial same-stat data tampering is outside this cache contract.

`Weights` is rank-local, not the legacy all-GPU container. Vocabulary embedding must use local masked lookup + all-reduce, logits require all-gather. `Engine.load` remains intentionally guarded; this change does not claim whole-model generation works.

Tests: `python -m pytest -q tests/test_glm53_wcache.py` (11 CPU regressions). Real first-build/hot-hit rank0 globals, dense0, shared3/12, full10 passed exact CPU/H2D equality. Whole-model eight-rank audit: `python tests/test_glm53_weights_load.py --output /mnt/data2/kw/glm53_int4_tp8/wcache_audit`; disables cold builders and compares every loaded tensor with shm plus vocabulary globals with original source. Inspect status before claiming completion.

## Verified 2026-10-01

INT4 conversion: 600/600 shards complete. Final-layout shm: 632/632 files (78 layers + globals per rank). Eight simultaneous rank loads passed, 10.32–12.34 seconds each, with cold builders disabled. Every loaded tensor compared exactly against its CPU snapshot; globals also compared with source. CUDA allocated weight bytes: 55,492,511,744 per rank (~51.68 GiB). These numbers exclude KV, execution workspace and graph capture, and do not certify full-model generation. Audit receipts: `/mnt/data2/kw/glm53_int4_tp8/wcache_audit/`. CPU cache regression: 11 passed.
