# GLM53 cache / workspace repair

## Scope
- Reuse DSv4.1 strategy/cold_kv.py and host_arena.py byte-for-byte.
- PrefixState publishes target KV, sparse index KV and DFlash K/V at the same committed boundary. GPU slot pages live for the generator lifetime; reset changes lengths, not page ownership.
- Prefill/decode own fixed MoE, pair-attention, index and latent scratch. Request lengths select bounded views, not fresh plans. Index scans use logical context width, not the 144K allocation width.
- Idle command broadcast uses Gloo, with a 30-second heartbeat; GPU collectives remain NCCL.
- No replacement kernels. Production total/output/prefill-chunk limits remain 147456/8192/12288.

## Verified on node09
Evidence: /mnt/data2/kw/glm53_int4_tp8/service_audit/.
- cache_numeric_rank{0..7}.jsonl: 111 real-weight state fields, 1578 rows, byte-exact host roundtrip on every rank; full and suffix paths each repeated 16 times reproducibly.
- index_buffers_rank{0..7}.json: 10 cases/rank, prefill/decode scratch vs original allocation paths identical, contiguous stable pointers, CUDA graph replay. Supplied-scratch wrappers tested with torch.empty/full forbidden.
- Geometry: hot/cold, cross-page, branch, truncation and segmented growth through 166 rows.
- OpenAI protocol contract; HTTP hot/cold/growing conversation; cancellation and subsequent SSE completion; GPU utilization returns to zero while idle.
- 14206-token request crossed the 12288 prefill chunk; repeat hit 14205 tokens.

## Measured service timings (not universal performance guarantees)
| Request | Cached/input tokens | TTFT seconds |
|---|---:|---:|
| Short initial | 0/1590 | 1.261777 |
| Short hot | 1589/1590 | 0.261175 |
| Short cold after another prompt | 1589/1590 | 0.236490 |
| Conversation growth, first shape | 1590/1614 | 2.708909 |
| Conversation repeat | 1613/1614 | 0.233133 |
| Long, partially cached | 1587/14206 | 6.778452 |
| Long repeat | 14205/14206 | 0.889585 |

## Boundaries / remaining work
- Full prefill and cached suffix recompute are not bitwise equivalent and can produce different text. Full-vs-suffix next-token KL was approximately 0.00747 in the isolated fixture; forced Q8 top-1 comparisons agreed. This is NOT a general model-quality or cache-on/off token-equivalence proof. Hot and cold output agree in the tested request.
- Fixed scratch does not mean all PyTorch intermediates or first-shape compilation disappear. First unseen growth shape still showed a 2.71-second TTFT; no attribution to JIT alone has been proven.
- 144K allocation/configuration retained; full 144K generation and sustained memory-pressure eviction are not covered by these tests.
- Host arena grows on demand; budget is two full-context payloads per rank. No claim of end-to-end zero allocation or broad decode throughput acceleration.

## Reproduce small gates
python tests/audit_glm53_cache_geometry.py
python -m torch.distributed.run --standalone --nproc-per-node=8 tests/audit_glm53_index_buffers.py
python -c "import sys,runpy;sys.path.insert(0,'.');runpy.run_path('tests/test_openai_protocol.py',run_name='__main__')"
Run the eight-rank gate only with sufficient free device memory and no conflicting work.
