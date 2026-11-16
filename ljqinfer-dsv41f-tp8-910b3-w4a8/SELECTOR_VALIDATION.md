# Compressed decode selector validation

Implementation: `b582553` (following compressed-coordinate fix `3dfc302`).

For ratio=2, fold even/odd query rows into the batch dimension, run one vendor indexer plan, and restore original query order. Each residue group has the correct compressed-key causal staircase. The key bank is shared, not duplicated; ratio=1 preserves the original query layout.

## Acceptance

- 13 local unit tests passed, including 8 selector tests.
- Real NPU independent CPU top-k oracle: 65 cases / 858 queries passed.
- Full service matrix: 21 cases passed, including up to 262K retrieval and concurrent requests.
- Complete streaming, four disconnect/recovery pairs, and twelve concurrent 1024-token generations passed.
- Three additional fresh-prefix retrieval requests passed.

| Input size | Previous production TTFT (s) | Packed selector TTFT (s) | Fresh-prefix repeat (s) |
|---|---:|---:|---:|
| 8K | 2.825 | 1.208 | 1.203 |
| 32K | 4.800 | 4.303 | 3.934 |
| 131K | 25.760 | 17.429 | 16.671 |
| 262K | 63.136 | 53.687 | Not tested |

These are observed runs, not a guarantee of universal speedup. Decode model-step times were approximately unchanged, sometimes 1–2% slower. The earlier two-plan run's severe TTFT anomaly has not been causally attributed; external traffic and other confounders were not conclusively excluded.

The packed plan reduced startup allocated memory by about 731.56 MiB per device versus the two-plan candidate, leaving approximately 2.75 MiB above the previous production implementation.

## Deployment

A23 was validated on port 8001 and then returned to production port 8000. Only the frontend was restarted for the port change; the validated engine remained running. Port 8000 health and authenticated real inference passed; port 8001 no longer listened. A21/A22 were not modified in this rollout.
