# DFlash fusion engine integration

Baseline e6ab07c, node09 TP8. Fresh sequential baseline/candidate engine suite.
Share fixed-128 RoPE frequencies across six draft/KV-append layers; fuse
rotation preserving both BF16 product roundings and greedy path selection.
Original RMS reductions, model weights, routing and collectives unchanged.

## Results
- 12 prompts including code/thinking, all eight ranks completed.
- Mean complete decode round: 44.103706 -> 43.616257 ms (-1.105%).
- All 12 cases faster in this paired suite.
- Mean per-case decode TPS: 68.986989 -> 69.756461.
- All eight ranks: token hashes, step counts, accepted counts/histograms identical.
- Cache cold/hot/reset outputs and cache-hit metadata match corresponding baseline paths.
  Restore timing excluded from equality; cold==hot is not claimed.
- 58 bitwise leaf checks: strided RoPE, high positions, path ties.
- All eight ranks passed KV append graph n=1..8 regression.

## Boundaries
Prior isolated draft graph 3.595 -> 3.214 ms is not engine round timing.
Engine suite excludes prefill/HTTP; complete rounds use synchronized boundary.
Separate profile deltas must not be summed into an end-to-end claim.
One fresh paired suite, not repeated ABBA or long-context/concurrency certification.
40ms remains unmet. Aggressive RMS fusion rejected due to numerical mismatch.
HTTP remains stopped; no push performed.

Evidence: /mnt/data2/kw/glm53_int4_tp8/service_audit/decode40/draft_fusion/engine_integration
See integration_summary.json, baseline/, candidate/, leaf.log,
append_graph.log and append_production.rank0..7.json.
