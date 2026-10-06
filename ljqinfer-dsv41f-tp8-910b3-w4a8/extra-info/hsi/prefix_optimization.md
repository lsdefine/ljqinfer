# Prefix dispatch and HSI optimization

Replaces the 8192/maximum-capacity split with one verify program captured for geometric prefixes 8192 through 524288. Selects smallest prefix covering max(starts)+6. Length-dependent dispatch remains; the 8K-to-524K jump is removed. More startup captures are required. Full-layer work still scales with context; fixed 16384 candidate pool/top512 unchanged.

HSI kernels: vector clearing/block maxima, batched GM-to-UB reads, strict-less-than heap pruning preserving ties. Experimental sel_p2 changes NOT included.

Validation: compilation, 12 pool/select/replay oracle cases and 6 paged-score cases passed (score error 0). TP8 all ranks passed B1-B4 prefix capture and two 12-token B1 generations.

Full decode round wall times, NOT per-token latency:
- 8000 input: 49.2193, 46.9925 ms.
- 8400 input: 47.4263, 47.3809 ms.
No observed cross-8192 cliff in these samples; still above <40 ms. Only two samples per case, no controlled old 8400 run: no speedup-factor claim.
Evidence: /tmp/decode_followup/unified_source/smoke2/generation.json and summary.json. Summary timing extractor reports zero samples due to filtering; times above are from generation.json/full_decode_steps. Initial smoke test used wrong attribute score_graphs; corrected verify_graphs test passed.
