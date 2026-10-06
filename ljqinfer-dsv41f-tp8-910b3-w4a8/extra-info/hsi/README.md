# Decoder HSI candidate implementation

Implements paper section 2.3.2 and released model configuration: L20 Full produces a persistent per-query 16384-position pool after TP SUM, selecting 2047 earlier blocks by their maximum reduced score plus the newest block (8 positions each). L24/28/32/36 independently rescore only this pool and select Top512 logical KV IDs. Reuse layers retain their existing selection sharing. Every pool slot is overwritten; invalid positions are -1. Per-query causal bounds include uncommitted Q6 pending rows; page-table lookup handles committed rows.

No selectable old full-scan Reindex fallback was added. Full-layer scanning, short/full graphs, Past capacity and configured memory budget are unchanged. Rebuild ops/decode/attention.cpp into libdecode_attention.so and restart to use the new entry points.

## Reproduce mathematical checks

From the repository root, with the CANN/PyTorch NPU environment active and device 0 free:

```sh
bash ops/decode/build.sh ops/decode/attention.cpp ops/decode/libdecode_attention.so
python extra-info/hsi/check_hsi.py
python extra-info/hsi/check_hsi_scores.py
```

The references directly implement block-max selection and weighted ReLU dot-product equations on CPU, not the previous decode output.

Verified on A2-3:
- Pool and Top512: 12 dynamic graph-replay cases, B1/B4, widths 64/32768/131072; exact IDs including ties, candidate-cap crossing, inactive rows and per-query causal boundaries.
- Paged candidate scores: 6 graph-replay cases, B1/B4, missing pages, pending rows, invalid slots and inactive rows. Masks exact; maximum absolute error 0 for the sampled binary-fraction inputs. This does not imply arbitrary-input floating-point bit equality.
- TP8 operational integration: all 8 ranks completed, B1-4 graph capture and pool-sharing/fixed-Reindex-collective assertions passed; 80-token and 8000-token inputs each generated 12 output tokens. Sources were hash-checked unchanged during execution. Evidence: /tmp/decode_followup/hsi_impl/integration/{summary.json,generation.json,rank*.json}.
- Independent read-only review found no blocking semantic defect. Inactive selection remains all -1, matching existing sel_p4 initialization.

## Limits

The two generation prompts are below the 16384 candidate capacity and 8192 short-graph boundary. Long-context candidate exclusion is covered by the independent kernel checks, not yet by a full-model generation beyond the candidate cap. Full-graph capture is covered, not long-context end-to-end replay. No full-model numerical-equivalence claim is made. TP reduction determinism in existing captured graphs remains a separate issue. No speedup claim: candidate scoring uses a correctness-first vector implementation and pool/selection contain scalar heap work; dedicated optimization remains. The initial Full layer is still context-dependent, so the entire round is not O(1).
