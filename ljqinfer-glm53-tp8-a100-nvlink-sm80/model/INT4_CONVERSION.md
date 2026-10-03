# GLM53 TP8 INT4 conversion
Run from the unique repository:
```
python -m model.convert_glm53_all /mnt/data2/kw/GLM-5.3-UNCENSORED-FP8 /mnt/data2/kw/glm53_int4_tp8
```
Eight independent GPU workers produce layers 3..77, 8 rank files/layer (600 total).
Existing G64 symmetric signed INT4 MSE search + four least-squares steps is unchanged.
Gate/up/down routed weights are INT4; no duplicate FP8 down bank. Router, norms and shared expert retain the existing loader dtypes. Native MTP layer 78 is excluded.
This is an engine-specific cache, NOT a standalone Hugging Face checkpoint: original FP8 source remains required for attention/indexer, dense layers and global weights. Do not delete the source.
Each shard is saved to a temporary file, read back and compared tensor-by-tensor (including finite checks), fsynced, then atomically renamed. A receipt includes SHA256 and source/converter manifest digest. Restart verifies completed receipts and resumes missing work; changed source stats/config/index or converter rejects mixing. Manifest is not a full source-content hash.
Monitor conversion.status.json, rank0..7.status.json and rank0..7.log. Only supervisor state complete plus all 600 receipts indicates conversion completion; quality and full-engine validation remain separate.
Initial live validation: first two complete layers, all eight ranks, 16 roundtrip-checked shards; second layer 20.2-21.1s/rank including save/checksum. Not a full-conversion completion claim.
