
## 2026-09-13 decode 进度
- 绿: tests/test_decode_attention.py(2) 算子级与 prefill 逐值一致 + 对 Past 零副作用
- 绿: tests/test_decode_layer.py(1) released 全尺寸层级 parity + 零副作用 (commit 9f78dbb)
- 全量回归: 219 passed / 38 skipped
- 新增: DecodeAttention.commit(accepted) 发布路径 -- 全接受走影子 carry 快路径,
  部分接受在真 carry 上用 compress_rows 重压缩 accepted 前缀 (待测)
- 契约: scratch.staged/carry/pending 均按 **kv_source 层** 索引; compress 对 carry in-place
- 下一步: commit parity 测试 -> decode block(mHC/router/MoE) -> 图捕获 -> B1Q6 测速
