from pathlib import Path
import unittest


class HotPathContractTest(unittest.TestCase):
    def test_decode_has_one_fixed_native_path(self):
        decode = Path("model/decode_graph.py").read_text()
        kernels = Path("ops/kernels.py").read_text()
        self.assertNotIn("LJQ_DISABLE_", decode + kernels)
        for dead in ("_gdn_attention_reference", "def causal_conv_sequence",
                     "def fused_gated_delta_update", "def gated_delta_update"):
            self.assertNotIn(dead, decode + kernels)
        self.assertIn("causal_conv_decode(", decode)
        self.assertIn("paired_l2norm_decode(", decode)
        self.assertIn("qk_rms_norm_rope_decode(", decode)
        self.assertIn("gated_delta_decode(", decode)
        self.assertIn("def _sequence_all_reduce", decode)
        self.assertIn("def _sequence_qk_norm_rope", decode)
        self.assertIn("all_reduce=self._sequence_all_reduce", decode)
        self.assertEqual(decode.count("return self._sequence_all_reduce(y)"), 2)
        collective = decode.split("def _sequence_all_reduce", 1)[1].split(
            "def _sequence_qk_norm_rope", 1)[0]
        self.assertIn("return self.engine.all_reduce(tensor)", collective)
        self.assertIn("gathered = self.rt.all_gather(tensor)", collective)
        self.assertIn("torch.add(gathered[0], gathered[1], out=tensor)", collective)
        self.assertIn("tensor.add_(gathered[3])", collective)
        self.assertIn("tensor.add_(gathered[2])", collective)
        self.assertNotIn("for row in range(self.batch_size)", collective)
        self.assertNotIn("torch.cat", collective)
        qk = decode.split("def _sequence_qk_norm_rope", 1)[1].split(
            "def _full_attention", 1)[0]
        self.assertNotIn("for row in range", qk)
        self.assertNotIn("torch.cat", qk)
        self.assertEqual(qk.count("K.qk_rms_norm_rope_decode("), 2)
        self.assertIn("token_count in (8, 16, 24, 32)", kernels)
        self.assertIn("q.shape == (token_count, 6, 256)", kernels)
        self.assertIn("k.shape == (token_count, 1, 256)", kernels)
        bindings = Path("model/hccl.py").read_text()
        self.assertIn("lib.HcclGroupStart.argtypes = []", bindings)
        self.assertIn("lib.HcclGroupEnd.argtypes = []", bindings)

    def test_decode_native_abi_is_b1_to_b4_q8(self):
        decode = Path("model/decode_graph.py").read_text()
        kernels = Path("ops/kernels.py").read_text()
        conv = Path("ops/ascendc/gdn_conv_vec.cpp").read_text()
        recurrent = Path("ops/ascendc/gdn_recurrent_b1_baseptr.cpp").read_text()
        recurrent_baseptr = Path("ops/ascendc/gdn_recurrent_baseptr.cpp").read_text()
        self.assertIn("self.batch_size not in (1, 2, 3, 4)", decode)
        self.assertIn("batch not in (1, 2, 3, 4)", kernels)
        self.assertNotIn("pending[:, 0].copy_(base_state)", decode)
        self.assertIn("base_state=base_state", decode)
        self.assertIn("gdn_recurrent_baseptr_launch", kernels)
        self.assertIn("ljq_gdn_recurrent_baseptr_aiv.so", kernels)
        self.assertIn("ljq_qk_rms_norm_rope_aiv.so", kernels)
        self.assertIn("ljq_gdn_l2_pair_aiv.so", kernels)
        self.assertIn("batch_native = token_count > 8", kernels)
        self.assertIn("batch_native = rows > 48", kernels)
        self.assertIn("baseState", recurrent_baseptr)
        self.assertIn("inputBatchStride", conv)
        self.assertIn("for (uint32_t row = batchRow; row < batchRow + 1; ++row)", conv)
        self.assertIn("gdn_conv_vec<<<blocksPerBatch, nullptr, stream>>>", conv)
        self.assertIn("gdn_recurrent_b1base_direct<<<24, nullptr, stream>>>", recurrent)
        self.assertIn("if (batch != 1) return 1;", recurrent)
        self.assertIn("baseState", recurrent)
        self.assertNotIn("unused_batch", recurrent)
        self.assertNotIn("LJQ_GDN_B1Q8_DISABLE", kernels)
        self.assertNotIn("gdn_recurrent_b1q8_launch", kernels)
        self.assertIn("beta_logits = bv.to(vv.dtype).contiguous()", decode)
        self.assertNotIn("torch.sigmoid(bv.float())", decode)
        self.assertNotIn("torch.sigmoid(beta_logits.float()).to(v.dtype)", kernels)
        self.assertEqual(recurrent.count("Muls(betaf, betaf, -1.0f, T * NV);"), 1)
        self.assertEqual(recurrent_baseptr.count("Muls(betaf, betaf, -1.0f, T * NV);"), 2)

    def test_dflash_batch_conv_writes_preallocated_row_slices(self):
        dflash = Path("model/dflash2.py").read_text()
        conv = dflash.split("def _grouped_conv(self, hidden", 1)[1].split(
            "def _row_linear", 1)[0]
        kernels = Path("ops/kernels.py").read_text()
        self.assertIn("output = torch.empty_like(hidden)", conv)
        self.assertIn("out=output[lo:hi]", conv)
        self.assertNotIn("torch.cat", conv)
        self.assertIn("out=None", kernels.split(
            "def dflash_grouped_conv_b1q8", 1)[1].split(
                "def causal_conv_decode", 1)[0])

    def test_dflash_batch_reduce_uses_one_exact_tp4_collective(self):
        dflash = Path("model/dflash2.py").read_text()
        collective = dflash.split("def _row_all_reduce", 1)[1].split(
            "def _mlp", 1)[0]
        self.assertIn(
            "self.owner.engine.collective.all_gather(tensor)", collective)
        self.assertIn(
            "torch.add(gathered[0], gathered[1], out=tensor)", collective)
        self.assertIn("tensor.add_(gathered[3])", collective)
        self.assertIn("tensor.add_(gathered[2])", collective)
        self.assertNotIn("for batch_row in range", collective)
        self.assertNotIn("torch.cat", collective)

    def test_dynamic_cancel_is_only_honored_at_boarding_safe_points(self):
        api = Path("model/model_api.py").read_text()
        dynamic = api.split("def decode_dflash_batch_dynamic(", 1)[1].split(
            "def generate_dflash_batch(", 1)[0]
        self.assertIn("cancel_safe_point = steps % boarding_interval_steps == 0",
                      dynamic)
        self.assertIn("honor_cancels=cancel_safe_point", dynamic)
        self.assertIn(
            "if select_active_rows is None and not cancel_safe_point:", dynamic)
        self.assertIn(
            "selected_rows = [row for row in active_rows if not done[row]]",
            dynamic)
        self.assertLess(
            dynamic.index("if select_active_rows is None and not cancel_safe_point:"),
            dynamic.index("honor_cancels=cancel_safe_point"))
        self.assertNotIn("cancels[row].is_set()):", dynamic)

    def test_decode_entries_keep_draft_ids_resident_until_verify(self):
        api = Path("model/model_api.py").read_text()
        static = api.split("def decode_dflash_batch(", 1)[1].split(
            "def decode_dflash_batch_dynamic(", 1)[0]
        dynamic = api.split("def decode_dflash_batch_dynamic(", 1)[1].split(
            "def generate_dflash_batch(", 1)[0]
        for path in (static, dynamic):
            self.assertIn('getattr(self.drafter, "draft_batch_device", None)', path)
            self.assertIn('if callable(device_draft) and hasattr(graph, "prepare_draft")', path)
            self.assertIn("graph.prepare_draft(", path)
            self.assertIn("resident_paths = device_paths.cpu().tolist()", path)
            self.assertLess(path.index("graph.prepare_draft("),
                            path.index("graph.replay()"))
            self.assertLess(path.index("_global_argmax_rows("),
                            path.index("resident_paths = device_paths.cpu().tolist()"))

    def test_prefill_has_no_token_decoder_or_debug_switch(self):
        blocks = Path("model/blocks.py").read_text()
        self.assertNotIn("def _gdn_token", blocks)
        self.assertNotIn("LJQ_DISABLE_FUSED_PREFILL_ATTN", blocks)

    def test_gdn_ab_projection_is_one_packed_gemm(self):
        weights = Path("model/weights.py").read_text()
        blocks = Path("model/blocks.py").read_text()
        decode = Path("model/decode_graph.py").read_text()
        self.assertIn("ab: Any", weights)
        self.assertNotIn("    a: Any", weights)
        self.assertNotIn("    b: Any", weights)
        for hot_path in (blocks, decode):
            self.assertIn("K.bf16_linear(hidden, w.ab)", hot_path)
            self.assertNotIn("K.bf16_linear(hidden, w.a)", hot_path)
            self.assertNotIn("K.bf16_linear(hidden, w.b)", hot_path)


if __name__ == "__main__":
    unittest.main()
