from pathlib import Path
from unittest.mock import patch

import torch
from ops.kernels import K, C
import unittest


class HotPathContractTest(unittest.TestCase):
    def test_decode_has_one_fixed_native_path(self):
        decode = Path("model/decode_graph.py").read_text()
        kernels = Path("ops/kernels.py").read_text()
        self.assertNotIn("LJQ_DISABLE_", decode + kernels)
        for dead in ("_gdn_attention_reference", "def causal_conv_sequence",
                     "def fused_gated_delta_update", "def gated_delta_update"):
            self.assertNotIn(dead, decode + kernels)
        self.assertIn("gdn_conv_norm_decode(", decode)
        self.assertNotIn("K.causal_conv_decode(", decode)
        self.assertNotIn("K.paired_l2norm_decode(", decode)
        self.assertIn("qk_rms_norm_rope_decode(", decode)
        self.assertIn("gated_delta_decode(", decode)
        self.assertIn("def _sequence_all_reduce", decode)
        self.assertIn("def _sequence_qk_norm_rope", decode)
        self.assertIn("mlp = self._mlp(normed, w.mlp)", decode)
        self.assertEqual(decode.count("return self._sequence_all_reduce(y)"), 2)
        collective = decode.split("def _sequence_all_reduce", 1)[1].split(
            "def _sequence_qk_norm_rope", 1)[0]
        self.assertIn("reduced = tensor.float()", collective)
        self.assertIn("self.engine.all_reduce(reduced)", collective)
        self.assertNotIn("tensor.copy_(reduced)", collective)
        self.assertIn("return reduced", collective)
        self.assertNotIn("all_gather", collective)
        self.assertNotIn("if self.batch_size", collective)
        self.assertNotIn("for row in range(self.batch_size)", collective)
        self.assertNotIn("torch.cat", collective)
        qk = decode.split("def _sequence_qk_norm_rope", 1)[1].split(
            "def _full_attention", 1)[0]
        self.assertNotIn("for row in range", qk)
        self.assertNotIn("torch.cat", qk)
        self.assertEqual(qk.count("K.qk_rms_norm_rope_decode("), 2)
        self.assertIn("return C.qk_norm_rope(", kernels)
        self.assertIn("return C.recurrent(", kernels)
        self.assertIn("return C.conv(", kernels)
        for retired in ("torch_npu", "ctypes", "gdn_aot", "os.getenv", "os.environ"):
            self.assertNotIn(retired, kernels)
        self.assertIn("torch.cuda.CUDAGraph()", decode)
        self.assertIn("torch.cuda.graph(graph, pool=pool)", decode)
        self.assertIn("torch.cuda.synchronize(self.device)", decode)
        self.assertNotIn("torch.npu", decode)
        self.assertNotIn("from .hccl", decode)

    def test_verify_packing_has_single_leaf_and_frozen_m32(self):
        decode = Path("model/decode_graph.py").read_text()
        body = decode.split("def _linear(", 1)[1].split("def _mlp", 1)[0]
        self.assertIn("K.pack_verify_rows(flat)", body)
        self.assertNotIn("torch.zeros", body)
        self.assertIn("F.linear(packed, weight)", body)
        x = torch.empty((8, 16), dtype=torch.bfloat16)
        marker = object()
        with patch.object(K, "_bf16_cuda"), patch.object(C, "pack_verify_rows", return_value=marker) as leaf:
            self.assertIs(K.pack_verify_rows(x), marker)
            leaf.assert_called_once_with(x)
        for shape in ((0, 16), (33, 16), (8, 0)):
            with self.assertRaises(ValueError):
                C.pack_verify_rows(torch.empty(shape))

    def test_reduce_fp32_boundary_defers_rounding_to_norm(self):
        # Execute the actual method body on CPU with only NCCL substituted.
        # CUDA/NCCL numerical equivalence is separately gated on all four ranks.
        import ast
        from types import SimpleNamespace
        source = Path("model/decode_graph.py").read_text()
        cls = next(n for n in ast.parse(source).body
                   if isinstance(n, ast.ClassDef) and n.name == "DecodeGraphRunner")
        method = next(n for n in cls.body
                      if isinstance(n, ast.FunctionDef) and n.name == "_sequence_all_reduce")
        scope = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     "<production_reduce_cpu_contract>", "exec"), scope)
        reduce = scope["_sequence_all_reduce"]
        for batch in (1, 2, 3, 4):
            with self.subTest(batch=batch):
                x = torch.arange(batch * 8 * 16, dtype=torch.float32).reshape(batch*8, 16).to(torch.bfloat16)
                before = x.clone()
                calls = []
                def collective(value):
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertEqual(value.shape, x.shape)
                    self.assertTrue(torch.equal(x, before))
                    calls.append(value)
                    value.mul_(4)
                    return value
                owner = SimpleNamespace(engine=SimpleNamespace(all_reduce=collective))
                result = reduce(owner, x)
                self.assertIs(result, calls[0])
                self.assertEqual(result.dtype, torch.float32)
                self.assertEqual(len(calls), 1)
                self.assertTrue(torch.equal(x, before))
                self.assertTrue(torch.equal(result, before.float()*4))

    def test_decode_cuda_abi_is_b1_to_b4_q8(self):
        decode = Path("model/decode_graph.py").read_text()
        kernels = Path("ops/kernels.py").read_text()
        cuda = Path("ops/cuda_ops.py").read_text()
        self.assertIn("self.batch_size not in (1, 2, 3, 4)", decode)
        self.assertIn("self.query_width != 8", decode)
        self.assertNotIn("pending[:, 0].copy_(base_state)", decode)
        self.assertIn("base_state=base_state", decode)
        self.assertIn("g, beta_logits = K.gdn_gate_prepare(", decode)
        self.assertNotIn("torch.nn.functional.softplus(", decode)
        self.assertNotIn("torch.sigmoid(bv.float())", decode)
        self.assertNotIn("torch.sigmoid(beta_logits.float()).to(v.dtype)", kernels)
        recurrent = cuda.split("def _recurrent(", 1)[1].split("def recurrent(", 1)[0]
        self.assertIn("tl.load(BASE +", recurrent)
        self.assertNotIn("tl.store(BASE", recurrent)
        self.assertIn("for t in range(8)", recurrent)
        self.assertEqual(recurrent.count("tl.sigmoid("), 1)
        self.assertIn(".to(BETA.dtype.element_ty).to(tl.float32)", recurrent)
        self.assertIn("slot = tl.load(IDX + row)", recurrent)
        self.assertIn("tl.store(STATES +", recurrent)

    def test_cuda_leaf_guard_rejects_cpu_without_launch(self):
        with patch.object(C, "conv") as leaf:
            x = torch.empty(1, 8, 2560, dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "same-device BF16"):
                K.causal_conv_decode(x, x, x, x)
            leaf.assert_not_called()
        with self.assertRaisesRegex(ValueError, "immutable BF16 base_state"):
            K.gated_delta_decode(*([None] * 9))

    def test_conv_shape_validation_and_pending_identity_without_gpu(self):
        # Scoped leaf/device mocks test the ABI, not numerical CUDA execution.
        for b in (1, 2, 3, 4, 5):
            with self.subTest(batch=b):
                x = torch.empty(b, 8, 2560, dtype=torch.bfloat16)
                base = torch.empty(b, 3, 2560, dtype=torch.bfloat16)
                pending = torch.empty_like(x)
                weight = torch.empty(4, 2560, dtype=torch.bfloat16)
                marker = object()
                with patch.object(K, "_bf16_cuda"), patch.object(C, "conv", return_value=marker) as leaf:
                    if b == 5:
                        with self.assertRaisesRegex(ValueError, "B1..4,Q8"):
                            K.causal_conv_decode(x, base, pending, weight)
                        leaf.assert_not_called()
                    else:
                        self.assertIs(K.causal_conv_decode(x, base, pending, weight), marker)
                        self.assertEqual(leaf.call_count, 1)
                        self.assertEqual(len(leaf.call_args.args), 4)
                        for got, expected in zip(leaf.call_args.args, (x, base, weight, pending)):
                            self.assertIs(got, expected)
                    leaf.reset_mock()
                    with self.assertRaisesRegex(ValueError, "B1..4,Q8"):
                        K.causal_conv_decode(x[:, :7], base, pending, weight)
                    leaf.assert_not_called()

    def test_recurrent_routes_immutable_base_and_snapshot_indices(self):
        for b in (1, 2, 3, 4, 5):
            with self.subTest(batch=b):
                # Meta storage validates shape/stride with no GPU allocation.
                make = lambda *shape, dtype=torch.bfloat16: torch.empty(shape, device="meta", dtype=dtype)
                q, k = make(b*8, 4, 128), make(b*8, 4, 128)
                v = make(b*8, 12, 128)
                g, beta = make(b*8, 12, dtype=torch.float32), make(b*8, 12)
                state, base = make(b*8, 12, 128, 128), make(b, 12, 128, 128)
                idx = make(b*8, dtype=torch.int32)
                lengths, accepted = make(b, dtype=torch.int32), make(b, dtype=torch.int32)
                marker = object()
                with patch.object(K, "_bf16_cuda"), patch.object(C, "recurrent", return_value=marker) as leaf:
                    args = (q, k, v, g, beta, state, lengths, idx, accepted)
                    if b == 5:
                        with self.assertRaisesRegex(ValueError, "B1..4"):
                            K.gated_delta_decode(*args, base_state=base)
                        leaf.assert_not_called()
                    else:
                        self.assertIs(K.gated_delta_decode(*args, base_state=base), marker)
                        self.assertEqual(leaf.call_count, 1)
                        expected = (q, k, v, g, beta, base, state, idx)
                        self.assertEqual(len(leaf.call_args.args), len(expected))
                        for got, want in zip(leaf.call_args.args, expected):
                            self.assertIs(got, want)
                    leaf.reset_mock()
                    bad = (q, k, v, g.to(torch.bfloat16), beta, state, lengths, idx, accepted)
                    with self.assertRaisesRegex(ValueError, "FP32"):
                        K.gated_delta_decode(*bad, base_state=base)
                    leaf.assert_not_called()

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
        self.assertIn("K.bf16_linear(hidden, w.ab)", blocks)
        self.assertIn("self._linear_packed(hidden, w.ab)", decode)
        for hot_path in (blocks, decode):
            self.assertNotIn("K.bf16_linear(hidden, w.a)", hot_path)
            self.assertNotIn("K.bf16_linear(hidden, w.b)", hot_path)


if __name__ == "__main__":
    unittest.main()
