"""Static contract for allocation-free routed-MoE decode."""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA = [
    "prefill_moe_dequant.cu",
    "prefill_moe_dequant_v11_packed.cu",
    "prefill_moe_dequant_v12_shared_tn.cu",
]


class DecodeMoeAbiTest(unittest.TestCase):
    def test_cpp_public_abi_is_explicit_out(self):
        src = (ROOT / "ops/prefill_moe_fused.cpp").read_text()
        match = re.search(r"torch::Tensor moe_rank_forward_decode_fused\((.*?)\)\{", src, re.S)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1).count("torch::Tensor"), 8)
        body = src[match.end():src.index("\n}", match.end())]
        self.assertIn("moe_decode_iq_fused_out_cuda(g,u,d,x,ei,ew,h,y)", body)
        self.assertNotIn("moe_decode_iq_fused_cuda(g,u,d,x,ei,ew)", body)

    def test_model_has_two_down_cache_workspace_calls(self):
        path = ROOT / "model/blocks.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "decode"
                 and isinstance(n.func.value, ast.Attribute)
                 and n.func.value.attr == "down_cache"]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(len(call.args) == 10 for call in calls))
        src = path.read_text()
        self.assertEqual(src.count("K.down_cache.decode("), 2)
        self.assertEqual(src.count("ws.ws_routed_h[r], routed,"), 2)
        self.assertNotIn("yf.zero_()", src)
        self.assertEqual(src.count("K.residual_add_copy.combine_moe_f32_to_f16("), 3)
        self.assertEqual(src.count("out = ws.partial[r]"), 3)
        self.assertNotIn("out.copy_(routed)", src)

    def test_route_nonfinite_fallback_contract(self):
        src = (ROOT / "ops/decode_moe_route_q13.cu").read_text()
        self.assertIn("__shared__ int has_nonfinite;", src)
        self.assertIn("if (!isfinite(p) || !isfinite(sel[tid]))", src)
        self.assertIn("if (has_nonfinite && tid == 0)", src)
        self.assertIn("if (!has_nonfinite && tid < 128)", src)
        self.assertIn("for (int i = 0; i < 128; ++i)", src)

    def test_graph_owns_hidden_workspace(self):
        src = (ROOT / "model/decode_backend.py").read_text()
        self.assertEqual(src.count("g.ws_routed_h.append("), 1)
        self.assertIn("T * TOPK, LOCAL_ROUTED_FF", src)
        self.assertIn("g.ws_routed.append(torch.zeros(T, D, device=dev, dtype=torch.float16))", src)

    def test_cuda_out_entry_is_allocation_free(self):
        for name in CUDA:
            src = (ROOT / "ops" / name).read_text()
            body = src.split("torch::Tensor moe_decode_iq_fused_out_cuda", 1)[1]
            body = body.split("torch::Tensor moe_decode_iq_fused_cuda", 1)[0]
            self.assertNotIn("torch::empty", body, name)
            self.assertNotIn("torch::zeros", body, name)
            self.assertIn("return y;", body, name)


if __name__ == "__main__":
    unittest.main()
