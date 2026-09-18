import ast
from pathlib import Path
import unittest


class GDNChunkContractTest(unittest.TestCase):
    def test_prefill_has_one_size_independent_chunk_path(self):
        source = Path("model/blocks.py").read_text()
        tree = ast.parse(source)
        fn = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "gdn_attention")
        calls = [node for node in ast.walk(fn)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        chunk_calls = [node for node in calls
                       if node.func.attr == "chunk_gated_delta_sequence"]
        recurrent_calls = [node for node in calls
                           if node.func.attr == "gated_delta_update"]
        self.assertEqual(len(chunk_calls), 1)
        self.assertEqual(recurrent_calls, [])
        self.assertIn("y, chunk_states = K.chunk_gated_delta_sequence", source)
        self.assertIn(
            "padded_tokens = (sequence_tokens + 63) // 64 * 64", source)
        self.assertIn("ctx.gdn_conv.index_copy_(0, state_indices, conv)", source)
        self.assertIn("ctx.gdn_recurrent.index_copy_(0, state_indices, rec)", source)
        self.assertIn("multi-sequence GDN requires exactly one decode row", source)
        for dead_dispatch in ("tokens >= 64", "aligned =", "range(aligned",
                              "return_chunk_states"):
            self.assertNotIn(dead_dispatch, source)

    def test_chunk_operator_has_one_fixed_return_contract(self):
        source = Path("ops/kernels.py").read_text()
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "KernelBackend")
        fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "chunk_gated_delta_sequence")
        returns = [node for node in ast.walk(fn) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertIsInstance(returns[0].value, ast.Tuple)
        self.assertEqual(len(returns[0].value.elts), 2)
        self.assertNotIn("return_chunk_states", ast.unparse(fn))


if __name__ == "__main__":
    unittest.main()
