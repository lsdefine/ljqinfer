import ast
from pathlib import Path
import unittest


class PagedDecodeContractTest(unittest.TestCase):
    def test_single_query_uses_unmasked_nonsparse_fia(self):
        source = Path("model/blocks.py").read_text()
        tree = ast.parse(source)
        fn = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_paged_attention")
        body = ast.unparse(fn)
        self.assertIn("query_tokens = int(q.shape[0])", body)
        self.assertIn("mask = None", body)
        self.assertIn("sparse_mode = 0", body)
        self.assertIn("if query_tokens != 1", body)
        self.assertIn("sparse_mode = 3", body)
        self.assertIn("atten_mask=mask", body)
        self.assertIn("actual_seq_lengths=[query_tokens]", body)
        self.assertIn("sparse_mode=sparse_mode", body)


if __name__ == "__main__":
    unittest.main()
