import ast
import hashlib
from pathlib import Path
import unittest


class GDNAOTContractTest(unittest.TestCase):
    def setUp(self):
        self.source_path = Path("ops/gdn_aot.py")
        self.source = self.source_path.read_text()
        self.tree = ast.parse(self.source)

    def _literal_assignment(self, name):
        node = next(
            node for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name
                    for target in node.targets))
        return ast.literal_eval(node.value)

    def test_frozen_assets_exactly_match_hash_manifest(self):
        expected = self._literal_assignment("_EXPECTED_SHA256")
        root = Path("ops/gdn_aot")
        actual = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file() and path.name != "README.md"
        }
        self.assertEqual(actual, expected)

    def test_loader_fails_closed_and_validates_fp32(self):
        self.assertIn("GDN AOT asset hash mismatch", self.source)
        self.assertIn("stale GDN AOT module loaded", self.source)
        self.assertIn("GDN AOT requires fp32 kg/vc/bc/gc inputs", self.source)
        self.assertNotIn("except Exception", self.source)

    def test_production_path_defaults_to_aot_with_explicit_torch_ab(self):
        source = Path("ops/kernels.py").read_text()
        self.assertIn('"LJQ_GDN_PREPARE_BACKEND", "aot"', source)
        self.assertIn('prepare_backend not in ("aot", "torch")', source)
        self.assertIn("from .gdn_aot import prepare_wy", source)
        self.assertIn("u, w = prepare_wy(kg, vc, bc, gc)", source)


if __name__ == "__main__":
    unittest.main()
