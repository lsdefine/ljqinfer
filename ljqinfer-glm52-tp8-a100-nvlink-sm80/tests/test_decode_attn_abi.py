"""Static lock for the single B-agnostic decode-attention boundary."""
from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABI_FILE = ROOT / "ops" / "decode_attn.py"
MODEL_DIR = ROOT / "model"
PUBLIC_SHA256 = "e950283ae2ed3d46b07908e90d148b80997736de61b506a8fdf57747803c8291"
POSITIONAL = ["op"]
KEYWORD_ONLY = [
    "x", "positions", "kv_workspace", "page_tables", "context_lengths",
    "attn_norm", "q_a", "q_a_norm", "q_b", "kv_a", "kv_a_norm",
    "k_b", "v_b", "attn_out",
]


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    found = [n for n in module.body
             if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise AssertionError(f"expected one {name}, found {len(found)}")
    return found[0]


class DecodeAttentionAbiTest(unittest.TestCase):
    def test_public_boundary_is_frozen(self) -> None:
        fn = _function(_module(ABI_FILE), "decode_attn")
        self.assertEqual([a.arg for a in fn.args.args], POSITIONAL)
        self.assertEqual([a.arg for a in fn.args.kwonlyargs], KEYWORD_ONLY)
        self.assertIsNone(fn.args.vararg)
        self.assertIsNone(fn.args.kwarg)
        digest = hashlib.sha256(ast.dump(
            fn, annotate_fields=True, include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest, PUBLIC_SHA256)

    def test_bn_implementation_matches_public_signature(self) -> None:
        fn = _function(_module(ABI_FILE), "_decode_attn_bn_impl")
        self.assertEqual([a.arg for a in fn.args.args], POSITIONAL)
        self.assertEqual([a.arg for a in fn.args.kwonlyargs], KEYWORD_ONLY)
        self.assertIsNone(fn.args.vararg)
        self.assertIsNone(fn.args.kwarg)

    def test_deleted_b1_paths_stay_deleted(self) -> None:
        abi_src = ABI_FILE.read_text(encoding="utf-8")
        self.assertNotIn("_decode_attn_b1_impl", abi_src)
        cpp = (ROOT / "ops" / "nova_decode_attn.cpp").read_text(encoding="utf-8")
        self.assertNotIn("forward_rank_cached_inplace_tc_k0", cpp)
        self.assertNotIn("forward_rank_paged_batch_q2", cpp)

    def test_model_paths_use_only_the_public_boundary(self) -> None:
        calls = 0
        forbidden = []
        for path in MODEL_DIR.glob("*.py"):
            module = _module(path)
            for node in ast.walk(module):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "decode_attn"):
                    calls += 1
                if isinstance(node, ast.Attribute) and node.attr in {
                    "forward_rank_cached_inplace_tc_k0",
                    "forward_rank_paged_batch_k0",
                }:
                    forbidden.append((path.name, node.lineno, node.attr))
        # Base B1/B2 leaf moved to the private projected ABI
        # (forward_rank_paged_batch_k0_projected) so attn projections overlap
        # the TP8 latent collective; the public decode_attn boundary now has
        # exactly one caller: the MTP leaf.
        self.assertEqual(calls, 1)  # MTP leaf only
        self.assertEqual(forbidden, [])

    def test_old_parallel_python_interfaces_are_gone(self) -> None:
        self.assertFalse((ROOT / "ops" / "decode_attn_batch.py").exists())
        blocks = (MODEL_DIR / "blocks.py").read_text(encoding="utf-8")
        self.assertNotIn("decode_attn_rank", blocks)


if __name__ == "__main__":
    unittest.main()
