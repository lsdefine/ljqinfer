"""CPU gate for production GDN snapshot destination metadata, not GPU numerics."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch


def production_indices(batch):
    tree = ast.parse(Path("model/decode_graph.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DecodeGraphRunner")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    matches = [n for n in ast.walk(init) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
        and t.value.id == "self" and t.attr == "_gdn_state_indices" for t in n.targets)]
    assert len(matches) == 1, "snapshot metadata must have one production definition"
    scope = dict(torch=torch, self=SimpleNamespace(device="cpu"), b=batch, q=8)
    exec(compile(ast.Module(body=matches, type_ignores=[]), "production_metadata", "exec"), scope)
    return scope["self"]._gdn_state_indices


def assert_row_ownership(indices, batch):
    assert indices.dtype == torch.int32
    assert indices.shape == (batch * 8,)
    assert torch.equal(indices.reshape(batch, 8), torch.arange(batch * 8, dtype=torch.int32).reshape(batch, 8))
    assert indices.unique().numel() == batch * 8
    # Distinct candidate snapshots must remain distinct for any accepted count.
    for count in (1, 3, 8):
        slots = indices.reshape(batch, 8)[:, count - 1]
        assert torch.equal(slots, torch.arange(batch, dtype=torch.int32) * 8 + count - 1)


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_production_snapshot_indices_have_disjoint_row_ownership(batch):
    assert_row_ownership(production_indices(batch), batch)


@pytest.mark.parametrize("batch", [2, 3, 4])
def test_gate_rejects_original_repeated_index_mutation(batch):
    with pytest.raises(AssertionError):
        assert_row_ownership(torch.arange(8, dtype=torch.int32).repeat(batch), batch)
