"""The numpy port vs the official torch selection (extracted from the pinned source)."""
import ast
import types

import numpy as np
import pytest
from conftest import ROOT

from ccpilot.compressors.specpc import SpecPCParams, official_select, pilot_select

torch = pytest.importorskip("torch")
OFFICIAL = ROOT / "external/specpc/draft_approx_llm/specpc/specpc.py"


def _official_select_fn():
    import pandas as pd
    import torch.nn.functional as F

    mod = ast.parse(OFFICIAL.read_text())
    keep = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in ("_unique", "specpc_select_input_token_indices")]
    ns = {"torch": torch, "F": F, "np": np, "pd": pd, "SpecPCConfig": object}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "official_specpc", "exec"), ns)
    return ns["specpc_select_input_token_indices"]


def _cfg(p: SpecPCParams, cap):
    return types.SimpleNamespace(kernel_size=p.kernel_size, neighbor_tokens=p.neighbor_tokens, pool_type=p.pool_type,
                                 max_capacity_prompt=cap, window_size=p.window_size)


@pytest.mark.parametrize("L,cap", [(3000, 1024), (5000, 2048), (900, 1024), (17000, 2048)])
def test_exact_match_without_pooling_ties(L, cap):
    fn = _official_select_fn()
    p = SpecPCParams(pool_type=None)
    rng = np.random.default_rng(L)
    x = rng.random(L)  # continuous -> no ties
    ref = fn(torch.tensor(x[None], dtype=torch.float64), _cfg(p, cap))[0].numpy()
    ours = official_select(x, cap, p)
    np.testing.assert_array_equal(ours, ref)


def test_with_max_pooling_tie_sensitivity():
    """Max-pooling creates plateaus; torch.topk tie order is implementation-defined.
    We only require high overlap and identical length; the exact agreement rate is reported."""
    fn = _official_select_fn()
    p = SpecPCParams()
    agree = []
    for seed in range(5):
        x = np.random.default_rng(seed).random(15000).astype(np.float32)
        ref = set(fn(torch.tensor(x[None]), _cfg(p, 2048))[0].tolist())
        ours = set(official_select(x, 2048, p).tolist())
        assert len(ours) == len(ref) == 2048
        agree.append(len(ours & ref) / 2048)
    print("pooled agreement (CPU torch vs stable port):", [round(a, 4) for a in agree])
    assert min(agree) > 0.5


def test_pilot_select_stays_in_context_and_budget():
    p = SpecPCParams()
    x = np.random.default_rng(0).random(12000)
    sel = pilot_select(x, 100, 11950, 1900, p)
    assert len(sel) == 1900 and sel.min() >= 100 and sel.max() < 11950
    assert np.all(np.diff(sel) > 0)
    assert pilot_select(x, 100, 600, 1900, p).tolist() == list(range(100, 600))


def test_official_selector_backend_matches_port_without_ties():
    from ccpilot.compressors.specpc import load_official_select

    sel, origin = load_official_select(allow_source_extract=True)
    p = SpecPCParams(pool_type=None)
    x = np.random.default_rng(3).random(9000)
    np.testing.assert_array_equal(sel(x, 2048, p), official_select(x, 2048, p))
    # pilot: official function on the context slice with window_size=0 == port (no pooling)
    a, b, C = 120, 8900, 1900
    np.testing.assert_array_equal(sel(x[a:b], C, p, window_size=0) + a, pilot_select(x, a, b, C, p))
