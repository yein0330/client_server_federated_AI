import ast
import re
import string
from collections import Counter

from conftest import ROOT

from ccpilot.data import select_splits
from ccpilot.textutil import join_units, normalize_answer, parse_passages, qa_f1, split_units

OFFICIAL_METRICS = ROOT / "external/specpc/dataset/longbench/metrics.py"


def _official_fns():
    src = OFFICIAL_METRICS.read_text()
    mod = ast.parse(src)
    keep = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in ("normalize_answer", "f1_score", "qa_f1_score")]
    ns = {"re": re, "string": string, "Counter": Counter}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "official_metrics", "exec"), ns)
    return ns


def test_qa_f1_matches_official():
    ns = _official_fns()
    cases = [("Miller v. California", "Miller v. California"), ("The Beatles band", "Beatles"),
             ("yes", "no"), ("Gary Cooper and David Niven", "David Niven"), ("", "x"), ("an apple.", "Apple")]
    for p, g in cases:
        assert normalize_answer(p) == ns["normalize_answer"](p)
        assert abs(qa_f1(p, g) - ns["qa_f1_score"](p, g)) < 1e-12


def test_units_tile_every_context(lb_items):
    for r in lb_items:
        units = split_units(r["context"])
        assert "".join(u.text(r["context"]) for u in units) == r["context"]
        assert [u.idx for u in units] == list(range(len(units)))


def test_passage_titles_parse(lb_items):
    for r in lb_items[:20]:
        ps = parse_passages(r["context"])
        assert ps and all(p["title"] for p in ps)
        assert [p["num"] for p in ps] == list(range(1, len(ps) + 1))


def test_splitter_keeps_initials():
    ctx = "Passage 1:\nX\nBased on the trilogy by E. L. James. It was released in the U.S. Senate era. Next one.It joined."
    texts = [u.text(ctx) for u in split_units(ctx) if u.kind == "sentence"]
    assert any("E. L. James." in t for t in texts)
    assert any("U.S. Senate" in t for t in texts)
    assert any(t.startswith("It joined") for t in texts)


def test_join_units_inserts_space_for_nonadjacent():
    ctx = "Passage 1:\nT\nA one.Bee two. C three."
    us = [u for u in split_units(ctx) if u.kind == "sentence"]
    assert join_units(ctx, [us[0], us[2]]) == "A one. C three."


def test_split_selection_deterministic_and_disjoint():
    ids = [f"s{i:03d}" for i in range(200)]
    strata = {s: ("bridge" if i % 5 else "comparison") for i, s in enumerate(ids)}
    a = select_splits(ids, strata, 20, 80, seed=7)
    b = select_splits(list(reversed(ids)), strata, 20, 80, seed=7)
    assert a == b and len(a["dev"]) == 20 and len(a["eval"]) == 80
    assert not set(a["dev"]) & set(a["eval"])
    assert Counter(strata[s] for s in a["eval"]) == Counter({"bridge": 64, "comparison": 16})


def test_annotations_never_reference_outputs(anns):
    a = next(iter(anns.values()))
    assert set(a) >= {"facts", "bridges", "headers", "manual"}
    assert a["manual"]["status"] == "pending"
