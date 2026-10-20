import json
import re

import pytest
from conftest import ROOT, server_args
from mocks import lexical_score_fn, mock_llmlingua_fn

from ccpilot import payload as P
from ccpilot.analysis import failure_category, summarize
from ccpilot.compressors.base import CompressRequest, FullContext
from ccpilot.compressors.llmlingua2 import LLMLingua2
from ccpilot.compressors.sentence_select import BM25Select, UniformSelect
from ccpilot.compressors.specpc import SpecPCPilot, SpecPCRepro, load_official_select
from ccpilot.evidence import score_preservation
from ccpilot.pipeline import run
from ccpilot.prompt import PromptSpec
from ccpilot.records import read_jsonl, validate
from ccpilot.server import ServerProcess

TOKREF = "Qwen/Qwen2.5-14B-Instruct@cf98f3b"


def _hdr(tok, **kw):
    h = dict(v=1, assembly="server_template", prompt_spec=PromptSpec().spec_id, tokenizer=TOKREF,
             tok_sha=tok.sha, req="0123456789abcdef", question="Who?")
    h.update(kw)
    return h


def test_payload_roundtrip_and_whitelist(tok):
    ids = tok.encode("Passage 1:\nA\nHello world.")
    buf = P.encode(_hdr(tok), ids)
    h, out = P.decode(buf, vocab_size=tok.vocab_size, max_ctx=100)
    assert out == ids and h["n_ctx"] == len(ids)
    assert len(buf) == 8 + len(json.dumps(h, sort_keys=True, separators=(",", ":")).encode()) + 4 * len(ids)
    for bad in (dict(sample_id="x"), dict(selected_idx=[1]), dict(method="bm25")):
        with pytest.raises(P.BoundaryError):
            P.encode(_hdr(tok, **bad), ids)
    with pytest.raises(P.BoundaryError):
        P.decode(buf[:-4], vocab_size=tok.vocab_size, max_ctx=100)
    with pytest.raises(P.BoundaryError):
        P.decode(P.encode(_hdr(tok), [tok.vocab_size + 5]), vocab_size=tok.vocab_size, max_ctx=100)
    with pytest.raises(P.BoundaryError):
        P.decode(buf, vocab_size=tok.vocab_size, max_ctx=2)


@pytest.mark.parametrize("B", [1024, 2048])
def test_sentence_methods_meet_budget_on_real_items(tok, lb_items, B):
    prompt = PromptSpec()
    for r in lb_items[:6]:
        fixed = prompt.fixed_len(tok, r["input"])
        for comp in (BM25Select(), UniformSelect()):
            req = CompressRequest(r["context"], r["input"] if comp.uses_question else None, B - fixed, tok, prompt)
            res = comp.compress(req)
            n = len(prompt.assemble(tok, res.context_ids, r["input"]))
            assert n <= B, (comp.method_id, n)
            assert B - n < 200, f"{comp.method_id} under-fills by {B - n}"  # dev-time sanity, not a hard rule


def test_question_unaware_methods_never_get_question(tok, lb_items):
    seen = []

    def spy(context, target):
        seen.append(context)
        return mock_llmlingua_fn(context, target)
    r = lb_items[0]
    comp = LLMLingua2(compress_fn=spy)
    assert comp.uses_question is False
    req = CompressRequest(r["context"], None, 900, tok, PromptSpec())
    res = comp.compress(req)
    assert all(r["input"] not in c for c in seen) or r["input"] in r["context"]
    assert len(res.context_ids) <= 900 and res.kept_intervals


CANARY = "ZQXCANARY"


def _canary_item(tok):
    passages = []
    for k in range(1, 41):
        body = " ".join(f"Filler sentence {k}-{j} about unrelated topic number {j}." for j in range(12))
        if k % 4 == 0:
            body += f" The secret code is {CANARY}{k}."
        passages.append(f"Passage {k}:\nTopic {k}\n{body}")
    return dict(sample_id="canary0", question="What is topic number 3 about?", context="\n".join(passages),
                answers=["x"], hotpot=None)


def test_process_boundary_canary(tok, tmp_path):
    """Dropped context must not reach the server process; server sees payload bytes only."""
    item = _canary_item(tok)
    srv = ServerProcess("mock", server_args(debug_return_visible_text=True))  # pilot_common
    try:
        prompt = PromptSpec()
        for comp, should_see in ((FullContext(), True), (UniformSelect(), None), (BM25Select(), None),
                                 (SpecPCPilot(lexical_score_fn(tok), score_fn_id="mock"), None)):
            ctx_budget = 600 - prompt.fixed_len(tok, item["question"])
            res = comp.compress(CompressRequest(item["context"], item["question"] if comp.uses_question else None,
                                                ctx_budget, tok, prompt))
            buf = P.encode(_hdr(tok, question=item["question"]), res.context_ids)
            assert CANARY.encode() not in buf or should_see  # raw ids, but also no text leak in header
            out = srv(buf)
            assert out["error"] is None
            vis = out["visible_text"]
            sent = tok.decode(res.context_ids)
            assert set(re.findall(CANARY + r"\d+", vis)) == set(re.findall(CANARY + r"\d+", sent))
            if should_see:
                assert len(re.findall(CANARY + r"\d+", vis)) == 10
        bad = srv(P.encode(_hdr(tok, tok_sha="ffffffffffffffff"), [1, 2]))
        assert "tokenizer mismatch" in bad["error"]
    finally:
        srv.close()


def test_server_worker_does_not_import_dataset_modules():
    src = (ROOT / "ccpilot/server_worker.py").read_text()
    for dep in ("server_worker.py", "payload.py", "prompt.py", "textutil.py"):
        s = (ROOT / "ccpilot" / dep).read_text()
        assert not re.search(r"^\s*(from|import)\s+(ccpilot\.)?(data|evidence|pipeline|diagnostic)\b", s, re.M), dep
    assert "ccpilot.data" not in src


def test_end_to_end_mock_run(tok, joined, anns, tmp_path):
    splits = json.loads((ROOT / "data/processed/splits.json").read_text())
    dev = [it for it in joined if it["sample_id"] in set(splits["dev"])][:3]
    servers = {"pilot": ServerProcess("mock", server_args("pilot_common")),
               "official_repro": ServerProcess("mock", server_args("official_repro"))}
    sfn = lexical_score_fn(tok)
    sel, origin = load_official_select(allow_source_extract=True)
    comps = [FullContext(), BM25Select(), UniformSelect(), LLMLingua2(compress_fn=mock_llmlingua_fn),
             SpecPCPilot(sfn, score_fn_id="mock-lexical", selector=sel, selector_origin=origin),
             SpecPCRepro(sfn, score_fn_id="mock-lexical", selector=sel, selector_origin=origin)]
    try:
        run(dev, anns, comps, [1024, 2048], tok=tok, prompt=PromptSpec(), servers=servers, run_id="mock-test",
            cfg={"mock": True}, split="dev", data_revision="lb-5e628be/a0005ab2", tokenizer_ref=TOKREF,
            out_dir=str(tmp_path))
    finally:
        for s_ in servers.values():
            s_.close()
    recs = read_jsonl(tmp_path / "results.jsonl")
    prov = read_jsonl(tmp_path / "provenance.jsonl")
    assert len(recs) == len(prov) == 3 * (1 + 5 * 2)
    for r in recs:
        assert validate(r) == [], validate(r)
        assert r["error"] is None, r["error"]
        assert r["budget_ok"], (r["method"], r["final_server_input_tokens"], r["budget_nominal"])
        assert "kept_intervals" not in r
        assert r["gen_profile"] == ("official_repro" if r["method"] == "specpc_repro" else "pilot_common")
        if r["method"] == "specpc_repro":
            assert r["final_server_input_tokens"] == r["budget_nominal"]
    for p_ in prov:
        if p_["method"].startswith("specpc"):
            assert p_["info"]["selector"] == origin
    rows = summarize(recs)
    assert {(x["method"], x["budget"]) for x in rows} >= {("full", None), ("sent_bm25", 1024)}
    for r in recs:
        failure_category(r)


def test_repro_uses_joint_tokenization_and_official_window(tok, lb_items):
    sel, origin = load_official_select(allow_source_extract=True)
    r = lb_items[0]
    comp = SpecPCRepro(lexical_score_fn(tok), selector=sel, selector_origin=origin)
    comp.total_budget = 1024
    res = comp.compress(CompressRequest(r["context"], r["input"], 1024, tok, PromptSpec()))
    p = PromptSpec()
    joint = tok.encode(p.prefix_text() + r["context"] + p.suffix_text(r["input"]))
    assert res.info["n_total"] == len(joint) and len(res.info["raw_ids"]) == 1024
    assert res.info["raw_ids"][-64:] == joint[-64:]  # official tail window kept


def test_restoration_diagnostic_rules(tok, joined, manual_dev):
    from ccpilot.diagnostic import build_conditions
    from ccpilot.manual import diag_eligible, evaluate_manual, score_requirements

    prompt = PromptSpec()
    items = {i["sample_id"]: i for i in joined}
    n_app = 0
    for sid, a in manual_dev.items():
        if not diag_eligible(a)[0]:
            continue
        it = items[sid]
        C = 1024 - prompt.fixed_len(tok, it["question"])
        res = UniformSelect().compress(CompressRequest(it["context"], None, C, tok, prompt))
        ev = evaluate_manual(it["context"], a, res.kept_intervals, tok.decode(res.context_ids))
        missing = [e["rid"] for e in ev["requirements"] if not e["intact"]]
        out = build_conditions(it["context"], tok.encode(it["context"]), tok.offsets(it["context"]),
                               res.kept_intervals, a, missing, C, it["answers"])
        if not out["applicable"]:
            continue
        n_app += 1
        c = out["conditions"]
        assert c["diag_restore"]["n_tokens"] == c["diag_control"]["n_tokens"] <= C
        for name in ("diag_restore", "diag_control"):
            assert len(prompt.assemble(tok, c[name]["context_ids"], it["question"])) <= 1024
        after = score_requirements(it["context"], a, c["diag_restore"]["kept_intervals"])
        assert all(e["status"] == "preserved" for e in after["evidence"] if e["rid"] in missing), sid
        before = score_requirements(it["context"], a, res.kept_intervals)
        ctrl = score_requirements(it["context"], a, c["diag_control"]["kept_intervals"])
        assert ctrl["n_preserved"] >= before["n_preserved"]  # common removal never touches required quotes
    assert n_app >= 5
