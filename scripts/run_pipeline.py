"""Run the pilot grid. In this stage only --backend mock is permitted.

Mock mode: real data, real server tokenizer, real BM25/uniform selection and budget fitting,
real payload/process boundary; SpecPC scores and LLMLingua-2 compression are MOCKED and
the server answers with a fixed heuristic. Outputs validate plumbing only - not methods.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ccpilot.analysis import failure_category, summarize  # noqa: E402
from ccpilot.compressors.base import FullContext  # noqa: E402
from ccpilot.compressors.llmlingua2 import LLMLingua2  # noqa: E402
from ccpilot.compressors.sentence_select import BM25Select, UniformSelect  # noqa: E402
from ccpilot.compressors.specpc import SpecPCPilot, SpecPCRepro  # noqa: E402
from ccpilot.data import join_items, load_hotpot_dev, load_longbench  # noqa: E402
from ccpilot.pipeline import config_id, run  # noqa: E402
from ccpilot.prompt import HFTokenizer, PromptSpec  # noqa: E402
from ccpilot.records import read_jsonl  # noqa: E402
from ccpilot.server import ServerProcess  # noqa: E402

TOK = ROOT / "data/raw/tokenizers/Qwen2.5-14B-Instruct/tokenizer.json"
TOKREF = "Qwen/Qwen2.5-14B-Instruct@cf98f3b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/pilot_v0.json"))
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--backend", default="mock", choices=["mock", "hf"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run_id", default=None)
    a = ap.parse_args()
    if a.backend != "mock" or os.environ.get("CCPILOT_ALLOW_MODELS") == "1":
        sys.exit("GPU/model runs are not authorised in this stage.")
    if a.split == "eval" and not json.loads(Path(a.config).read_text()).get("frozen"):
        sys.exit("eval split requires a frozen config (frozen=true) - finish dev first.")
    cfg = json.loads(Path(a.config).read_text())
    run_id = a.run_id or f"{a.backend}-{a.split}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(dict(cfg, _config_id=config_id(cfg), _backend=a.backend), indent=1))

    tok = HFTokenizer(TOK, TOKREF)
    items = join_items(load_longbench(ROOT / cfg["data"]["longbench_file"]),
                       load_hotpot_dev(ROOT / cfg["data"]["hotpot_file"]))
    ids = set(json.loads((ROOT / "data/processed/splits.json").read_text())[a.split])
    items = [it for it in items if it["sample_id"] in ids][: a.limit]
    anns = {x["sample_id"]: x for x in map(json.loads, (ROOT / "data/processed/annotations_auto.jsonl").read_text().splitlines())}

    from mocks import lexical_score_fn, mock_llmlingua_fn
    sfn = lexical_score_fn(tok)
    from ccpilot.compressors.specpc import load_official_select
    sel, origin = load_official_select(allow_source_extract=True)  # mock env: source_extract
    comps = [FullContext(), BM25Select(), UniformSelect(), LLMLingua2(compress_fn=mock_llmlingua_fn),
             SpecPCPilot(sfn, score_fn_id="MOCK-lexical", selector=sel, selector_origin=origin),
             SpecPCRepro(sfn, score_fn_id="MOCK-lexical", selector=sel, selector_origin=origin)]
    sargs = dict(tokenizer_path=str(TOK), tokenizer_name=TOKREF, max_input_tokens=cfg["server"]["max_input_tokens"])
    servers = {"pilot": ServerProcess("mock", dict(sargs, gen_profile="pilot_common")),
               "official_repro": ServerProcess("mock", dict(sargs, gen_profile="official_repro"))}
    srv = servers["pilot"]
    t0 = time.perf_counter()
    try:
        run(items, anns, comps, cfg["budgets"], tok=tok, prompt=PromptSpec(), servers=servers, run_id=run_id, cfg=cfg,
            split=a.split, data_revision=cfg["data"]["revision_tag"], tokenizer_ref=TOKREF, out_dir=str(out))
    finally:
        for s_ in servers.values():
            s_.close()
    recs = read_jsonl(out / "results.jsonl")
    rows = summarize(recs)
    fc = {}
    for r in recs:
        c = failure_category(r)
        if c:
            fc.setdefault(r["method"], {}).setdefault(c, 0)
            fc[r["method"]][c] += 1
    timing = {}
    for r in recs:
        timing.setdefault(r["method"], []).append(r["t_compress_s"] + r["t_fit_s"])
    summary = dict(run_id=run_id, backend=a.backend, n_items=len(items), wall_s=time.perf_counter() - t0,
                   server_load_s=srv.t_load_s, NOTE="MOCK backend: scores/answers are not method results",
                   cells=rows, failure_prelabels=fc,
                   client_time_s={m: dict(mean=sum(v) / len(v), max=max(v)) for m, v in timing.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for x in rows:
        print(f"{x['method']:14s} B={str(x['budget']):5s} n={x['n']:3d} budget_ok={x['budget_ok_rate']:.2f} "
              f"len p50/max={x['input_len']['p50']}/{x['input_len']['max']} bytes p50={x['payload_bytes']['p50']} "
              f"facts_kept={x['fact_preserved_rate'] if x['fact_preserved_rate'] is None else round(x['fact_preserved_rate'], 3)}")
    print("client compress+fit time (s):", {m: round(v['mean'], 3) for m, v in summary['client_time_s'].items()})
    print("out:", out)


if __name__ == "__main__":
    main()
