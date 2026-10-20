"""Real-model dev pilot (S = Qwen2.5-0.5B-Instruct, L = Qwen2.5-7B-Instruct; see configs/real_dev_v1.json).

Phases (each writes under runs/real/<run_id>/):
  1 compress  : specpc_pilot (draft attention, official selection on GPU tensor) + sent_bm25 + full
                -> artifacts/compress.jsonl   (one artifact per item x method; reused by S and L)
  2 diag-build: diagnostic inputs from saved specpc artifacts + AI annotations (no answers exist yet)
                -> artifacts/diag.jsonl
  3 answer S  : S-full, S-compressed(specpc)           -> results/main.jsonl
  4 answer L  : L-full, L-compressed(specpc), L-compressed(bm25) -> results/main.jsonl
  5 diag L    : L on diag_proj / diag_restore / diag_control -> results/diag.jsonl
No mock component is importable from here (tests/mocks is never on sys.path); backend is "hf".
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccpilot.compressors.base import FullContext  # noqa: E402
from ccpilot.compressors.sentence_select import BM25Select  # noqa: E402
from ccpilot.compressors.specpc import SpecPCParams, SpecPCPilot, load_official_select  # noqa: E402
from ccpilot.data import join_items, load_hotpot_dev, load_longbench  # noqa: E402
from ccpilot.diagnostic import build_conditions  # noqa: E402
from ccpilot.manual import diag_eligible, evaluate_manual, load_manual  # noqa: E402
from ccpilot.prompt import HFTokenizer, PromptSpec  # noqa: E402
from ccpilot.real import answer, ids_sha, make_artifact, write_jsonl  # noqa: E402
from ccpilot.records import validate  # noqa: E402
from ccpilot.server import ServerProcess  # noqa: E402

TOK = ROOT / "data/raw/tokenizers/Qwen2.5-14B-Instruct/tokenizer.json"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/real_dev_v1.json"))
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--ids", default="dev", help="'dev' | 'smoke' (first 3 dev ids, fixed)")
    ap.add_argument("--phases", default="1,2,3,4,5")
    a = ap.parse_args()
    if os.environ.get("CCPILOT_ALLOW_MODELS") != "1":
        sys.exit("set CCPILOT_ALLOW_MODELS=1")
    assert "mocks" not in sys.modules
    cfg = json.loads(Path(a.config).read_text())
    cfg_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    out = ROOT / "runs/real" / a.run_id
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(cfg, _config_id=cfg_id, _ids=a.ids), indent=1))
    phases = set(a.phases.split(","))

    tokref = cfg["tokenizer_ref"]
    tok = HFTokenizer(TOK, tokref)
    prompt = PromptSpec()
    items_all = join_items(load_longbench(ROOT / cfg["data"]["longbench_file"]),
                           load_hotpot_dev(ROOT / cfg["data"]["hotpot_file"]))
    dev = json.loads((ROOT / "data/processed/splits.json").read_text())["dev"]
    ids = dev[:3] if a.ids == "smoke" else dev
    items = [it for it in items_all if it["sample_id"] in set(ids)]
    items.sort(key=lambda it: ids.index(it["sample_id"]))
    auto = {x["sample_id"]: x for x in map(json.loads, (ROOT / "data/processed/annotations_auto.jsonl").read_text().splitlines())}
    manual = load_manual(ROOT / "data/annotations/manual_v1_dev.json", {it["sample_id"]: it["context"] for it in items_all})
    B = cfg["budget"]
    S, L = cfg["models"]["S"], cfg["models"]["L"]
    common = dict(tok=tok, prompt=prompt, run_id=a.run_id, cfg_id=cfg_id, split="dev",
                  data_revision=cfg["data"]["revision_tag"], tokenizer_ref=tokref)

    # ---- phase 1: compression artifacts
    art_path = out / "artifacts/compress.jsonl"
    if "1" in phases:
        import torch

        from ccpilot.adapters.specpc_draft import make_draft_score_fn
        p = SpecPCParams(**cfg["specpc_params"])
        sel, origin = load_official_select()
        assert origin == "import", origin
        log(f"phase1: loading draft {S['model']}@{S['revision'][:7]} on GPU {cfg['gpus']['draft']}")
        sfn = make_draft_score_fn(model_name=S["model"], revision=S["revision"], params=p, device="cuda:0")
        spc = SpecPCPilot(sfn, p, score_fn_id=f"draft:{S['model']}@{S['revision']}", selector=sel, selector_origin=origin)
        comps = [(FullContext(), "full"), (spc, f"specpc_pilot[draft={S['model']}@{S['revision'][:7]},select=official:{origin}]"),
                 (BM25Select(), "sent_bm25[k1=1.5,b=0.75]")]
        arts = []
        for it in items:
            for comp, cid in comps:
                art = make_artifact(it, comp, None if comp.method_id == "full" else B, tok, prompt, cid)
                arts.append(art)
            log(f"  compressed {it['sample_id'][:8]}")
        write_jsonl(art_path, arts)
        del sfn, spc
        gc.collect()
        torch.cuda.empty_cache()
        log(f"phase1 done: {len(arts)} artifacts, peak draft mem {torch.cuda.max_memory_allocated()/2**20:.0f} MB")
    arts = {(x["sample_id"], x["method"]): x for x in map(json.loads, art_path.read_text().splitlines())}

    # ---- phase 2: diagnostic inputs (before any answer)
    diag_path = out / "artifacts/diag.jsonl"
    if "2" in phases:
        rows = []
        for it in items:
            sid = it["sample_id"]
            ann = manual.get(sid)
            base = dict(sample_id=sid, method="specpc_pilot", budget=B)
            if ann is None or not diag_eligible(ann)[0]:
                rows.append(dict(base, applicable=False, reason="not diag-eligible: " + (diag_eligible(ann)[1] if ann else "no annotation")))
                continue
            art = arts[(sid, "specpc_pilot")]
            ev = evaluate_manual(it["context"], ann, art["kept_intervals"], tok.decode(art["context_ids"]))
            missing = [e["rid"] for e in ev["requirements"] if not e["intact"]]
            C = B - art["fixed_tokens"]
            d = build_conditions(it["context"], tok.encode(it["context"]), tok.offsets(it["context"]),
                                 art["kept_intervals"], ann, missing, C, it["answers"])
            row = dict(base, applicable=d["applicable"], reason=d.get("reason"), missing_rids=missing,
                       evidence_before=ev, **{k: v for k, v in d.items() if k not in ("applicable", "reason", "conditions")})
            if d["applicable"]:
                row["conditions"] = {k: dict(v, context_ids_sha=ids_sha(v["context_ids"])) for k, v in d["conditions"].items()}
            rows.append(row)
        diag_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        log(f"phase2 done: {sum(r['applicable'] for r in rows)} applicable / {len(rows)}")

    def server_for(m, gpu):
        log(f"loading answer model {m['model']}@{m['revision'][:7]} on GPU {gpu}")
        return ServerProcess("hf", dict(tokenizer_path=str(TOK), tokenizer_name=tokref, max_input_tokens=32736,
                                        gen_profile=cfg["gen_profile"], model=m["model"], revision=m["revision"],
                                        attn_implementation=cfg["attn_implementation"]),
                             cuda_visible_devices=str(gpu), python=sys.executable)

    def run_answers(role, m, gpu, plan, path):
        srv = server_for(m, gpu)
        log(f"  {role} loaded in {srv.t_load_s:.1f}s")
        try:
            for it in items:
                sid = it["sample_id"]
                for cond, art, label in plan(it):
                    rec = answer(it, art, condition=cond, server=srv, auto_ann=auto.get(sid), manual_ann=manual.get(sid),
                                 method_label=label, **common)
                    errs = validate(rec)
                    if errs:
                        rec["error"] = (rec["error"] or "") + f" | schema: {errs}"
                    write_jsonl(path, [rec])
                    log(f"  {cond:24s} {sid[:8]} n_in={rec['final_server_input_tokens']} f1={rec['qa_f1']} "
                        f"{'ERR '+rec['error'] if rec['error'] else ''}")
        finally:
            srv.close()

    main_path = out / "results/main.jsonl"
    if "3" in phases:
        run_answers("S", S, cfg["gpus"]["S"], lambda it: [
            ("S-full", arts[(it["sample_id"], "full")], "full"),
            ("S-compressed", arts[(it["sample_id"], "specpc_pilot")], "specpc_pilot")], main_path)
    if "4" in phases:
        run_answers("L", L, cfg["gpus"]["L"], lambda it: [
            ("L-full", arts[(it["sample_id"], "full")], "full"),
            ("L-compressed", arts[(it["sample_id"], "specpc_pilot")], "specpc_pilot"),
            ("L-compressed[bm25]", arts[(it["sample_id"], "sent_bm25")], "sent_bm25")], main_path)
    if "5" in phases and diag_path.exists():
        drows = {r["sample_id"]: r for r in map(json.loads, diag_path.read_text().splitlines())}

        def plan(it):
            d = drows.get(it["sample_id"])
            if not d or not d["applicable"]:
                return []
            base = arts[(it["sample_id"], "specpc_pilot")]
            return [(f"L-{k}", dict(base, artifact_id=f"{base['artifact_id']}#{k}", context_ids=v["context_ids"],
                                    context_ids_sha=v["context_ids_sha"], kept_intervals=[tuple(x) for x in v["kept_intervals"]],
                                    t_compress_s=0.0, t_fit_s=0.0, fit_iterations=0), k)
                    for k, v in d["conditions"].items()]
        run_answers("L", L, cfg["gpus"]["L"], plan, out / "results/diag.jsonl")
    log("done")


if __name__ == "__main__":
    main()
