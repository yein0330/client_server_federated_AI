"""real-dev-v2: follow-up diagnostics reusing real-dev-v1 compression artifacts unchanged.

  A  S-full, L-full, S-compressed, L-compressed with max_new_tokens=128 (profile pilot_common_128;
     everything else identical to v1)                                  -> results/main.jsonl
  B  L-question-only (question_only_spec, empty context), same profile  -> results/main.jsonl
  C  L-evidence-delete / L-control-delete (DEL-v1) on supported items  -> results/del.jsonl
  D  L-context-restore / L-context-control (CTX-v1) on items selected by a fixed mechanical rule
     from A: manual_v1 support=supported, all required evidence intact in the base compression,
     L-full (em or contains) and L-compressed (not em and not contains)   -> results/ctx.jsonl
All inputs for C are built before any answer; D inputs only depend on the A outcome via the rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccpilot.data import join_items, load_hotpot_dev, load_longbench  # noqa: E402
from ccpilot.diag2 import RULES, build_context_restore, build_deletion, residual_check  # noqa: E402
from ccpilot.manual import evaluate_manual, load_manual  # noqa: E402
from ccpilot.prompt import HFTokenizer, PromptSpec, question_only_spec  # noqa: E402
from ccpilot.real import answer, ids_sha, write_jsonl  # noqa: E402
from ccpilot.records import validate  # noqa: E402
from ccpilot.server import ServerProcess  # noqa: E402

TOK = ROOT / "data/raw/tokenizers/Qwen2.5-14B-Instruct/tokenizer.json"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--base_run", default="real-dev-v1")
    a = ap.parse_args()
    assert os.environ.get("CCPILOT_ALLOW_MODELS") == "1"
    assert "mocks" not in sys.modules
    base_cfg = json.loads((ROOT / "configs/real_dev_v1.json").read_text())
    cfg = dict(base_cfg, pilot="ctxcomp-real-dev-v2", gen_profile="pilot_common_128",
               reuse_artifacts=f"runs/real/{a.base_run}/artifacts/compress.jsonl", rules=RULES,
               question_only_spec=question_only_spec().spec_id)
    cfg_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    out = ROOT / "runs/real" / a.run_id
    if out.exists():
        sys.exit(f"{out} exists - refusing to overwrite")
    (out / "artifacts").mkdir(parents=True)
    (out / "results").mkdir()
    (out / "config.json").write_text(json.dumps(dict(cfg, _config_id=cfg_id), indent=1))

    tok = HFTokenizer(TOK, cfg["tokenizer_ref"])
    P, Q = PromptSpec(), question_only_spec()
    items_all = join_items(load_longbench(ROOT / cfg["data"]["longbench_file"]), load_hotpot_dev(ROOT / cfg["data"]["hotpot_file"]))
    dev = json.loads((ROOT / "data/processed/splits.json").read_text())["dev"]
    items = sorted((it for it in items_all if it["sample_id"] in set(dev)), key=lambda it: dev.index(it["sample_id"]))
    auto = {x["sample_id"]: x for x in map(json.loads, (ROOT / "data/processed/annotations_auto.jsonl").read_text().splitlines())}
    manual = load_manual(ROOT / "data/annotations/manual_v1_dev.json", {it["sample_id"]: it["context"] for it in items_all})
    base_path = ROOT / cfg["reuse_artifacts"]
    arts = {(x["sample_id"], x["method"]): x for x in map(json.loads, base_path.read_text().splitlines())}
    (out / "artifacts/base_artifacts_sha256.txt").write_text(hashlib.sha256(base_path.read_bytes()).hexdigest() + "\n")
    B = cfg["budget"]
    common = dict(tok=tok, run_id=a.run_id, cfg_id=cfg_id, split="dev", data_revision=cfg["data"]["revision_tag"],
                  tokenizer_ref=cfg["tokenizer_ref"])

    # ---- inputs built before any answer: question-only + DEL-v1
    qart = {}
    for it in items:
        base = arts[(it["sample_id"], "full")]
        qart[it["sample_id"]] = dict(base, artifact_id=f"{it['sample_id']}#question-only#None", method="question_only",
                                     compressor_id="none(question only)", context_ids=[], context_ids_sha=ids_sha([]),
                                     kept_intervals=None, pre_fit_context_tokens=0, t_compress_s=0.0, t_fit_s=0.0,
                                     fixed_tokens=Q.fixed_len(tok, it["question"]))
    dels = []
    for it in items:
        sid = it["sample_id"]
        ann = manual.get(sid)
        row = dict(sample_id=sid, rule=RULES["DEL"])
        if ann is None or ann["support"] != "supported":
            dels.append(dict(row, applicable=False, reason=f"support={ann and ann['support']}"))
            continue
        base = arts[(sid, "specpc_pilot")]
        d = build_deletion(it["context"], tok.encode(it["context"]), tok.offsets(it["context"]),
                           base["info"]["selected_token_pos"], ann, it["answers"])
        row.update(applicable=d["applicable"], reason=d.get("reason"))
        if d["applicable"]:
            row.update(n_removed=d["n_removed"], n_runs=d["n_runs"], run_lengths=d["run_lengths"], conditions={})
            for k, v in d["conditions"].items():
                vis = tok.decode(v["context_ids"])
                row["conditions"][k] = dict(v, context_ids_sha=ids_sha(v["context_ids"]),
                                            residual=residual_check(ann, it["context"], vis, it["answers"]))
        dels.append(row)
    (out / "artifacts/del.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dels))
    log(f"DEL inputs: {sum(r['applicable'] for r in dels)} applicable / {len(dels)}")

    def server(m, gpu):
        log(f"loading {m['model']}@{m['revision'][:7]} on GPU {gpu}")
        return ServerProcess("hf", dict(tokenizer_path=str(TOK), tokenizer_name=cfg["tokenizer_ref"], max_input_tokens=32736,
                                        gen_profile=cfg["gen_profile"], model=m["model"], revision=m["revision"],
                                        attn_implementation=cfg["attn_implementation"]),
                             cuda_visible_devices=str(gpu), python=sys.executable)

    def go(srv, it, cond, art, path, prompt=P, label=None, edit=None, evidence=True):
        sid = it["sample_id"]
        rec = answer(it, art, condition=cond, server=srv, prompt=prompt, method_label=label,
                     auto_ann=auto.get(sid) if evidence else None, manual_ann=manual.get(sid) if evidence else None, **common)
        rec["edit"] = edit
        errs = validate(rec)
        if errs:
            rec["error"] = (rec["error"] or "") + f" | schema: {errs}"
        write_jsonl(path, [rec])
        log(f"  {cond:22s} {sid[:8]} n_in={rec['final_server_input_tokens']} out={rec['n_out_tokens']} "
            f"hitmax={rec['hit_max_new_tokens']} {'ERR ' + rec['error'] if rec['error'] else ''}")
        return rec

    main_p = out / "results/main.jsonl"
    S, L = cfg["models"]["S"], cfg["models"]["L"]
    srv = server(S, cfg["gpus"]["S"])
    try:
        for it in items:
            go(srv, it, "S-full", arts[(it["sample_id"], "full")], main_p, label="full")
            go(srv, it, "S-compressed", arts[(it["sample_id"], "specpc_pilot")], main_p, label="specpc_pilot")
    finally:
        srv.close()
    srv = server(L, cfg["gpus"]["L"])
    try:
        Lres = {}
        for it in items:
            sid = it["sample_id"]
            Lres[(sid, "L-full")] = go(srv, it, "L-full", arts[(sid, "full")], main_p, label="full")
            Lres[(sid, "L-compressed")] = go(srv, it, "L-compressed", arts[(sid, "specpc_pilot")], main_p, label="specpc_pilot")
            go(srv, it, "L-question-only", qart[sid], main_p, prompt=Q, label="question_only", evidence=False)
        for row in dels:
            if not row["applicable"]:
                continue
            it = next(i for i in items if i["sample_id"] == row["sample_id"])
            base = arts[(row["sample_id"], "specpc_pilot")]
            for k, v in row["conditions"].items():
                art = dict(base, artifact_id=f"{base['artifact_id']}#{k}", context_ids=v["context_ids"],
                           context_ids_sha=v["context_ids_sha"], kept_intervals=[tuple(x) for x in v["kept_intervals"]],
                           t_compress_s=0.0, t_fit_s=0.0, fit_iterations=0)
                go(srv, it, f"L-{k}", art, out / "results/del.jsonl", label=k, edit=dict(v["edit"], residual=v["residual"]))
        # ---- D: targets by the fixed mechanical rule
        ctx_rows = []
        for it in items:
            sid = it["sample_id"]
            ann = manual.get(sid)
            base = arts[(sid, "specpc_pilot")]
            lf, lc = Lres[(sid, "L-full")], Lres[(sid, "L-compressed")]
            ev = evaluate_manual(it["context"], ann, base["kept_intervals"], tok.decode(base["context_ids"])) if ann else None
            is_target = (ann is not None and ann["support"] == "supported" and ev["all_intact"]
                         and (lf["em"] or lf["contains"]) and not (lc["em"] or lc["contains"]))
            if not is_target:
                continue
            d = build_context_restore(it["context"], tok.encode(it["context"]), tok.offsets(it["context"]),
                                      base["info"]["selected_token_pos"], ann, it["answers"], B - base["fixed_tokens"])
            row = dict(sample_id=sid, rule=RULES["CTX"], applicable=d["applicable"], reason=d.get("reason"),
                       related_passages=d.get("related_passages"), n_add=d.get("n_add"))
            if d["applicable"]:
                row["conditions"] = {k: dict(v, context_ids_sha=ids_sha(v["context_ids"])) for k, v in d["conditions"].items()}
            ctx_rows.append(row)
        (out / "artifacts/ctx.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ctx_rows))
        log(f"CTX targets: {[r['sample_id'][:8] for r in ctx_rows]}  applicable={sum(r['applicable'] for r in ctx_rows)}")
        for row in ctx_rows:
            if not row["applicable"]:
                continue
            it = next(i for i in items if i["sample_id"] == row["sample_id"])
            base = arts[(row["sample_id"], "specpc_pilot")]
            for k, v in row["conditions"].items():
                art = dict(base, artifact_id=f"{base['artifact_id']}#{k}", context_ids=v["context_ids"],
                           context_ids_sha=v["context_ids_sha"], kept_intervals=[tuple(x) for x in v["kept_intervals"]],
                           t_compress_s=0.0, t_fit_s=0.0, fit_iterations=0)
                go(srv, it, f"L-{k}", art, out / "results/ctx.jsonl", label=k, edit=v["edit"])
    finally:
        srv.close()
    log("done")


if __name__ == "__main__":
    main()
