"""Analysis of a real dev run: condition table, four-path patterns, transitions, cases, BM25 vs SpecPC,
cost, evidence (char retention / key terms), diagnostic denominators. Writes report.json + tables.md."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccpilot.analysis import paired_bootstrap  # noqa: E402

run = ROOT / "runs/real" / sys.argv[1]
rows = [json.loads(x) for x in (run / "results/main.jsonl").read_text().splitlines()]
rev = json.loads((run / "semantic_review_ai.json").read_text())["reviewed"]
for i, r in enumerate(rows):
    r["sem"] = "correct" if r["em"] else rev[str(i)][0]
    r["sem_note"] = "" if r["em"] else rev[str(i)][1]
manual = {a["sample_id"]: a for a in json.loads((ROOT / "data/annotations/manual_v1_dev.json").read_text())["items"]}
support = {k: v["support"] for k, v in manual.items()}
diag = [json.loads(x) for x in (run / "artifacts/diag.jsonl").read_text().splitlines()]
arts = {(a["sample_id"], a["method"]): a for a in map(json.loads, (run / "artifacts/compress.jsonl").read_text().splitlines())}

C = ["S-full", "L-full", "S-compressed", "L-compressed", "L-compressed[bm25]"]
by = defaultdict(dict)
for r in rows:
    by[r["sample_id"]][r["condition"]] = r
sids = sorted(by)
ok = lambda r, k: (r["em"] if k == "em" else r["sem"] == "correct")
q = lambda xs: dict(min=min(xs), p50=sorted(xs)[len(xs) // 2], max=max(xs)) if xs else None


def table(subset, label):
    out = []
    for c in C:
        rs = [by[s][c] for s in subset if c in by[s] and not by[s][c]["error"]]
        out.append(dict(condition=c, subset=label, n=len(rs), f1=sum(r["qa_f1"] for r in rs) / len(rs),
                        em=sum(r["em"] for r in rs) / len(rs), contains=sum(r["contains"] for r in rs) / len(rs),
                        sem_ai=sum(r["sem"] == "correct" for r in rs) / len(rs),
                        n_sem_ambiguous=sum(r["sem"] == "ambiguous" for r in rs),
                        hit_max=sum(bool(r["hit_max_new_tokens"]) for r in rs)))
    return out


supported = [s for s in sids if support.get(s) == "supported"]
rep = dict(run=sys.argv[1], n_items=len(sids), n_rows=len(rows), errors=sum(1 for r in rows if r["error"]),
           budget_violations=sum(1 for r in rows if not r["budget_ok"]),
           server_input_check_failed=sum(1 for r in rows if not r["server_input_check"]),
           conditions=table(sids, "all20") + table(supported, "supported17"))

# same compressed context for S and L
rep["shared_compression_check"] = dict(
    same_context_sha=sum(by[s]["S-compressed"]["context_ids_sha"] == by[s]["L-compressed"]["context_ids_sha"] for s in sids),
    same_server_input_len=sum(by[s]["S-compressed"]["final_server_input_tokens"] == by[s]["L-compressed"]["final_server_input_tokens"] for s in sids),
    n=len(sids))

# four-path patterns, transitions, cases
for k in ("em", "sem"):
    pat = Counter()
    for s in sids:
        d = by[s]
        pat["SF={} LF={} SC={} LC={}".format(*[int(ok(d[c], k)) for c in C[:4]])] += 1
    rep[f"patterns_{k}"] = dict(sorted(pat.items(), key=lambda kv: -kv[1]))
    tr = {}
    for m in ("S", "L"):
        t = Counter()
        for s in sids:
            t[f"full_{'ok' if ok(by[s][m+'-full'], k) else 'wrong'}->comp_{'ok' if ok(by[s][m+'-compressed'], k) else 'wrong'}"] += 1
        tr[m] = dict(t)
    t = Counter()
    for s in sids:
        t[f"specpc_{'ok' if ok(by[s]['L-compressed'], k) else 'wrong'}/bm25_{'ok' if ok(by[s]['L-compressed[bm25]'], k) else 'wrong'}"] += 1
    tr["L_specpc_vs_bm25"] = dict(t)
    rep[f"transitions_{k}"] = tr
    cases = {
        "1_Sfull_wrong_Lfull_ok_Lcomp_ok": [s for s in sids if not ok(by[s]["S-full"], k) and ok(by[s]["L-full"], k) and ok(by[s]["L-compressed"], k)],
        "2_Lfull_ok_Lcomp_wrong": [s for s in sids if ok(by[s]["L-full"], k) and not ok(by[s]["L-compressed"], k)],
        "3_Sfull_ok_Scomp_ok_Lfull_ok_Lcomp_wrong": [s for s in sids if ok(by[s]["S-full"], k) and ok(by[s]["S-compressed"], k) and ok(by[s]["L-full"], k) and not ok(by[s]["L-compressed"], k)],
        "4_Lfull_wrong_Lcomp_ok": [s for s in sids if not ok(by[s]["L-full"], k) and ok(by[s]["L-compressed"], k)],
        "x_Sfull_wrong_Scomp_ok": [s for s in sids if not ok(by[s]["S-full"], k) and ok(by[s]["S-compressed"], k)],
        "x_Scomp_ok_Lcomp_wrong": [s for s in sids if ok(by[s]["S-compressed"], k) and not ok(by[s]["L-compressed"], k)],
    }
    rep[f"cases_{k}"] = cases
    rep[f"paired_{k}"] = {
        "L-compressed minus L-full": paired_bootstrap([float(ok(by[s]["L-compressed"], k)) - float(ok(by[s]["L-full"], k)) for s in sids]),
        "S-compressed minus S-full": paired_bootstrap([float(ok(by[s]["S-compressed"], k)) - float(ok(by[s]["S-full"], k)) for s in sids]),
        "L-full minus S-full": paired_bootstrap([float(ok(by[s]["L-full"], k)) - float(ok(by[s]["S-full"], k)) for s in sids]),
        "L-compressed[specpc] minus L-compressed[bm25]": paired_bootstrap([float(ok(by[s]["L-compressed"], k)) - float(ok(by[s]["L-compressed[bm25]"], k)) for s in sids]),
    }
rep["paired_f1"] = {
    "L-compressed minus L-full": paired_bootstrap([by[s]["L-compressed"]["qa_f1"] - by[s]["L-full"]["qa_f1"] for s in sids]),
    "S-compressed minus S-full": paired_bootstrap([by[s]["S-compressed"]["qa_f1"] - by[s]["S-full"]["qa_f1"] for s in sids]),
    "L-compressed[specpc] minus L-compressed[bm25]": paired_bootstrap([by[s]["L-compressed"]["qa_f1"] - by[s]["L-compressed[bm25]"]["qa_f1"] for s in sids]),
}

# cost
rep["cost"] = {c: dict(input_tokens=q([by[s][c]["final_server_input_tokens"] for s in sids]),
                       payload_bytes=q([by[s][c]["payload_bytes"] for s in sids]),
                       t_server_s=q([round(by[s][c]["t_server_s"], 3) for s in sids]),
                       gpu_peak_mem_mb=q([round(by[s][c]["gpu_peak_mem_mb"]) for s in sids]),
                       t_compress_s=q([round(by[s][c]["t_compress_s"] + by[s][c]["t_fit_s"], 3) for s in sids]))
               for c in C}

# evidence per compressed artifact (AI-written annotation; char retention != semantic preservation)
ev = {}
for m, c in (("specpc_pilot", "L-compressed"), ("sent_bm25", "L-compressed[bm25]")):
    rs = [by[s][c] for s in sids if by[s][c]["evidence_manual_ai"]]
    ev[m] = dict(n_items=len(rs), n_required=sum(r["evidence_manual_ai"]["n_required"] for r in rs),
                 n_char_retained=sum(r["evidence_manual_ai"]["n_char_preserved"] for r in rs),
                 n_intact=sum(r["evidence_manual_ai"]["n_intact"] for r in rs),
                 items_all_intact=sum(r["evidence_manual_ai"]["all_intact"] for r in rs),
                 auto_alignment_char_items_all=sum(bool(r["evidence"] and r["evidence"]["all_preserved"]) for r in rs),
                 not_intact=[(r["sample_id"][:8], e["rid"], e["frac"], e["key_terms_missing"]) for r in rs
                             for e in r["evidence_manual_ai"]["requirements"] if not e["intact"]])
rep["evidence_manual_ai"] = ev
rep["diag"] = dict(Counter(d["reason"] or "applicable" for d in diag))
rep["hit_max_rows"] = [(r["condition"], r["sample_id"][:8]) for r in rows if r["hit_max_new_tokens"]]
rep["support"] = dict(Counter(support[s] for s in sids))
rep["item_support"] = {s[:8]: support[s] for s in sids}
(run / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))

# per-item matrix for the write-up
lines = ["| item | support | gold | S-full | L-full | S-comp | L-comp | L-bm25 |", "|---|---|---|---|---|---|---|---|"]
mark = {"correct": "O", "incorrect": "X", "ambiguous": "?"}
for s in sids:
    d = by[s]
    lines.append(f"| {s[:8]} | {support[s]} | {d['S-full']['answers'][0][:30]} | " +
                 " | ".join(f"{mark[d[c]['sem']]}{'' if d[c]['em'] else '*' if d[c]['sem']=='correct' else ''}" for c in C) + " |")
(run / "item_matrix.md").write_text("\n".join(lines) + "\n\nO correct, X incorrect, ? ambiguous (AI semantic review); * = correct semantically but EM=0\n")
print(json.dumps({k: v for k, v in rep.items() if k not in ("item_support",)}, indent=1, ensure_ascii=False)[:9000])
