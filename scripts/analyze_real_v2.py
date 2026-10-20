"""Analysis for real-dev-v2 (blinded AI semantic labels; v1 re-labelled blind for the 32 vs 128 comparison)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccpilot.analysis import paired_bootstrap  # noqa: E402

R2, R1 = ROOT / "runs/real/real-dev-v2", ROOT / "runs/real/real-dev-v1"
rows = {f: [json.loads(x) for x in (R2 / f"results/{f}.jsonl").read_text().splitlines()] for f in ("main", "del", "ctx")}
key = json.loads((R2 / "blind_review_key.json").read_text())
lab = json.loads((R2 / "blind_review_ai.json").read_text())["labels"]
for f in rows:
    for r in rows[f]:
        r["sem"], r["sem_note"] = ("correct", "em") if r["em"] else (None, None)
text2lab = {}
for b, ref in key.items():
    f, i = ref.split(":")
    r = rows[f][int(i)]
    r["sem"], r["sem_note"] = lab[b]
    text2lab[(r["sample_id"], r["prediction_raw"])] = lab[b]
v1 = [json.loads(x) for x in (R1 / "results/main.jsonl").read_text().splitlines()]
v1lab = {v["v1_row"]: (v["label"], v["note"]) for v in json.loads((R2 / "blind_review_v1_ai.json").read_text())["labels"].values()}
for i, r in enumerate(v1):
    r["sem"], r["sem_note"] = ("correct", "em") if r["em"] else (v1lab.get(i) or text2lab[(r["sample_id"], r["prediction_raw"])])
assert all(r["sem"] for f in rows for r in rows[f]) and all(r["sem"] for r in v1)

support = {a["sample_id"]: a["support"] for a in json.loads((ROOT / "data/annotations/manual_v1_dev.json").read_text())["items"]}
by2, by1 = defaultdict(dict), defaultdict(dict)
for f in rows:
    for r in rows[f]:
        by2[r["sample_id"]][r["condition"]] = r
for r in v1:
    by1[r["sample_id"]][r["condition"]] = r
sids = sorted(by2)
ok = lambda r: r["sem"] == "correct"
M = {"correct": "O", "incorrect": "X", "ambiguous": "?"}
rep = dict(run="real-dev-v2", n_rows={f: len(v) for f, v in rows.items()},
           duplicates=sum(v - 1 for v in Counter((r["sample_id"], r["condition"]) for f in rows for r in rows[f]).values()),
           errors=sum(1 for f in rows for r in rows[f] if r["error"]),
           hit_max_v2=sum(1 for f in rows for r in rows[f] if r["hit_max_new_tokens"]))

def summ(rs):
    return dict(n=len(rs), f1=round(sum(r["qa_f1"] for r in rs) / len(rs), 3), em=round(sum(r["em"] for r in rs) / len(rs), 3),
                sem_correct=sum(r["sem"] == "correct" for r in rs), sem_ambiguous=sum(r["sem"] == "ambiguous" for r in rs),
                hit_max=sum(bool(r["hit_max_new_tokens"]) for r in rs))

C4 = ["S-full", "L-full", "S-compressed", "L-compressed"]
rep["conditions_v2_128"] = {c: summ([by2[s][c] for s in sids]) for c in C4 + ["L-question-only"]}
rep["conditions_v1_32_blind_relabel"] = {c: summ([by1[s][c] for s in sids]) for c in C4}
# 1. 32 vs 128
chg = []
for s in sids:
    for c in C4:
        a, b = by1[s][c], by2[s][c]
        same_input = a["context_ids_sha"] == b["context_ids_sha"] and a["server_input_sha"] == b["server_input_sha"]
        if a["sem"] != b["sem"] or a["prediction_raw"] != b["prediction_raw"]:
            chg.append(dict(item=s[:8], cond=c, same_input=same_input, v1=a["sem"], v1_hitmax=a["hit_max_new_tokens"],
                            v2=b["sem"], v2_hitmax=b["hit_max_new_tokens"], v1_ans=a["prediction_raw"][:90], v2_ans=b["prediction_raw"][:200]))
rep["v1_vs_v2_changed"] = chg
rep["inputs_identical_v1_v2"] = sum(by1[s][c]["server_input_sha"] == by2[s][c]["server_input_sha"] for s in sids for c in C4)
rep["persisting_compression_failures_128"] = {
    m: dict(full_ok_comp_not=[s[:8] for s in sids if ok(by2[s][f"{m}-full"]) and not ok(by2[s][f"{m}-compressed"])],
            full_not_comp_ok=[s[:8] for s in sids if not ok(by2[s][f"{m}-full"]) and ok(by2[s][f"{m}-compressed"])])
    for m in ("S", "L")}
pat = Counter("".join("1" if ok(by2[s][c]) else "0" for c in C4) for s in sids)
rep["four_path_patterns_128 (SF LF SC LC)"] = dict(pat.most_common())
# 2. question-only
qo = []
for s in sids:
    d = by2[s]
    qo.append(dict(item=s[:8], support=support[s], q_only=d["L-question-only"]["sem"], full=d["L-full"]["sem"], comp=d["L-compressed"]["sem"],
                   q_only_ans=d["L-question-only"]["prediction_raw"][:80]))
rep["question_only"] = qo
rep["question_only_counts"] = dict(
    q_only_correct=[x["item"] for x in qo if x["q_only"] == "correct"],
    q_only_wrong_comp_correct=[x["item"] for x in qo if x["q_only"] != "correct" and x["comp"] == "correct"],
    q_only_wrong_full_correct=[x["item"] for x in qo if x["q_only"] != "correct" and x["full"] == "correct"],
    comp_correct_only=[x["item"] for x in qo if x["comp"] == "correct" and x["q_only"] != "correct" and x["full"] != "correct"],
    q_only_correct_but_comp_wrong=[x["item"] for x in qo if x["q_only"] == "correct" and x["comp"] != "correct"])
# 3. deletion
dl = []
for s in sids:
    d = by2[s]
    if "L-evidence-delete" not in d:
        continue
    e, c, b = d["L-evidence-delete"], d["L-control-delete"], d["L-compressed"]
    dl.append(dict(item=s[:8], base=b["sem"], ev_del=e["sem"], ctrl_del=c["sem"], f1_base=round(b["qa_f1"], 3),
                   f1_ev=round(e["qa_f1"], 3), f1_ctrl=round(c["qa_f1"], 3), n_removed=e["edit"]["removed_tokens"],
                   n_in=(e["final_server_input_tokens"], c["final_server_input_tokens"]),
                   gold_still_visible_after_ev_del=bool(e["edit"]["residual"]["gold_string_still_visible"]),
                   ev_ans=e["prediction_raw"][:80], ctrl_ans=c["prediction_raw"][:80]))
rep["deletion"] = dl
rep["deletion_summary"] = dict(
    n=len(dl), base_correct=sum(x["base"] == "correct" for x in dl), ev_correct=sum(x["ev_del"] == "correct" for x in dl),
    ctrl_correct=sum(x["ctrl_del"] == "correct" for x in dl),
    transitions={k: dict(Counter(f"base_{'ok' if x['base']=='correct' else 'no'}->{'ok' if x[k]=='correct' else 'no'}" for x in dl)) for k in ("ev_del", "ctrl_del")},
    ev_drop_only=[x["item"] for x in dl if x["base"] == "correct" and x["ev_del"] != "correct" and x["ctrl_del"] == "correct"],
    ctrl_drop_only=[x["item"] for x in dl if x["base"] == "correct" and x["ctrl_del"] != "correct" and x["ev_del"] == "correct"],
    both_drop=[x["item"] for x in dl if x["base"] == "correct" and x["ev_del"] != "correct" and x["ctrl_del"] != "correct"],
    paired_correct_ev_minus_ctrl=paired_bootstrap([float(x["ev_del"] == "correct") - float(x["ctrl_del"] == "correct") for x in dl]),
    paired_f1_ev_minus_base=paired_bootstrap([x["f1_ev"] - x["f1_base"] for x in dl]),
    paired_f1_ctrl_minus_base=paired_bootstrap([x["f1_ctrl"] - x["f1_base"] for x in dl]),
    ev_correct_with_gold_visible=sum(x["ev_del"] == "correct" and x["gold_still_visible_after_ev_del"] for x in dl),
    ev_correct_without_gold_visible=sum(x["ev_del"] == "correct" and not x["gold_still_visible_after_ev_del"] for x in dl),
    equal_lengths=all(x["n_in"][0] == x["n_in"][1] for x in dl))
# 4. ctx
rep["ctx"] = [dict(item=r["sample_id"][:8], cond=r["condition"], sem=r["sem"], ans=r["prediction_raw"], n_in=r["final_server_input_tokens"],
                   added=r["edit"]["added_tokens"], removed=r["edit"]["removed_tokens"]) for r in rows["ctx"]]
(R2 / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))

L = ["| item | support | S-full 32→128 | L-full 32→128 | S-comp 32→128 | L-comp 32→128 | L-q-only | L-ev-del | L-ctrl-del |", "|---|---|---|---|---|---|---|---|---|"]
for s in sids:
    d1, d2 = by1[s], by2[s]
    cell = lambda c: f"{M[d1[c]['sem']]}{'ᵗ' if d1[c]['hit_max_new_tokens'] else ''}→{M[d2[c]['sem']]}{'ᵗ' if d2[c]['hit_max_new_tokens'] else ''}"
    g = lambda c: M[d2[c]["sem"]] if c in d2 else "–"
    L.append(f"| {s[:8]} | {support[s]} | " + " | ".join(cell(c) for c in C4) + f" | {g('L-question-only')} | {g('L-evidence-delete')} | {g('L-control-delete')} |")
(R2 / "item_matrix.md").write_text("\n".join(L) + "\n\nO correct, X incorrect, ? ambiguous (blind AI judgement, not human). ᵗ = hit max_new_tokens. – = not applicable.\n")
print(json.dumps(rep, indent=1, ensure_ascii=False)[:12000])
