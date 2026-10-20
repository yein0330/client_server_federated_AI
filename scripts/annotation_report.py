"""Summarise manual annotations: per-item required-evidence counts, support, eligibility."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccpilot.data import load_longbench  # noqa: E402
from ccpilot.manual import diag_eligible, load_manual, required  # noqa: E402
from ccpilot.textutil import normalize_answer  # noqa: E402

lb = {r["_id"]: r for r in load_longbench(ROOT / "data/raw/longbench/data/hotpotqa.jsonl")}
ctx = {k: v["context"] for k, v in lb.items()}
out = {}
for name in ("dev", "diag"):
    m = load_manual(ROOT / f"data/annotations/manual_v1_{name}.json", ctx)
    rows = []
    for sid, a in m.items():
        req = required(a)
        el, why = diag_eligible(a)
        ans = lb[sid]["answers"]
        yesno = all(normalize_answer(x) in ("yes", "no") for x in ans)
        present = yesno or any(normalize_answer(x) in normalize_answer(ctx[sid]) for x in ans)
        rows.append(dict(sample_id=sid, support=a["support"], n_required=len(req),
                         n_non_required=len(a["requirements"]) - len(req),
                         n_alts=[len(r["alts"]) for r in req],
                         verdicts=dict(Counter(r["verdict"] for r in req)),
                         answer_string_in_context=("n/a(yes/no)" if yesno else present),
                         eligible=el, exclude_reason=why))
    out[name] = dict(n_items=len(rows), support=dict(Counter(r["support"] for r in rows)),
                     n_eligible=sum(r["eligible"] for r in rows),
                     n_required_total=sum(r["n_required"] for r in rows),
                     required_per_item=dict(Counter(r["n_required"] for r in rows)),
                     verdicts=dict(sum((Counter(r["verdicts"]) for r in rows), Counter())),
                     answer_string_absent=sum(1 for r in rows if r["answer_string_in_context"] is False),
                     items=rows)
(ROOT / "data/annotations/annotation_report.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "items"})
    for r in v["items"]:
        print("  ", r["sample_id"][:8], r["support"], "req", r["n_required"], "alts", r["n_alts"], r["verdicts"],
              "ans_str", r["answer_string_in_context"], "eligible" if r["eligible"] else "EXCL:" + r["exclude_reason"])
