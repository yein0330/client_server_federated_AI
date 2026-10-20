"""Print what an annotator needs to verify one item's required evidence (no model outputs).

usage: python scripts/annotate_view.py SAMPLE_ID [--grep TERM ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccpilot.data import join_items, load_hotpot_dev, load_longbench  # noqa: E402
from ccpilot.evidence import _f1  # noqa: E402
from ccpilot.textutil import parse_passages, split_units, word_tokens  # noqa: E402


def load():
    items = join_items(load_longbench(ROOT / "data/raw/longbench/data/hotpotqa.jsonl"),
                       load_hotpot_dev(ROOT / "data/raw/hotpotqa/distractor/validation-00000-of-00001.parquet"))
    anns = {x["sample_id"]: x for x in map(json.loads, (ROOT / "data/processed/annotations_auto.jsonl").read_text().splitlines())}
    return {it["sample_id"]: it for it in items}, anns


def show(it, ann, greps):
    ctx, h = it["context"], it["hotpot"]
    ps = {p["title"]: p for p in parse_passages(ctx)}
    units = split_units(ctx)
    print(f"=== {it['sample_id']}  type={h['type']} level={h['level']}")
    print(f"Q: {it['question']}\nA: {it['answers']}\npassages: {[p['title'] for p in parse_passages(ctx)]}")
    for f in ann["facts"]:
        print(f"\n-- {f['eid']} [{f['tier']} {f['score']}] {f['title']} #{f['sent_id']}")
        print(f"   HP : {f.get('source_sentence', '')}")
        print(f"   LB : {ctx[f['start']:f['end']].strip() if f['start'] is not None else '-'}")
        p = ps.get(f["title"])
        if p and f.get("source_sentence"):
            q = word_tokens(f["source_sentence"])
            cands = sorted(((_f1(q, word_tokens(u.text(ctx))), u) for u in units
                            if u.kind == "sentence" and p["start"] <= u.start < p["end"]), key=lambda x: -x[0])[:3]
            for sc, u in cands:
                if f["start"] is None or not (u.start >= f["start"] and u.end <= f["end"]):
                    print(f"   alt {sc:.2f}: {u.text(ctx).strip()[:300]}")
    terms = list(greps) + [a for a in it["answers"] if a.lower() not in ("yes", "no")]
    for t in terms:
        hits = [m.start() for m in re.finditer(re.escape(t), ctx, re.I)]
        print(f"\n[grep '{t}'] {len(hits)} hits")
        for s in hits[:6]:
            pt = next((p["title"] for p in ps.values() if p["start"] <= s < p["end"]), "?")
            print(f"   <{pt}> ...{ctx[max(0, s - 140):s + 140].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--grep", nargs="*", default=[])
    a = ap.parse_args()
    items, anns = load()
    for sid in a.ids:
        show(items[sid], anns[sid], a.grep)
        print()
