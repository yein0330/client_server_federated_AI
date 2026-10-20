"""Build the frozen sample manifest, splits and automatic evidence annotations.

Model-free: uses only the LongBench v1 / HotpotQA files and the server tokenizer.json.
Selection uses only (sample_id, HotpotQA type) and a fixed seed - no model outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccpilot.data import (HOTPOT_REVISION, LONGBENCH_REVISION, join_items, load_hotpot_dev,  # noqa: E402
                          load_longbench, select_splits, sha256_file)
from ccpilot.evidence import align_item  # noqa: E402
from ccpilot.prompt import HFTokenizer, PromptSpec  # noqa: E402
from ccpilot.textutil import normalize_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longbench", default=str(ROOT / "data/raw/longbench/data/hotpotqa.jsonl"))
    ap.add_argument("--hotpot", default=str(ROOT / "data/raw/hotpotqa/distractor/validation-00000-of-00001.parquet"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/raw/tokenizers/Qwen2.5-14B-Instruct/tokenizer.json"))
    ap.add_argument("--n_dev", type=int, default=20)
    ap.add_argument("--n_eval", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--max_input_tokens", type=int, default=32768 - 32)
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lb = load_longbench(a.longbench)
    hp = load_hotpot_dev(a.hotpot)
    items = join_items(lb, hp)
    tok = HFTokenizer(a.tokenizer, "Qwen/Qwen2.5-14B-Instruct@cf98f3b")
    prompt = PromptSpec()

    manifest, anns, excluded = [], [], []
    for it in items:
        h = it["hotpot"]
        ctx_tokens = len(tok.encode(it["context"]))
        fixed = prompt.fixed_len(tok, it["question"])
        row = dict(sample_id=it["sample_id"], hotpot_id=h and h["hotpot_id"], type=h and h["type"],
                   level=h and h["level"], context_tokens=ctx_tokens, fixed_tokens=fixed,
                   prefix_tokens=len(prompt.prefix_ids(tok)), suffix_tokens=len(prompt.suffix_ids(tok, it["question"])),
                   server_input_tokens=ctx_tokens + fixed, answers=it["answers"],
                   answer_agrees_hotpot=bool(h) and h["answer"] in it["answers"])
        reason = None
        if h is None:
            reason = "R1-no-hotpot-match"
        elif row["server_input_tokens"] > a.max_input_tokens:
            reason = "R2-over-max-input"
        row["excluded"] = reason
        manifest.append(row)
        if reason:
            excluded.append((it["sample_id"], reason))
            continue
        anns.append(align_item(it))

    eligible = [r["sample_id"] for r in manifest if not r["excluded"]]
    strata = {r["sample_id"]: r["type"] for r in manifest}
    splits = select_splits(eligible, strata, a.n_dev, a.n_eval, a.seed)
    assert not set(splits["dev"]) & set(splits["eval"])
    by = {r["sample_id"]: r for r in manifest}
    qn = {it["sample_id"]: normalize_answer(it["question"]) for it in items}
    dup_q = sorted({qn[d] for d in splits["dev"]} & {qn[e] for e in splits["eval"]})
    dup_hp = sorted({by[d]["hotpot_id"] for d in splits["dev"]} & {by[e]["hotpot_id"] for e in splits["eval"]})
    # Shared gold titles between dev and eval (same Wikipedia article as evidence) - reported, not excluded.
    titles = {a_["sample_id"]: {f["title"] for f in a_["facts"]} for a_ in anns}
    dev_t = set().union(*(titles[s] for s in splits["dev"]))
    shared_title_eval = sorted(s for s in splits["eval"] if titles[s] & dev_t)

    tier = Counter(f["tier"] for x in anns for f in x["facts"])
    report = dict(
        longbench=dict(revision=LONGBENCH_REVISION, file_sha256=sha256_file(a.longbench), n=len(lb)),
        hotpot=dict(revision=HOTPOT_REVISION, file_sha256=sha256_file(a.hotpot)),
        tokenizer=dict(name=tok.name, sha=tok.sha), seed=a.seed, n_dev=a.n_dev, n_eval=a.n_eval,
        n_matched=sum(1 for it in items if it["hotpot"]), excluded=excluded,
        answers_agree=sum(r["answer_agrees_hotpot"] for r in manifest),
        type_counts=dict(Counter(r["type"] for r in manifest)),
        split_type_counts={k: dict(Counter(by[s]["type"] for s in v)) for k, v in splits.items()},
        dup_question_dev_eval=dup_q, dup_hotpot_id_dev_eval=dup_hp,
        eval_items_sharing_gold_title_with_dev=len(shared_title_eval),
        fact_tiers=dict(tier), n_facts=sum(tier.values()),
        items_all_facts_high=sum(x["all_facts_high"] for x in anns),
        items_by_split_all_facts_high={k: sum(1 for s in v if next(x for x in anns if x["sample_id"] == s)["all_facts_high"])
                                       for k, v in splits.items()},
        suffix_tokens_gt_64=sum(1 for r in manifest if r["suffix_tokens"] > 64),
        server_input_tokens=_q([r["server_input_tokens"] for r in manifest]),
        fixed_tokens=_q([r["fixed_tokens"] for r in manifest]),
        items_le_budget={str(b): sum(1 for r in manifest if r["server_input_tokens"] <= b) for b in (1024, 2048)},
        yesno_items=sum(1 for x in anns if x["answer"]["yesno"]),
        answer_not_in_context=sum(1 for x in anns if not x["answer"]["yesno"] and x["answer"]["n_occurrences"] == 0),
    )
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (out / "annotations_auto.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in anns))
    (out / "splits.json").write_text(json.dumps(splits, indent=1))
    (out / "data_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("excluded",)}, indent=1))


def _q(xs):
    xs = sorted(xs)
    return dict(min=xs[0], p10=xs[len(xs) // 10], p50=xs[len(xs) // 2], p90=xs[9 * len(xs) // 10], max=xs[-1])


if __name__ == "__main__":
    main()
