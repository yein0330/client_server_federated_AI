"""Paired analysis against the full-context reference (exploratory at pilot sample sizes)."""
from __future__ import annotations

import random
from collections import defaultdict


def cell_key(r):
    return (r["method"], r["budget_nominal"], r["question_access"], r["flow"])


def paired_bootstrap(diffs: list[float], n_boot=10000, seed=0, alpha=0.05):
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return dict(mean=sum(diffs) / n, lo=means[int(alpha / 2 * n_boot)], hi=means[int((1 - alpha / 2) * n_boot) - 1], n=n)


def transitions(full_ok: dict[str, bool], cond_ok: dict[str, bool]) -> dict:
    t = {"full_ok->comp_ok": 0, "full_ok->comp_wrong": 0, "full_wrong->comp_ok": 0, "full_wrong->comp_wrong": 0}
    for sid, c in cond_ok.items():
        if sid not in full_ok:
            continue
        f = full_ok[sid]
        t[f"full_{'ok' if f else 'wrong'}->comp_{'ok' if c else 'wrong'}"] += 1
    t["n_paired"] = sum(t.values())
    return t


def summarize(records: list[dict], correctness="em") -> list[dict]:
    ok = [r for r in records if not r.get("error") and not r.get("excluded")]
    full = {r["sample_id"]: r for r in ok if r["method"] == "full"}
    cells = defaultdict(list)
    for r in ok:
        cells[cell_key(r)].append(r)
    rows = []
    for key, rs in sorted(cells.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n = len(rs)
        ev = [r["evidence"] for r in rs if r.get("evidence")]
        paired = [r for r in rs if r["sample_id"] in full]
        rows.append(dict(
            method=key[0], budget=key[1], question_access=key[2], flow=key[3], n=n,
            n_errors=sum(1 for r in records if cell_key(r) == key and r.get("error")),
            mean_f1=sum(r["qa_f1"] for r in rs) / n,
            acc=sum(bool(r[correctness]) for r in rs) / n,
            budget_ok_rate=sum(r["budget_ok"] for r in rs) / n,
            input_len=_quant([r["final_server_input_tokens"] for r in rs]),
            payload_bytes=_quant([r["payload_bytes"] for r in rs]),
            fact_preserved_rate=(sum(e["n_preserved"] for e in ev) / max(1, sum(e["n_known"] for e in ev))) if ev else None,
            all_facts_preserved_rate=(sum(e["all_preserved"] for e in ev) / len(ev)) if ev else None,
            f1_diff_vs_full=paired_bootstrap([r["qa_f1"] - full[r["sample_id"]]["qa_f1"] for r in paired]),
            acc_diff_vs_full=paired_bootstrap([float(r[correctness]) - float(full[r["sample_id"]][correctness]) for r in paired]),
            transitions=transitions({s: bool(f[correctness]) for s, f in full.items()},
                                    {r["sample_id"]: bool(r[correctness]) for r in rs}),
        ))
    return rows


def _quant(xs):
    xs = sorted(xs)
    if not xs:
        return None
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    return dict(min=xs[0], p50=q(0.5), p90=q(0.9), max=xs[-1])


def failure_category(rec: dict, correctness="em") -> str | None:
    """Mechanical pre-label; final category is assigned in manual review."""
    if rec.get(correctness) or rec.get("evidence") is None:
        return None
    e = rec["evidence"]
    if e["uncertain"] or e["n_known"] < e["n_facts"]:
        return "uncertain_mapping_or_judgement"
    if any(x["status"] == "lost" for x in e["evidence"]):
        return "required_evidence_deleted"
    if any(x["status"] == "partial" for x in e["evidence"]) or (e["bridge_entity_kept"] and not all(e["bridge_entity_kept"].values())):
        return "partial_evidence_relation_damaged"
    return "evidence_sufficient_server_failed"
