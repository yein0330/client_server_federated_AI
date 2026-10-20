"""Map HotpotQA supporting facts onto LongBench v1 contexts and score preservation.

LongBench hotpotqa passages are full (later-revision) Wikipedia articles, not the HotpotQA
lead paragraphs, so supporting-fact sentences are aligned by token-F1 against sentence
units (and adjacent unit pairs) of the passage with the same title. Alignment outputs a
confidence tier; tiers below "high" require manual verification before evaluation.
Annotations are evaluation-only and must never be passed to a compressor.
"""
from __future__ import annotations

import re
from collections import Counter

from .textutil import Unit, normalize_answer, parse_passages, split_units, word_tokens

TIER_HIGH, TIER_MED = 0.8, 0.5


def _f1(a: list[str], b: list[str]) -> float:
    c = sum((Counter(a) & Counter(b)).values())
    if not c:
        return 0.0
    p, r = c / len(b), c / len(a)
    return 2 * p * r / (p + r)


def align_item(item: dict) -> dict:
    ctx, h = item["context"], item["hotpot"]
    units = split_units(ctx)
    passages = {p["title"]: p for p in parse_passages(ctx)}
    facts = []
    for k, (title, sid) in enumerate(h["supporting_facts"]):
        f = dict(eid=f"sf{k}", kind="fact", title=title, sent_id=sid, start=None, end=None,
                 score=0.0, tier="unmapped", note="")
        para = h["paragraphs"].get(title, [])
        if sid >= len(para):
            f["note"] = "sent_id out of range in HotpotQA paragraph"
            facts.append(f)
            continue
        sent = para[sid].strip()
        f["source_sentence"] = sent
        p = passages.get(title)
        if p is None:
            f["note"] = "title not among LongBench passages"
            facts.append(f)
            continue
        cand = [u for u in units if u.kind == "sentence" and p["start"] <= u.start < p["end"]]
        q = word_tokens(sent)
        if len(q) < 4:
            f["note"] = "short source sentence (<4 tokens); alignment unreliable"
        best = (0.0, None, None)
        for i, u in enumerate(cand):
            for j in (i, i + 1):
                if j >= len(cand):
                    continue
                s, e = u.start, cand[j].end
                sc = _f1(q, word_tokens(ctx[s:e]))
                if sc > best[0] + 1e-9:
                    best = (sc, s, e)
        sc, s, e = best
        norm_sent = " ".join(sent.split())
        exact = norm_sent and norm_sent in " ".join(ctx[p["start"]:p["end"]].split())
        f.update(score=round(sc, 4), start=s, end=e,
                 tier="exact" if exact and sc >= TIER_HIGH else
                 "high" if sc >= TIER_HIGH else "medium" if sc >= TIER_MED else "unmapped")
        if f["tier"] == "unmapped":
            f["start"] = f["end"] = None
        facts.append(f)

    # Bridge candidates: an SF title mentioned inside another title's mapped SF span.
    bridges = []
    if h["type"] == "bridge":
        titles = sorted({t for t, _ in h["supporting_facts"]})
        for t in titles:
            name = re.sub(r"\s*\(.*?\)\s*$", "", t)
            pat = re.compile(re.escape(name), re.I)
            for f in facts:
                if f["title"] == t or f["start"] is None:
                    continue
                for m in pat.finditer(ctx, f["start"], f["end"]):
                    bridges.append(dict(eid=f"br{len(bridges)}", kind="bridge_mention", entity=name,
                                        in_fact=f["eid"], start=m.start(), end=m.end()))
    # Header units of SF passages (title lines carry entity identity in this format).
    headers = []
    for t in sorted({t for t, _ in h["supporting_facts"]}):
        p = passages.get(t)
        if p:
            headers.append(dict(eid=f"hd{len(headers)}", kind="sf_header", title=t,
                                start=p["start"], end=p["body_start"]))
    ans_norm = [normalize_answer(a) for a in item["answers"]]
    yesno = all(a in ("yes", "no") for a in ans_norm)
    ans_spans = []
    if not yesno:
        for a in item["answers"]:
            for m in re.finditer(re.escape(a), ctx, re.I):
                ans_spans.append((m.start(), m.end()))
    in_fact = any(f["start"] is not None and f["start"] <= s and e <= f["end"]
                  for s, e in ans_spans for f in facts)
    return dict(
        sample_id=item["sample_id"], hotpot_id=h["hotpot_id"], type=h["type"], level=h["level"],
        n_units=len(units), facts=facts, bridges=bridges, headers=headers,
        answer=dict(yesno=yesno, n_occurrences=len(ans_spans), in_mapped_fact=in_fact),
        all_facts_high=all(f["tier"] in ("exact", "high") for f in facts),
        needs_manual=[f["eid"] for f in facts if f["tier"] not in ("exact", "high")],
        # Manual fields (filled before any server output is inspected):
        manual=dict(status="pending", condition_or_negation=None, alternative_evidence=None,
                    corrected_spans={}, annotator=None, notes=""),
    )


# ---------------------------------------------------------------- preservation

def merge_intervals(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def kept_fraction(ctx: str, kept: list[tuple[int, int]], s: int, e: int) -> float:
    """Fraction of non-whitespace chars of ctx[s:e] covered by kept intervals."""
    total = sum(1 for ch in ctx[s:e] if not ch.isspace())
    if total == 0:
        return 1.0
    cov = 0
    for ks, ke in kept:
        a, b = max(s, ks), min(e, ke)
        if a < b:
            cov += sum(1 for ch in ctx[a:b] if not ch.isspace())
    return cov / total


def status(frac: float) -> str:
    return "preserved" if frac >= 0.9 else "partial" if frac >= 0.5 else "lost"


def score_preservation(ctx: str, ann: dict, kept: list[tuple[int, int]]) -> dict:
    kept = merge_intervals(kept)
    rows = []
    for f in ann["facts"]:
        if f["start"] is None:
            rows.append(dict(eid=f["eid"], kind="fact", frac=None, status="unknown", tier=f["tier"]))
            continue
        fr = kept_fraction(ctx, kept, f["start"], f["end"])
        rows.append(dict(eid=f["eid"], kind="fact", frac=round(fr, 4), status=status(fr), tier=f["tier"]))
    br = {}
    for b in ann["bridges"]:
        fr = kept_fraction(ctx, kept, b["start"], b["end"])
        br[b["entity"]] = br.get(b["entity"], False) or fr >= 0.999
    hd = [dict(eid=x["eid"], frac=round(kept_fraction(ctx, kept, x["start"], x["end"]), 4))
          for x in ann["headers"]]
    known = [r for r in rows if r["status"] != "unknown"]
    return dict(
        evidence=rows, bridge_entity_kept=br, headers=hd,
        n_facts=len(rows), n_known=len(known),
        n_preserved=sum(r["status"] == "preserved" for r in known),
        all_preserved=(len(known) == len(rows) and all(r["status"] == "preserved" for r in known)),
        uncertain=any(r["tier"] not in ("exact", "high") for r in rows),
    )


def kept_from_units(units: list[Unit]) -> list[tuple[int, int]]:
    return merge_intervals([(u.start, u.end) for u in units])
