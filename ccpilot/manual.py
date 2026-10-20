"""Manual evidence annotations (rules v1, docs/06_annotation_rules.md).

An item has requirements; each requirement has >=1 alternatives; an alternative is a list of
quotes (passage title + exact text, whitespace-insensitive) that together state the needed
proposition. A requirement is preserved if ANY alternative has ALL its quotes preserved.
Annotations are evaluation-only and are never passed to a compressor.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .evidence import kept_fraction, merge_intervals, status
from .textutil import parse_passages

RULES_VERSION = "v1"
REQ_VERDICTS = {"verified", "paraphrase", "changed_nonessential", "changed_essential", "absent"}
SUPPORT = {"supported", "supported_premise_mismatch", "weak", "ambiguous", "not_entailed"}
ROLES = {"required", "non_required"}


def _norm_index(text: str):
    """Whitespace-collapsed copy of text plus map from normalised index -> original index."""
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
        idx.append(i)
    return "".join(out), idx


def resolve_quote(ctx: str, title: str, quote: str) -> tuple[int, int, int]:
    """Return (start, end, n_occurrences) of the first occurrence of quote inside passage `title`."""
    ps = [p for p in parse_passages(ctx) if p["title"] == title]
    if len(ps) != 1:
        raise KeyError(f"passage title not unique/found: {title!r}")
    p = ps[0]
    seg = ctx[p["start"]:p["end"]]
    ns, idx = _norm_index(seg)
    q = " ".join(quote.split())
    hits = [m.start() for m in re.finditer(re.escape(q), ns)]
    if not hits:
        raise KeyError(f"quote not found in {title!r}: {quote[:80]!r}")
    s = hits[0]
    return p["start"] + idx[s], p["start"] + idx[s + len(q) - 1] + 1, len(hits)


def resolve_item(ctx: str, ann: dict) -> dict:
    """Attach char spans to every quote; raises on any unresolved quote."""
    out = json.loads(json.dumps(ann))
    for r in out["requirements"]:
        assert r["role"] in ROLES and r["verdict"] in REQ_VERDICTS, r
        for alt in r["alts"]:
            for q in alt:
                q["start"], q["end"], q["n_occ"] = resolve_quote(ctx, q["t"], q["q"])
    assert out["support"] in SUPPORT, out["support"]
    return out


def load_manual(path: str | Path, contexts: dict[str, str]) -> dict[str, dict]:
    raw = json.loads(Path(path).read_text())
    return {a["sample_id"]: resolve_item(contexts[a["sample_id"]], a) for a in raw["items"]}


def required(ann: dict) -> list[dict]:
    return [r for r in ann["requirements"] if r["role"] == "required"]


def diag_eligible(ann: dict) -> tuple[bool, str]:
    if ann["support"] != "supported":
        return False, f"support={ann['support']}"
    if any(r["verdict"] in ("absent", "changed_essential") for r in required(ann)):
        return False, "required evidence absent/changed"
    if not required(ann):
        return False, "no required evidence"
    return True, ""


def score_requirements(ctx: str, ann: dict, kept: list[tuple[int, int]]) -> dict:
    kept = merge_intervals(kept)
    rows = []
    for r in required(ann):
        best = None
        for k, alt in enumerate(r["alts"]):
            fr = min(kept_fraction(ctx, kept, q["start"], q["end"]) for q in alt)
            if best is None or fr > best[0]:
                best = (fr, k)
        rows.append(dict(rid=r["rid"], kind=r.get("kind", "fact"), frac=round(best[0], 4), status=status(best[0]),
                         best_alt=best[1], n_alts=len(r["alts"])))
    return dict(evidence=rows, n_facts=len(rows), n_known=len(rows),
                n_preserved=sum(x["status"] == "preserved" for x in rows),
                all_preserved=bool(rows) and all(x["status"] == "preserved" for x in rows),
                uncertain=ann["support"] != "supported", bridge_entity_kept={}, headers=[], source="manual_v1")


def primary_spans(ann: dict) -> list[tuple[int, int]]:
    return [(q["start"], q["end"]) for r in required(ann) for q in r["alts"][0]]


def all_required_spans(ann: dict) -> list[tuple[int, int]]:
    return [(q["start"], q["end"]) for r in required(ann) for alt in r["alts"] for q in alt]


# ---------------------------------------------------------------- key terms (mechanical)
_STOP_CAP = {"The", "A", "An", "It", "He", "She", "They", "His", "Her", "Its", "Their", "In", "On", "At", "Of",
             "As", "By", "For", "From", "This", "That", "These", "Those", "There", "After", "Before", "When"}
_NEG = re.compile(r"\b(not|no|never|without|neither|nor|except|only)\b", re.I)


def key_terms(text: str) -> list[str]:
    """Mechanical key-term list of a quote: numbers, capitalised words (minus function words),
    negation/limiting words. NOT a semantic judgement."""
    nums = re.findall(r"\b\d[\w,.]*", text)  # whole numeric tokens incl. ordinals (6th) and 1,234
    caps = [w for w in re.findall(r"\b[A-Z][\w'’&.-]*", text) if w not in _STOP_CAP and len(w) > 1]
    negs = [m.group(0).lower() for m in _NEG.finditer(text)]
    seen, out = set(), []
    for t in nums + caps + negs:
        t = t.rstrip(".,")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def key_term_check(ctx: str, ann: dict, visible_context: str, best_alt: dict[str, int] | None = None) -> dict:
    """For each required requirement: which key terms of its (best) alternative appear in the
    server-visible context text. Word-boundary, case-sensitive (negations case-insensitive)."""
    rows = {}
    for r in required(ann):
        k = (best_alt or {}).get(r["rid"], 0)
        terms = []
        for q in r["alts"][k]:
            terms += key_terms(ctx[q["start"]:q["end"]])
        terms = list(dict.fromkeys(terms))
        miss = []
        for t in terms:
            flags = re.I if t.lower() in {"not", "no", "never", "without", "neither", "nor", "except", "only"} else 0
            if not re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", visible_context, flags):
                miss.append(t)
        rows[r["rid"]] = dict(alt=k, n_terms=len(terms), missing=miss)
    return dict(per_requirement=rows, all_terms_present=all(not v["missing"] for v in rows.values()))


def evaluate_manual(ctx: str, ann: dict, kept: list[tuple[int, int]], visible_context: str) -> dict:
    """AI-written annotation evaluation: 'evidence-span character retention' (NOT semantic
    preservation) + mechanical key-term presence."""
    ch = score_requirements(ctx, ann, kept)
    best = {e["rid"]: e["best_alt"] for e in ch["evidence"]}
    kt = key_term_check(ctx, ann, visible_context, best)
    for e in ch["evidence"]:
        e["key_terms_missing"] = kt["per_requirement"][e["rid"]]["missing"]
        e["intact"] = e["status"] == "preserved" and not e["key_terms_missing"]
    return dict(annotation="manual_v1 (AI-written, human audit pending)", support=ann["support"],
                metric_char="evidence-span character retention (>=0.9 preserved)",
                requirements=ch["evidence"], n_required=ch["n_facts"],
                n_char_preserved=ch["n_preserved"], all_char_preserved=ch["all_preserved"],
                n_intact=sum(e["intact"] for e in ch["evidence"]),
                all_intact=bool(ch["evidence"]) and all(e["intact"] for e in ch["evidence"]))
