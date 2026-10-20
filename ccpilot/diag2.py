"""real-dev-v2 diagnostics. Rules fixed in code BEFORE any v2 answer was generated.

All conditions are token subsequences of the source context tokenization (server tokenizer,
segmented assembly); the base is the saved real-dev-v1 specpc_pilot selection (positions P).
Question and template are never edited (they live in prefix/suffix).

DEL-v1  evidence-delete vs control-delete (no tokens added):
  E  = base tokens overlapping ANY quote of ANY alternative of ANY required requirement.
  Evidence runs = maximal runs of E that are consecutive in base order.
  Control: for each evidence run (longest first, then earlier), choose a run of the SAME length of
  consecutive base tokens that overlap no annotated quote (any role/alternative) and no gold-answer
  occurrence, not already chosen, minimising |centre - centre(evidence run)| in base order (similar
  document position, same number of runs); ties -> seeded RNG (sha256(sample_id)).
  Not applicable: |E| = 0, or no same-length control run exists.
CTX-v1  context restore vs control-add, for "evidence intact but failed" items:
  related passages = passages containing any required quote.
  A (restore) = unkept tokens of (a) the header unit of each related passage, (b) every sentence unit
  of a related passage that is partially kept, (c) the sentence units immediately before/after each
  unit that contains a required quote (in related passages).
  Control = |A| unkept tokens from sentence units of NON-related passages that overlap no annotated
  quote and contain no gold answer; seeded unit order; whole units, last one cut at token level.
  Common removal set R (same for both): excess = |P| + |A| - C; remove base tokens that overlap no
  required quote and do not belong to a related passage's header, farthest (chars) from any required
  quote first, ties -> later position. Final length of both = |P| (<= C).
  Not applicable: |A| = 0, not enough control tokens, or not enough removable tokens.
"""
from __future__ import annotations

import hashlib
import random

from .manual import required
from .textutil import normalize_answer, parse_passages, split_units

RULES = {"DEL": "DEL-v1", "CTX": "CTX-v1"}


def _ov(s, e, spans):
    return any(s < b and e > a for a, b in spans)


def _quotes(ann, required_only):
    out = []
    for r in ann["requirements"]:
        if required_only and r["role"] != "required":
            continue
        for alt in r["alts"]:
            out += [(q["start"], q["end"]) for q in alt]
    return out


def _answer_spans(ctx, answers):
    import re
    out = []
    for a in answers:
        if normalize_answer(a) in ("yes", "no") or not a.strip():
            continue
        out += [(m.start(), m.end()) for m in re.finditer(re.escape(a), ctx, re.I)]
    return out


def _rng(sample_id, tag):
    return random.Random(int(hashlib.sha256(f"{sample_id}/{tag}".encode()).hexdigest()[:16], 16))


def _spans_of(offsets, positions):
    """Merge token positions into char spans (for saving what was removed/added)."""
    out = []
    for p in sorted(positions):
        s, e = offsets[p]
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def build_deletion(ctx, ctx_ids, offsets, base_pos, ann, answers):
    req = _quotes(ann, True)
    allq = _quotes(ann, False)
    ans = _answer_spans(ctx, answers)
    P = list(base_pos)
    is_ev = [_ov(*offsets[p], req) for p in P]
    E_idx = [k for k, f in enumerate(is_ev) if f]
    if not E_idx:
        return dict(applicable=False, reason="DEL: no annotated evidence tokens in base")
    runs = []
    for k in E_idx:
        if runs and k == runs[-1][-1] + 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    cand = [not _ov(*offsets[p], allq) and not _ov(*offsets[p], ans) for p in P]
    taken = set()
    rng = _rng(ann["sample_id"], "DEL")
    ctrl = []
    for run in sorted(runs, key=lambda r: (-len(r), r[0])):
        L, c0 = len(run), (run[0] + run[-1]) / 2
        options = []
        for st in range(0, len(P) - L + 1):
            win = range(st, st + L)
            if all(cand[k] and k not in taken for k in win):
                options.append((abs((st + st + L - 1) / 2 - c0), rng.random(), st))
        if not options:
            return dict(applicable=False, reason=f"DEL: no non-evidence control run of length {L}")
        _, _, st = min(options)
        ctrl.append(list(range(st, st + L)))
        taken.update(range(st, st + L))
    E_set = set(E_idx)
    C_set = {k for r in ctrl for k in r}
    conds = {}
    for name, drop in (("evidence-delete", E_set), ("control-delete", C_set)):
        keep = [p for k, p in enumerate(P) if k not in drop]
        removed = [P[k] for k in sorted(drop)]
        conds[name] = dict(positions=keep, context_ids=[ctx_ids[p] for p in keep],
                           kept_intervals=[tuple(offsets[p]) for p in keep],
                           edit=dict(rule=RULES["DEL"], removed_tokens=len(removed), added_tokens=0,
                                     n_runs=len(runs), removed_char_spans=_spans_of(offsets, removed),
                                     removed_text=[ctx[s:e] for s, e in _spans_of(offsets, removed)]))
    assert len(conds["evidence-delete"]["positions"]) == len(conds["control-delete"]["positions"])
    return dict(applicable=True, conditions=conds, n_removed=len(E_set), n_runs=len(runs),
                run_lengths=[len(r) for r in runs])


def build_context_restore(ctx, ctx_ids, offsets, base_pos, ann, answers, context_budget):
    req = _quotes(ann, True)
    allq = _quotes(ann, False)
    ans = _answer_spans(ctx, answers)
    P = sorted(base_pos)
    Pset = set(P)
    pas = parse_passages(ctx)
    related = [p for p in pas if any(p["start"] <= a < p["end"] for a, _ in req)]
    rel_ranges = [(p["start"], p["end"]) for p in related]
    in_rel = lambda s: any(a <= s < b for a, b in rel_ranges)
    units = split_units(ctx)
    tok_of = {}
    j = 0
    for u in units:
        toks = []
        while j < len(offsets) and offsets[j][0] < u.end:
            if offsets[j][0] >= u.start:
                toks.append(j)
            j += 1
        tok_of[u.idx] = toks
    A = set()
    header_toks = set()
    for k, u in enumerate(units):
        if not in_rel(u.start):
            continue
        toks = tok_of[u.idx]
        if u.kind == "header":
            header_toks.update(toks)
            A.update(t for t in toks if t not in Pset)
        elif u.kind == "sentence":
            kept = sum(t in Pset for t in toks)
            if 0 < kept < len(toks):
                A.update(t for t in toks if t not in Pset)
            if _ov(u.start, u.end, req):
                for nb in (k - 1, k + 1):
                    if 0 <= nb < len(units) and units[nb].kind == "sentence" and in_rel(units[nb].start):
                        A.update(t for t in tok_of[units[nb].idx] if t not in Pset)
    n_add = len(A)
    info = dict(related_passages=[p["title"] for p in related], n_add=n_add)
    if n_add == 0:
        return dict(applicable=False, reason="CTX: nothing to restore", **info)
    cands = [u for u in units if u.kind == "sentence" and not in_rel(u.start) and ctx[u.start:u.end].strip()
             and not _ov(u.start, u.end, allq) and not _ov(u.start, u.end, ans)
             and any(t not in Pset for t in tok_of[u.idx])]
    rng = _rng(ann["sample_id"], "CTX")
    rng.shuffle(cands)
    ctrl = []
    for u in cands:
        for t in tok_of[u.idx]:
            if len(ctrl) < n_add and t not in Pset:
                ctrl.append(t)
        if len(ctrl) >= n_add:
            break
    if len(ctrl) < n_add:
        return dict(applicable=False, reason="CTX: not enough control tokens", **info)
    excess = len(P) + n_add - context_budget

    def dist(p):
        s, e = offsets[p]
        return min((max(a - e, s - b, 0) for a, b in req), default=0)
    removable = sorted((p for p in P if not _ov(*offsets[p], req) and p not in header_toks),
                       key=lambda p: (-dist(p), -p))
    if excess > len(removable):
        return dict(applicable=False, reason="CTX: not enough removable tokens", **info)
    R = set(removable[:max(0, excess)])
    conds = {}
    for name, add in (("context-restore", A), ("context-control", set(ctrl))):
        keep = sorted((set(P) - R) | add)
        conds[name] = dict(positions=keep, context_ids=[ctx_ids[p] for p in keep],
                           kept_intervals=[tuple(offsets[p]) for p in keep],
                           edit=dict(rule=RULES["CTX"], added_tokens=len(add), removed_tokens=len(R),
                                     added_char_spans=_spans_of(offsets, add),
                                     added_text=[ctx[s:e] for s, e in _spans_of(offsets, add)],
                                     removed_char_spans=_spans_of(offsets, R),
                                     removed_text=[ctx[s:e] for s, e in _spans_of(offsets, R)]))
    assert len(conds["context-restore"]["positions"]) == len(conds["context-control"]["positions"]) == len(P) - len(R) + n_add
    return dict(applicable=True, conditions=conds, removal_set_size=len(R), **info)


def residual_check(ann, ctx, visible: str, answers) -> dict:
    """Mechanical residue after evidence deletion: is a gold string or any key term of the deleted
    requirements still visible? (Alternative evidence outside the annotation is NOT ruled out.)"""
    import re

    from .manual import key_terms
    gold_left = [a for a in answers if normalize_answer(a) not in ("yes", "no") and re.search(re.escape(a), visible, re.I)]
    kt = {}
    for r in required(ann):
        terms = list(dict.fromkeys(t for q in r["alts"][0] for t in key_terms(ctx[q["start"]:q["end"]])))
        kt[r["rid"]] = [t for t in terms if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", visible)]
    return dict(gold_string_still_visible=gold_left, key_terms_still_visible=kt)
