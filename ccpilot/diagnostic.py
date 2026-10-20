"""Same-budget evidence restoration diagnostic - rules D1-D8 (docs/07_flows_and_diagnostic.md).

Uses gold (manual v1) evidence: it is a diagnostic, NOT a deployable method and NOT an oracle.
All rules are fixed before any server output exists and are identical for every method.

Representation (D1): every condition is a token subsequence of the original context under the
server tokenizer (segmented tokenization), transmitted as IDs with assembly=server_template.
Conditions (per item x method x budget, pilot flow only):
  diag_proj    (D2) method's kept char intervals projected to source tokens (token kept if
               >= 50% of its chars are covered). No edit. Isolates the representation change.
  diag_restore (D3) proj + every not-kept token of alternative #1 of each required requirement
               whose status is not "preserved" (requirements preserved via ANY alternative are
               left alone); then removal (D5) until the context budget holds.
  diag_control (D4) proj + exactly the same number of tokens from NON-evidence sentence units,
               then the same removal (D5).
Control units (D4): sentence units that (a) overlap no quote of any requirement (required,
  non-required or condition, all alternatives), (b) do not contain a gold answer string
  (normalised; not applied to yes/no), (c) are not already kept (projected coverage < 50%).
  Order: units in the passages of the restored requirements first, then all other passages;
  within each group a seeded shuffle (seed = sha256(sample_id) - fixed, output-independent).
  Whole units are added in order; the last unit is cut at the token level so that the number of
  added tokens equals n_add exactly (D6).
Removal (D5), method-agnostic: protected = tokens of every kept required-requirement quote (any
  alternative) + all inserted tokens. Remove unprotected kept tokens, farthest (char distance)
  from any required quote first; ties -> later position first.
Not applicable (D7): n_add == 0 (nothing missing) -> no diag rows; protected > budget -> excluded
  with reason. Reported denominators include both.
No iteration on server outputs (D8): the functions below take no server input.
"""
from __future__ import annotations

import hashlib
import random

from .manual import required
from .textutil import normalize_answer, split_units


def project_to_tokens(offsets, kept, ctx=None) -> list[bool]:
    kept = sorted(kept)
    mask, j = [], 0
    for s, e in offsets:
        while j < len(kept) and kept[j][1] <= s:
            j += 1
        cov, k = 0, j
        while k < len(kept) and kept[k][0] < e:
            cov += max(0, min(e, kept[k][1]) - max(s, kept[k][0]))
            k += 1
        mask.append(e > s and cov * 2 >= (e - s))
    return mask


def _overlaps(s, e, spans):
    return any(s < se and e > ss for ss, se in spans)


def _quotes(ann, which="all"):
    out = []
    for r in ann["requirements"]:
        if which == "required" and r["role"] != "required":
            continue
        for alt in r["alts"]:
            out += [(q["start"], q["end"]) for q in alt]
    return out


def _remove(mask, protected, budget, offsets, req_spans):
    mask = list(mask)
    excess = sum(mask) - budget
    if excess <= 0:
        return mask, 0, True
    def dist(i):
        s, e = offsets[i]
        return min((max(ss - e, s - se, 0) for ss, se in req_spans), default=0)
    cands = sorted((i for i, m in enumerate(mask) if m and not protected[i]), key=lambda i: (-dist(i), -i))
    for i in cands[:excess]:
        mask[i] = False
    return mask, min(excess, len(cands)), len(cands) >= excess


def build_conditions(ctx: str, ctx_ids: list[int], offsets: list[tuple[int, int]], kept: list[tuple[int, int]],
                     ann: dict, req_status: dict[str, str], context_budget: int, answers: list[str]) -> dict:
    """req_status: rid -> preservation status of the method's output (from manual.score_requirements)."""
    proj = project_to_tokens(offsets, kept)
    req_spans = _quotes(ann, "required")
    tok_req = [_overlaps(s, e, req_spans) for s, e in offsets]
    restore_spans = [(q["start"], q["end"]) for r in required(ann) if req_status.get(r["rid"]) != "preserved"
                     for q in r["alts"][0]]
    add = [_overlaps(s, e, restore_spans) and not p for (s, e), p in zip(offsets, proj)]
    n_add = sum(add)
    info = dict(n_add=n_add, restored_rids=[r["rid"] for r in required(ann) if req_status.get(r["rid"]) != "preserved"])
    if n_add == 0:
        return dict(applicable=False, reason="D7: nothing missing", **info)

    out = {"diag_proj": dict(mask=proj, n_added=0, n_removed=0, fits=sum(proj) <= context_budget)}
    restored = [p or a for p, a in zip(proj, add)]
    prot = [m and r for m, r in zip(restored, tok_req)]
    prot = [p or a for p, a in zip(prot, add)]
    m, rem, ok = _remove(restored, prot, context_budget, offsets, req_spans)
    out["diag_restore"] = dict(mask=m, n_added=n_add, n_removed=rem, fits=ok)

    # control units (D4)
    all_q = _quotes(ann, "all")
    ans = [normalize_answer(a) for a in answers if normalize_answer(a) not in ("yes", "no")]
    rest_titles = {q["t"] for r in required(ann) if r["rid"] in info["restored_rids"] for q in r["alts"][0]}
    from .textutil import parse_passages
    pas = parse_passages(ctx)
    title_of = lambda pos: next((p["title"] for p in pas if p["start"] <= pos < p["end"]), "")
    cands = []
    for u in split_units(ctx):
        if u.kind != "sentence" or not ctx[u.start:u.end].strip():
            continue
        if _overlaps(u.start, u.end, all_q):
            continue
        if ans and any(f" {a} " in f" {normalize_answer(ctx[u.start:u.end])} " for a in ans):
            continue
        toks = [i for i, (s, e) in enumerate(offsets) if s >= u.start and e <= u.end]
        if not toks or sum(proj[i] for i in toks) * 2 >= len(toks):
            continue
        cands.append((title_of(u.start) in rest_titles, u.start, toks))
    rng = random.Random(int(hashlib.sha256(ann["sample_id"].encode()).hexdigest()[:16], 16))
    near = [c for c in cands if c[0]]
    far = [c for c in cands if not c[0]]
    rng.shuffle(near)
    rng.shuffle(far)
    ctrl, inserted, added = list(proj), [False] * len(proj), 0
    for _, _, toks in near + far:
        for i in toks:
            if added >= n_add:
                break
            if not ctrl[i]:
                ctrl[i] = inserted[i] = True
                added += 1
        if added >= n_add:
            break
    prot2 = [(c and r) or ins for c, r, ins in zip(ctrl, tok_req, inserted)]
    m2, rem2, ok2 = _remove(ctrl, prot2, context_budget, offsets, req_spans)
    out["diag_control"] = dict(mask=m2, n_added=added, n_removed=rem2, fits=ok2 and added == n_add)
    for v in out.values():
        v["context_ids"] = [t for t, k in zip(ctx_ids, v["mask"]) if k]
        v["kept_intervals"] = [offsets[i] for i, k in enumerate(v["mask"]) if k]
        del v["mask"]
    return dict(applicable=True, conditions=out, **info)
