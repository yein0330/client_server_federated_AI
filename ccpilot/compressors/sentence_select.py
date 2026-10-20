"""Deterministic sentence-unit selection baselines.

bm25     : question-aware. Okapi BM25 (k1=1.5, b=0.75) of each unit against the question,
           per-item corpus = units of that item. Order: score desc, then position asc.
           Greedy skip-and-continue fill under the context budget; restore original order.
uniform  : question-unaware. Choose k units at evenly spaced positions
           floor((i + 0.5) * N / k), i < k, for the largest k whose assembly fits.
Fitting  : unit lengths are estimated by encoding each unit alone; the assembled context
           is re-encoded exactly and, while over budget, the last-added (bm25) / the
           selection with k-1 (uniform) is used. Never truncates mid-unit.
"""
from __future__ import annotations

import math
import time
from collections import Counter

from ..textutil import Unit, join_units, split_units, word_tokens
from .base import CompressionResult, Compressor, CompressRequest


def bm25_scores(units_tokens: list[list[str]], query: list[str], k1=1.5, b=0.75) -> list[float]:
    n = len(units_tokens)
    avgdl = sum(map(len, units_tokens)) / max(n, 1) or 1.0
    df = Counter(t for toks in units_tokens for t in set(toks))
    q = Counter(query)
    out = []
    for toks in units_tokens:
        tf = Counter(toks)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(toks) / avgdl))
        out.append(s)
    return out


def _assembled_len(req, units):
    return len(req.tok.encode(join_units(req.context, units)))


class BM25Select(Compressor):
    method_id = "sent_bm25"
    uses_question = True

    def compress(self, req: CompressRequest) -> CompressionResult:
        assert req.question is not None, "bm25 requires question access"
        t0 = time.perf_counter()
        units = [u for u in split_units(req.context) if u.text(req.context).strip()]
        scores = bm25_scores([word_tokens(u.text(req.context)) for u in units], word_tokens(req.question))
        order = sorted(range(len(units)), key=lambda i: (-scores[i], units[i].start))
        est = [len(req.tok.encode(u.text(req.context))) for u in units]
        chosen, used = [], 0
        for i in order:
            if used + est[i] <= req.context_budget:
                chosen.append(i)
                used += est[i]
        t1 = time.perf_counter()
        it = 0
        sel = [units[i] for i in chosen]
        while sel and _assembled_len(req, sel) > req.context_budget:
            chosen.pop()
            sel = [units[i] for i in chosen]
            it += 1
        text = join_units(req.context, sel)
        ids = req.tok.encode(text)
        t2 = time.perf_counter()
        return CompressionResult(
            ids, [(u.start, u.end) for u in sorted(sel, key=lambda u: u.start)], used,
            fit_iterations=it, t_compress_s=t1 - t0, t_fit_s=t2 - t1,
            info=dict(n_units=len(units), n_selected=len(sel),
                      selected_unit_idx=sorted(u.idx for u in sel),
                      unit_scores={units[i].idx: round(scores[i], 4) for i in chosen}))


def uniform_indices(n: int, k: int) -> list[int]:
    return sorted({min(n - 1, int((i + 0.5) * n / k)) for i in range(k)}) if k > 0 else []


class UniformSelect(Compressor):
    method_id = "sent_uniform"
    uses_question = False

    def compress(self, req: CompressRequest) -> CompressionResult:
        t0 = time.perf_counter()
        units = [u for u in split_units(req.context) if u.text(req.context).strip()]
        est = [len(req.tok.encode(u.text(req.context))) for u in units]
        n = len(units)
        k = n
        while k > 0 and sum(est[i] for i in uniform_indices(n, k)) > req.context_budget:
            k -= 1
        pre = sum(est[i] for i in uniform_indices(n, k))
        t1 = time.perf_counter()
        it = 0
        while k > 0 and _assembled_len(req, [units[i] for i in uniform_indices(n, k)]) > req.context_budget:
            k -= 1
            it += 1
        sel = [units[i] for i in uniform_indices(n, k)]
        ids = req.tok.encode(join_units(req.context, sel))
        t2 = time.perf_counter()
        return CompressionResult(
            ids, [(u.start, u.end) for u in sel], pre, fit_iterations=it,
            t_compress_s=t1 - t0, t_fit_s=t2 - t1,
            info=dict(n_units=n, n_selected=len(sel), selected_unit_idx=[u.idx for u in sel]))
