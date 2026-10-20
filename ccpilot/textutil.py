"""Text normalisation, QA scoring and deterministic sentence units.

qa_f1 / normalize_answer are ported verbatim in behaviour from
LongBench v1 metrics.py (identical copy in SpecPC dataset/longbench/metrics.py).
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(prediction: str, ground_truth: str) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, answers: list[str]) -> float:
    return max(qa_f1(prediction, a) for a in answers)


def exact_match(prediction: str, answers: list[str]) -> bool:
    p = normalize_answer(prediction)
    return any(p == normalize_answer(a) for a in answers)


def answer_contained(prediction: str, answers: list[str]) -> bool:
    """Normalised gold appears as a whole-word span of the normalised prediction."""
    p = f" {normalize_answer(prediction)} "
    return any(normalize_answer(a) and f" {normalize_answer(a)} " in p for a in answers)


# LongBench-style post-processing of generations (SpecPC format_prediction).
_ANSWER_TAGS = ["<think/>", "Answer:", "Summary:", "The final answer is:", "The answer is:"]


def format_prediction(pred: str) -> str:
    for tag in _ANSWER_TAGS:
        pred = pred.split(tag)[-1]
    return pred.strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------- passages / units

_PASSAGE_RE = re.compile(r"(?:^|\n)Passage (\d+):\n")
# Boundary after sentence punctuation (+optional closing quote/bracket) and whitespace,
# or the LongBench artefact "inmates.It" (no space) when the char before '.' is lower/digit.
_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?])[\"'”’)\]]?\s+(?=[\"“'(\[]?[A-Z0-9])"
    r"|(?<=[a-z0-9][.!?])(?=[A-Z][a-z])"
)


_ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "mt", "ft", "vs", "no", "gen", "col", "lt", "rev", "prof"}


def _is_abbrev(seg: str, pos: int) -> bool:
    """True if the punctuation ending at `pos` closes an initial (E. L.), a dotted
    acronym (U.S.) or a listed abbreviation - not a sentence end."""
    j = pos
    while j > 0 and seg[j - 1] in "\"'”’)]":
        j -= 1
    if j == 0 or seg[j - 1] != ".":
        return False
    k = j - 1
    w = k
    while w > 0 and seg[w - 1].isalpha():
        w -= 1
    word = seg[w:k]
    if len(word) == 1 and word.isupper():
        return True  # initial or last letter of U.S.
    return word.lower() in _ABBREV


@dataclass(frozen=True)
class Unit:
    """A contiguous char span [start, end) of the context. Units tile the context."""

    idx: int
    start: int
    end: int
    kind: str  # "header" | "sentence" | "gap"
    passage: int  # passage number (1-based), 0 before first passage

    def text(self, context: str) -> str:
        return context[self.start : self.end]


def parse_passages(context: str) -> list[dict]:
    """Return [{num, title, start, body_start, end}] char offsets into context."""
    ms = list(_PASSAGE_RE.finditer(context))
    out = []
    starts = [m.start() + (1 if context[m.start()] == "\n" else 0) for m in ms]
    for i, m in enumerate(ms):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(ms) else len(context)
        title_start = m.end()
        nl = context.find("\n", title_start, end)
        title_end = nl if nl != -1 else end
        body_start = min(title_end + 1, end)
        out.append(dict(num=int(m.group(1)), title=context[title_start:title_end].strip(),
                        start=start, body_start=body_start, end=end))
    return out


def split_units(context: str) -> list[Unit]:
    """Deterministic segmentation. Each passage header ("Passage k:\\nTitle\\n") is one unit;
    bodies are split per line, then per sentence. Trailing whitespace stays with the
    preceding unit so that ''.join(units) == context."""
    spans: list[tuple[int, int, str, int]] = []
    passages = parse_passages(context)
    if not passages:
        passages = [dict(num=0, title="", start=0, body_start=0, end=len(context))]
    elif passages[0]["start"] > 0:
        spans.append((0, passages[0]["start"], "gap", 0))
    for p in passages:
        if p["body_start"] > p["start"]:
            spans.append((p["start"], p["body_start"], "header", p["num"]))
        pos = p["body_start"]
        body = context[pos : p["end"]]
        for line in re.finditer(r"[^\n]*\n?", body):
            if not line.group(0):
                continue
            ls, le = pos + line.start(), pos + line.end()
            seg = context[ls:le]
            cuts = [m.end() for m in _SENT_BOUNDARY.finditer(seg) if not _is_abbrev(seg, m.start())]
            prev = 0
            for c in cuts + [len(seg)]:
                if c > prev:
                    spans.append((ls + prev, ls + c, "sentence", p["num"]))
                prev = c
    units = [Unit(i, s, e, k, n) for i, (s, e, k, n) in enumerate(spans)]
    assert "".join(u.text(context) for u in units) == context, "units must tile context"
    return units


def join_units(context: str, units: list[Unit]) -> str:
    """Concatenate selected units in original order; insert a space where a unit has no
    trailing whitespace and the next selected unit is not its original successor."""
    units = sorted(units, key=lambda u: u.start)
    parts = []
    for i, u in enumerate(units):
        t = u.text(context)
        nxt = units[i + 1] if i + 1 < len(units) else None
        if nxt is not None and nxt.start != u.end and not t[-1:].isspace():
            t += " "
        parts.append(t)
    return "".join(parts)
