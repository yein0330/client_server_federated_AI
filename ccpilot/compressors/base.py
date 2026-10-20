"""Compressor interface.

A compressor sees ONLY (context, question-if-allowed, budget, tokenizer). It returns the
context token IDs to transmit plus evaluation-only provenance (kept char intervals of the
original context). Provenance never enters the payload.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..prompt import PromptSpec, TokenizerLike


@dataclass
class CompressionResult:
    context_ids: list[int]
    kept_intervals: list[tuple[int, int]] | None  # eval-only; None if not recoverable
    pre_fit_context_len: int  # context tokens before budget fitting
    fit_iterations: int = 0
    fit_truncated: bool = False  # final hard truncation applied
    fit_failed: bool = False
    t_compress_s: float = 0.0
    t_fit_s: float = 0.0
    info: dict = field(default_factory=dict)


@dataclass
class CompressRequest:
    context: str
    question: str | None  # None when the method is run question-unaware
    context_budget: int  # server-tokenizer tokens available for the context segment
    tok: TokenizerLike
    prompt: PromptSpec


class Compressor:
    method_id: str = "base"
    uses_question: bool = False  # whether the selection signal can depend on the question

    def compress(self, req: CompressRequest) -> CompressionResult:
        raise NotImplementedError


class FullContext(Compressor):
    """Reference condition: the entire context is transmitted (not an oracle or upper bound)."""

    method_id = "full"

    def compress(self, req):
        t0 = time.perf_counter()
        ids = req.tok.encode(req.context)
        return CompressionResult(ids, [(0, len(req.context))], len(ids),
                                 t_compress_s=time.perf_counter() - t0)


def hard_truncate_tail(tok: TokenizerLike, ids: list[int], budget: int) -> list[int]:
    """Last-resort fitting: drop trailing context tokens (IDs are transmitted as-is,
    so no decode/re-encode drift)."""
    return list(ids[:budget])
