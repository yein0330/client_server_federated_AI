"""LLMLingua-2 adapter (UNVERIFIED: not executed in this stage; requires model download).

Facts from source (microsoft/LLMLingua @ 5a4c78a):
  * compress_prompt_llmlingua2(context, rate, target_token, ...) has no `question`
    argument -> the selection signal is question-unaware. We report it as such.
  * target_token / origin_tokens / compressed_tokens are counted with
    tiktoken.encoding_for_model("gpt-3.5-turbo"), not the server tokenizer.
  * Official LongBench script: --target_token 2000 --force_tokens "\\n,?,!,." and
    drop_consecutive=True (compress.py default); model
    microsoft/llmlingua-2-xlm-roberta-large-meetingbank.
Budget rule (pilot, fixed in dev): iterate target_token proportionally on the measured
server-tokenizer length of the compressed context, at most MAX_ITERS calls, keep the
longest feasible result; fall back to tail truncation of IDs (flagged).
"""
from __future__ import annotations

import os
import time

from .base import CompressionResult, Compressor, CompressRequest, hard_truncate_tail

MAX_ITERS = 4
SHRINK_MARGIN = 0.97
LOWER_OK = 0.95  # stop once within [LOWER_OK * C, C]


class LLMLingua2(Compressor):
    method_id = "llmlingua2"
    uses_question = False

    def __init__(self, compress_fn=None, model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                 force_tokens=("\n", "?", "!", "."), drop_consecutive=True):
        """compress_fn(context, target_token) -> (compressed_text, word_labels | None).
        Default builds the real PromptCompressor lazily (guarded)."""
        self.model_name, self.force_tokens, self.drop_consecutive = model_name, list(force_tokens), drop_consecutive
        self._fn = compress_fn

    def _real_fn(self):
        if os.environ.get("CCPILOT_ALLOW_MODELS") != "1":
            raise RuntimeError("model loading disabled in this stage (set CCPILOT_ALLOW_MODELS=1 when approved)")
        from llmlingua import PromptCompressor  # noqa: requires pip install at pinned commit

        pc = PromptCompressor(model_name=self.model_name, use_llmlingua2=True)

        def fn(context, target_token):
            r = pc.compress_prompt_llmlingua2([context], target_token=int(target_token),
                                              force_tokens=self.force_tokens,
                                              drop_consecutive=self.drop_consecutive,
                                              return_word_label=True)
            return r["compressed_prompt"], r.get("fn_labeled_original_prompt")
        return fn

    def compress(self, req: CompressRequest) -> CompressionResult:
        fn = self._fn or self._real_fn()
        self._fn = fn
        C = req.context_budget
        target = float(C)  # first call: nominal target in (tiktoken) units = server budget
        best, tries, t_comp, first_len = None, [], 0.0, None
        t_start = time.perf_counter()
        for it in range(MAX_ITERS):
            t0 = time.perf_counter()
            text, labels = fn(req.context, max(1, int(target)))
            t_comp += time.perf_counter() - t0
            ids = req.tok.encode(text)
            first_len = len(ids) if first_len is None else first_len
            tries.append(dict(target_token=int(target), server_len=len(ids)))
            if len(ids) <= C and (best is None or len(ids) > len(best[0])):
                best = (ids, labels)
            if LOWER_OK * C <= len(ids) <= C:
                break
            target = target * (C / max(len(ids), 1)) * (SHRINK_MARGIN if len(ids) > C else 1.0)
        truncated = False
        if best is None:
            ids, labels = req.tok.encode(text), labels
            ids, truncated = hard_truncate_tail(req.tok, ids, C), True
            best = (ids, labels)
        kept = word_labels_to_intervals(req.context, best[1]) if best[1] else None
        return CompressionResult(
            best[0], kept, first_len, fit_iterations=len(tries) - 1, fit_truncated=truncated,
            t_compress_s=t_comp, t_fit_s=time.perf_counter() - t_start - t_comp,
            info=dict(tries=tries, model=self.model_name, kept_intervals_note=(
                "word-label alignment; partial-word drops not represented" if kept else "unavailable")))


def word_labels_to_intervals(context: str, labeled: str, word_sep="\t\t|\t\t", label_sep=" "):
    """Map LLMLingua-2 '(word label)' sequence back onto char spans by forward search.
    Returns None if any word cannot be located (then preservation is 'unknown')."""
    out, cur = [], 0
    for item in labeled.split(word_sep):
        if not item:
            continue
        word, _, lab = item.rpartition(label_sep)
        pos = context.find(word, cur)
        if pos < 0:
            return None
        if lab.strip() == "1":
            out.append((pos, pos + len(word)))
        cur = pos + len(word)
    return out
