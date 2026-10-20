"""SpecPC token selection: numpy port of the official selection + the pilot variant.

Official reference: furiosa-ai/draft-based-approx-llm @ d70fabb,
draft_approx_llm/specpc/specpc.py::specpc_select_input_token_indices (batch size 1).

Two flows (kept separate in all reporting):
  * official_select  : reproduction flow. Budget = max_capacity_prompt over the WHOLE
                       templated prompt; only the last `window_size` tokens are protected,
                       so system prompt / instruction / question head may be dropped.
  * pilot_select     : pilot flow (implementation change). Prefix and suffix (template +
                       question) are always kept; selection runs over context positions only
                       with context budget C = B - |prefix| - |suffix|; no tail window.
Tie-breaking: torch.topk / torch.argsort tie order is implementation-defined and max-pooling
creates long plateaus of equal scores. This port breaks ties by lower index first (stable).
The torch-equivalence test therefore checks exact equality only for tie-free inputs.

Attention scores come from `score_fn(full_ids) -> np.ndarray[L]` (draft model forward with
lookahead; see adapters/specpc_draft.py, NOT executed in this stage) or a mock.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .base import CompressionResult, Compressor, CompressRequest


@dataclass(frozen=True)
class SpecPCParams:
    # Values from cfg/paper/specpc/longbench/qwen25_05b_14b/cmax_{1024,2048}/hotpotqa.yaml
    window_size: int = 64
    pool_type: str | None = "max"
    kernel_size: int = 64
    neighbor_tokens: int = 64
    reduction_type: str = "max"
    lookahead_tokens: int = 1
    starting_layer_index: int = 8
    weighted_query: bool = True


def _pool(x: np.ndarray, p: SpecPCParams) -> np.ndarray:
    if p.pool_type is None:
        return x
    k = p.kernel_size
    pad = ((k - 1) // 2, k - 1 - (k - 1) // 2)
    xp = np.pad(x, pad, mode="constant", constant_values=0)
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    return win.max(-1) if p.pool_type == "max" else win.mean(-1)


def _stable_topk(x: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-x, kind="stable")[:k]


def _unique_keep_order(a: np.ndarray) -> np.ndarray:
    _, first = np.unique(a, return_index=True)
    return a[np.sort(first)]


def _adj(nt: int) -> np.ndarray:
    adj = np.arange(-((nt - 1) // 2), nt - 1 - (nt - 1) // 2 + 1)
    return adj[np.argsort(np.abs(adj), kind="stable")]


def official_select(attn: np.ndarray, cap: int, p: SpecPCParams) -> np.ndarray:
    L = attn.shape[0]
    s = _pool(attn.astype(np.float64), p)
    ind = _stable_topk(s, cap) if cap < L else np.arange(L)
    if p.neighbor_tokens > 0:
        ind = _unique_keep_order(np.clip((ind[:, None] + _adj(p.neighbor_tokens)).reshape(-1), 0, L - 1))
    if p.window_size > L:
        raise ValueError("window_size cannot be greater than seq_len")
    ind = _unique_keep_order(np.concatenate([np.arange(L - p.window_size, L), ind]))
    return np.sort(ind[:cap])


def pilot_select(attn: np.ndarray, ctx_start: int, ctx_end: int, ctx_budget: int,
                 p: SpecPCParams) -> np.ndarray:
    """Absolute positions (within [ctx_start, ctx_end)) of selected context tokens."""
    s = _pool(attn.astype(np.float64), p)[ctx_start:ctx_end]
    n = ctx_end - ctx_start
    if ctx_budget >= n:
        return np.arange(ctx_start, ctx_end)
    ind = _stable_topk(s, ctx_budget)
    if p.neighbor_tokens > 0:
        ind = _unique_keep_order(np.clip((ind[:, None] + _adj(p.neighbor_tokens)).reshape(-1), 0, n - 1))
    return np.sort(ind[:ctx_budget]) + ctx_start


def load_official_select(allow_source_extract: bool = False):
    """Return (select_fn, origin). select_fn(attn: np.ndarray[L], cap, params) -> np.ndarray of
    sorted positions, computed by the OFFICIAL torch function
    draft_approx_llm.specpc.specpc.specpc_select_input_token_indices (pinned d70fabb).

    origin = "import": the official package is imported as-is (GPU env, full requirements).
    origin = "source_extract": only for the model-free env, where the official package's
             imports (einops, ...) are missing; the two official function bodies are compiled
             from the pinned source file unchanged. Never used for the GPU smoke.
    """
    import sys
    import types

    import torch

    repo = Path(__file__).resolve().parents[2] / "external" / "specpc"
    try:
        sys.path.insert(0, str(repo))
        from draft_approx_llm.specpc.specpc import specpc_select_input_token_indices as fn
        origin = "import"
    except ImportError:
        if not allow_source_extract:
            raise
        import ast

        import pandas as pd
        import torch.nn.functional as F

        src = (repo / "draft_approx_llm/specpc/specpc.py").read_text()
        keep = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                and n.name in ("_unique", "specpc_select_input_token_indices")]
        ns = {"torch": torch, "F": F, "np": np, "pd": pd, "SpecPCConfig": object}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(repo / "draft_approx_llm/specpc/specpc.py"), "exec"), ns)
        fn = ns["specpc_select_input_token_indices"]
        origin = "source_extract"

    def select(attn: np.ndarray, cap: int, p: SpecPCParams, window_size: int | None = None, device="cpu"):
        cfg = types.SimpleNamespace(kernel_size=p.kernel_size, neighbor_tokens=p.neighbor_tokens,
                                    pool_type=p.pool_type, max_capacity_prompt=cap,
                                    window_size=p.window_size if window_size is None else window_size)
        x = torch.as_tensor(np.asarray(attn), device=device)[None]
        return fn(x, cfg)[0].cpu().numpy()

    return select, origin


class SpecPCPilot(Compressor):
    """Pilot flow (implementation change vs official): prefix/suffix always kept; the OFFICIAL
    selection function is applied to the context slice of the draft scores with
    max_capacity_prompt = context budget and window_size = 0 (no tail window inside the context).
    Difference from official besides protection: max-pooling sees only context scores
    (zero padding at the context edges instead of neighbouring template tokens).
    Question-aware: draft attention is read from the final window, which contains the question.
    Transmits selected server-tokenizer IDs (no re-tokenization)."""

    method_id = "specpc_pilot"
    uses_question = True

    def __init__(self, score_fn: Callable[[list[int]], np.ndarray], params: SpecPCParams = SpecPCParams(),
                 score_fn_id: str = "unset", selector=None, selector_origin: str = "unset"):
        self.score_fn, self.params, self.score_fn_id = score_fn, params, score_fn_id
        self.selector, self.selector_origin = selector, selector_origin

    def compress(self, req: CompressRequest) -> CompressionResult:
        assert req.question is not None
        t0 = time.perf_counter()
        pre = req.prompt.prefix_ids(req.tok)
        ctx_ids = req.tok.encode(req.context)
        full = pre + ctx_ids + req.prompt.suffix_ids(req.tok, req.question)
        attn = np.asarray(self.score_fn(full), dtype=np.float64)
        assert attn.shape == (len(full),), "score_fn must return one score per input token"
        t1 = time.perf_counter()
        a, b = len(pre), len(pre) + len(ctx_ids)
        if req.context_budget >= len(ctx_ids):
            rel = list(range(len(ctx_ids)))
        elif self.selector is not None:
            rel = self.selector(attn[a:b], req.context_budget, self.params, window_size=0).tolist()
        else:  # numpy port: tests / tie study only
            rel = (pilot_select(attn, a, b, req.context_budget, self.params) - a).tolist()
        out_ids = [ctx_ids[i] for i in rel]
        offs = req.tok.offsets(req.context)
        kept = [offs[i] for i in rel]
        t2 = time.perf_counter()
        return CompressionResult(
            out_ids, kept, len(ctx_ids), t_compress_s=t1 - t0, t_fit_s=t2 - t1,
            info=dict(score_fn=self.score_fn_id, selector=self.selector_origin if self.selector else "numpy_port",
                      selected_token_pos=rel, n_context_tokens=len(ctx_ids)))


class SpecPCRepro(Compressor):
    """Official-flow reproduction (keeps official settings):
      * the whole chat-templated prompt string is tokenized JOINTLY (as official eval.py);
      * OFFICIAL selection over the whole prompt, budget = max_capacity_prompt = B, tail window 64;
      * output is the complete server input (payload assembly = client_raw).
    Server generation for this flow uses the official profile (see server_worker.GEN_PROFILES)."""

    method_id = "specpc_repro"
    uses_question = True

    def __init__(self, score_fn, params: SpecPCParams = SpecPCParams(), score_fn_id: str = "unset",
                 selector=None, selector_origin: str = "unset"):
        self.score_fn, self.params, self.score_fn_id = score_fn, params, score_fn_id
        self.selector, self.selector_origin = selector, selector_origin
        self.total_budget: int | None = None  # set by the runner (nominal B)

    def compress(self, req: CompressRequest) -> CompressionResult:
        t0 = time.perf_counter()
        pre_text = req.prompt.prefix_text()
        full_text = pre_text + req.context + req.prompt.suffix_text(req.question)
        full = req.tok.encode(full_text)
        offs = req.tok.offsets(full_text)
        attn = np.asarray(self.score_fn(full), dtype=np.float64)
        t1 = time.perf_counter()
        L = len(full)
        if self.total_budget >= L:
            sel = list(range(L))  # official: only reached with force_lookahead; selection keeps all
        elif self.selector is not None:
            sel = self.selector(attn, self.total_budget, self.params).tolist()
        else:
            sel = official_select(attn, self.total_budget, self.params).tolist()
        raw = [full[i] for i in sel]
        c0, c1 = len(pre_text), len(pre_text) + len(req.context)
        in_ctx = [i for i in sel if offs[i][0] >= c0 and offs[i][1] <= c1]
        kept = [(offs[i][0] - c0, offs[i][1] - c0) for i in in_ctx]
        n_ctx_tokens = sum(1 for s, e in offs if s >= c0 and e <= c1)
        pre_n = sum(1 for s, e in offs if e <= c0)
        suf_n = sum(1 for s, e in offs if s >= c1)
        info = dict(score_fn=self.score_fn_id, selector=self.selector_origin if self.selector else "numpy_port",
                    tokenization="joint", raw_ids=raw, n_total=L,
                    prefix_kept=sum(1 for i in sel if offs[i][1] <= c0), prefix_len=pre_n,
                    suffix_kept=sum(1 for i in sel if offs[i][0] >= c1), suffix_len=suf_n,
                    boundary_tokens=L - n_ctx_tokens - pre_n - suf_n)
        return CompressionResult([full[i] for i in in_ctx], kept, n_ctx_tokens,
                                 t_compress_s=t1 - t0, t_fit_s=time.perf_counter() - t1, info=info)
