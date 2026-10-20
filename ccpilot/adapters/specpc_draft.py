"""Draft-model attention scorer for SpecPC (UNVERIFIED: not executed in this stage).

Wraps the official generate_aggregated_attention (draft-based-approx-llm @ d70fabb) so the
selection signal is identical; only the selection step is replaced by ccpilot's pilot_select.
Requirements from the official repo: transformers==4.51.3, attn_implementation sdpa or
flash_attention_2 (eager returns attn weights and trips an assert in AttentionStoreFunction),
batch size 1 (AttentionCache asserts it).
GPU sync: torch.cuda.synchronize() before reading the clock on both sides of the forward.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

OFFICIAL_REPO = Path(__file__).resolve().parents[2] / "external" / "specpc"


def make_draft_score_fn(model_name="Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                        params=None, device="cuda:0", attn_implementation="flash_attention_2", model=None):
    """model: reuse an already-loaded draft (e.g. the one the official smoke loaded)."""
    if os.environ.get("CCPILOT_ALLOW_MODELS") != "1":
        raise RuntimeError("model loading disabled in this stage (set CCPILOT_ALLOW_MODELS=1 when approved)")
    import torch
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(OFFICIAL_REPO))
    from draft_approx_llm import SpecPCConfig  # official
    from draft_approx_llm.specpc.specpc_utils import generate_aggregated_attention  # official

    from ..compressors.specpc import SpecPCParams

    p = params or SpecPCParams()
    cfg = SpecPCConfig(max_capacity_prompt=10**9, window_size=p.window_size, pool_type=p.pool_type,
                       kernel_size=p.kernel_size, neighbor_tokens=p.neighbor_tokens,
                       reduction_type=p.reduction_type, lookahead_tokens=p.lookahead_tokens,
                       starting_layer_index=p.starting_layer_index, weighted_query=p.weighted_query)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.bfloat16,
                                                     attn_implementation=attn_implementation).to(device).eval()
        model.generation_config.pad_token_id = 151645  # as official eval.py::load_model (tokenizer eos)
    device = model.device
    # Draft lookahead kwargs exactly as official eval.py::run_sample passes them through
    # specpc_generate -> generate_aggregated_attention (draft generation_config otherwise applies,
    # incl. repetition_penalty=1.1). Identical in both flows: it is part of the compressor.
    gen = dict(do_sample=False, temperature=None, top_p=None)

    @torch.inference_mode()
    def score_fn(ids):
        """Returns the official aggregated attention for batch row 0 UNCHANGED (device and dtype
        as produced by AttentionCache, i.e. what official specpc_compress_prompt feeds to the
        selection function). The lookahead token(s) are kept for provenance only."""
        x = torch.tensor([ids], device=device)
        attn, look = generate_aggregated_attention(model, cfg, input_ids=x, attention_mask=torch.ones_like(x), **gen)
        torch.cuda.synchronize()
        score_fn.last_lookahead = look[0].tolist()
        return attn[0]

    score_fn.last_lookahead = None

    return score_fn
