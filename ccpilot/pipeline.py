"""Client-side runner: compress -> serialize -> server process -> score -> record."""
from __future__ import annotations

import hashlib
import json
import time

from . import payload as P
from .compressors.base import CompressRequest, Compressor
from .evidence import score_preservation
from .prompt import PromptSpec, TokenizerLike
from .records import SCHEMA_VERSION, JsonlWriter
from .textutil import answer_contained, best_f1, exact_match, format_prediction


def config_id(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run_cell(item: dict, ann: dict | None, comp: Compressor, budget: int | None, *, tok: TokenizerLike,
             prompt: PromptSpec, server, run_id: str, cfg_id: str, split: str, data_revision: str,
             tokenizer_ref: str, question_access: bool | None = None) -> tuple[dict, dict]:
    qa = comp.uses_question if question_access is None else question_access
    if qa and not comp.uses_question:
        raise ValueError(f"{comp.method_id} cannot use the question")
    q = item["question"]
    ctx_ids_full = tok.encode(item["context"])
    fixed = prompt.fixed_len(tok, q)
    ctx_budget = None if budget is None else budget - fixed
    rec = dict(
        schema_version=SCHEMA_VERSION, run_id=run_id, config_id=cfg_id, split=split,
        sample_id=item["sample_id"], hotpot_id=(item["hotpot"] or {}).get("hotpot_id"),
        data_revision=data_revision, method=comp.method_id, question_access=qa,
        question_in_server_input=True, flow="official_repro" if comp.method_id == "specpc_repro" else "pilot",
        gen_profile=None,
        budget_nominal=budget, orig_context_tokens=len(ctx_ids_full),
        orig_server_input_tokens=len(ctx_ids_full) + fixed, fixed_tokens=fixed, context_budget=ctx_budget,
        pre_fit_context_tokens=0, final_context_tokens=0, final_server_input_tokens=0, budget_ok=False,
        budget_slack=None, fit_iterations=0, fit_truncated=False, fit_failed=False, payload_bytes=0,
        payload_text_bytes=None, wire_bytes=None, t_compress_s=0.0, t_fit_s=0.0, t_serialize_s=0.0,
        t_transfer_s=None, t_ipc_roundtrip_s=0.0, t_server_s=0.0, server_input_sha=None, n_out_tokens=None,
        prediction_raw=None, prediction=None, answers=item["answers"], qa_f1=None, em=None, contains=None,
        evidence=None, error=None, excluded=None, backend="mock", condition=None, answer_model=None,
        compressor_id=None, compression_artifact=None, context_ids_sha=None, server_input_check=None,
        gpu_peak_mem_mb=None, hit_max_new_tokens=None, evidence_manual_ai=None, em_semantic_review=None,
    )
    prov = dict(run_id=run_id, sample_id=item["sample_id"], method=comp.method_id, budget=budget, flow=rec["flow"])
    if ctx_budget is not None and ctx_budget <= 0:
        rec["excluded"] = "R-budget-below-fixed"
        return rec, prov
    try:
        if comp.method_id == "specpc_repro":
            comp.total_budget = budget
        req = CompressRequest(item["context"], q if qa else None,
                              ctx_budget if ctx_budget is not None else len(ctx_ids_full), tok, prompt)
        res = comp.compress(req)
        raw = rec["flow"] == "official_repro"
        body = res.info["raw_ids"] if raw else res.context_ids
        header = dict(v=1, assembly="client_raw" if raw else "server_template",
                      prompt_spec="" if raw else prompt.spec_id, tokenizer=tokenizer_ref, tok_sha=tok.sha,
                      req=P.request_id(run_id, item["sample_id"]), question="" if raw else q)
        t0 = time.perf_counter()
        buf = P.encode(header, body)
        rec["t_serialize_s"] = time.perf_counter() - t0
        rec.update(pre_fit_context_tokens=res.pre_fit_context_len, final_context_tokens=len(res.context_ids),
                   fit_iterations=res.fit_iterations, fit_truncated=res.fit_truncated, fit_failed=res.fit_failed,
                   payload_bytes=len(buf), payload_text_bytes=len(tok.decode(res.context_ids).encode()),
                   t_compress_s=res.t_compress_s, t_fit_s=res.t_fit_s)
        prov.update(kept_intervals=res.kept_intervals, info={k: v for k, v in res.info.items() if k != "raw_ids"})
        out = server(buf)
        rec["t_ipc_roundtrip_s"] = out.get("t_roundtrip_s", 0.0)
        if out.get("error"):
            rec["error"] = "server: " + out["error"]
            return rec, prov
        n_in = out["server_input_len"]
        rec.update(gen_profile=out.get("gen_profile"), final_server_input_tokens=n_in, server_input_sha=out["input_sha"],
                   t_server_s=out["t_prefill_decode_s"], n_out_tokens=out["n_out_tokens"],
                   prediction_raw=out["answer"])
        rec["budget_ok"] = True if budget is None else n_in <= budget
        rec["budget_slack"] = None if budget is None else budget - n_in
        pred = format_prediction(out["answer"])
        rec.update(prediction=pred, qa_f1=best_f1(pred, item["answers"]), em=exact_match(pred, item["answers"]),
                   contains=answer_contained(pred, item["answers"]))
        if ann is not None and res.kept_intervals is not None:
            rec["evidence"] = score_preservation(item["context"], ann, res.kept_intervals)
    except Exception as e:
        rec["error"] = f"client: {type(e).__name__}: {e}"
    return rec, prov


def run(items, anns, compressors, budgets, *, tok, prompt, servers, run_id, cfg, split, data_revision,
        tokenizer_ref, out_dir):
    cid = config_id(cfg)
    res_w, prov_w = JsonlWriter(f"{out_dir}/results.jsonl"), JsonlWriter(f"{out_dir}/provenance.jsonl")
    try:
        for it in items:
            for comp in compressors:
                for b in ([None] if comp.method_id == "full" else budgets):
                    rec, prov = run_cell(it, anns.get(it["sample_id"]), comp, b, tok=tok, prompt=prompt,
                                         server=servers["official_repro" if comp.method_id == "specpc_repro" else "pilot"],
                                         run_id=run_id, cfg_id=cid, split=split,
                                         data_revision=data_revision, tokenizer_ref=tokenizer_ref)
                    res_w.write(rec, validate_result=True)
                    prov_w.write(prov)
    finally:
        res_w.close()
        prov_w.close()
