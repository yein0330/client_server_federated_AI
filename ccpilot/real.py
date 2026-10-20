"""Real-model pilot: compression artifacts are produced ONCE and saved, then answered by any number
of answer models (S, L) through the payload/process boundary. Diagnostic inputs are built from the
saved artifacts + annotations before any answer exists."""
from __future__ import annotations

import hashlib
import json
import struct
import time

from . import payload as P
from .compressors.base import CompressRequest
from .evidence import score_preservation
from .manual import evaluate_manual
from .records import SCHEMA_VERSION
from .textutil import answer_contained, best_f1, exact_match, format_prediction


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()[:16]


def make_artifact(item, comp, budget, tok, prompt, compressor_id: str) -> dict:
    q = item["question"]
    fixed = prompt.fixed_len(tok, q)
    full_ids = tok.encode(item["context"])
    if comp.method_id == "full":
        res = comp.compress(CompressRequest(item["context"], None, len(full_ids), tok, prompt))
    else:
        res = comp.compress(CompressRequest(item["context"], q if comp.uses_question else None, budget - fixed, tok, prompt))
    info = {k: v for k, v in res.info.items() if k not in ("raw_ids",)}
    return dict(artifact_id=f"{item['sample_id']}#{comp.method_id}#{budget}", sample_id=item["sample_id"],
                method=comp.method_id, budget=budget, question_access=comp.uses_question if comp.method_id != "full" else False,
                compressor_id=compressor_id, context_ids=res.context_ids, context_ids_sha=ids_sha(res.context_ids),
                kept_intervals=res.kept_intervals, orig_context_tokens=len(full_ids), fixed_tokens=fixed,
                pre_fit_context_tokens=res.pre_fit_context_len, fit_iterations=res.fit_iterations,
                fit_truncated=res.fit_truncated, fit_failed=res.fit_failed,
                t_compress_s=res.t_compress_s, t_fit_s=res.t_fit_s, info=info)


def answer(item, art, *, condition, server, tok, prompt, run_id, cfg_id, split, data_revision, tokenizer_ref,
           auto_ann=None, manual_ann=None, backend="hf", method_label=None) -> dict:
    q = item["question"]
    budget = art["budget"]
    rec = dict(
        schema_version=SCHEMA_VERSION, run_id=run_id, config_id=cfg_id, split=split, sample_id=item["sample_id"],
        hotpot_id=(item["hotpot"] or {}).get("hotpot_id"), data_revision=data_revision,
        method=method_label or art["method"], question_access=art["question_access"], question_in_server_input=True,
        flow="pilot", gen_profile=None, budget_nominal=budget, orig_context_tokens=art["orig_context_tokens"],
        orig_server_input_tokens=art["orig_context_tokens"] + art["fixed_tokens"], fixed_tokens=art["fixed_tokens"],
        context_budget=None if budget is None else budget - art["fixed_tokens"],
        pre_fit_context_tokens=art["pre_fit_context_tokens"], final_context_tokens=len(art["context_ids"]),
        final_server_input_tokens=0, budget_ok=False, budget_slack=None, fit_iterations=art["fit_iterations"],
        fit_truncated=art["fit_truncated"], fit_failed=art["fit_failed"], payload_bytes=0,
        payload_text_bytes=len(tok.decode(art["context_ids"]).encode()), wire_bytes=None,
        t_compress_s=art["t_compress_s"], t_fit_s=art["t_fit_s"], t_serialize_s=0.0, t_transfer_s=None,
        t_ipc_roundtrip_s=0.0, t_server_s=0.0, server_input_sha=None, n_out_tokens=None, prediction_raw=None,
        prediction=None, answers=item["answers"], qa_f1=None, em=None, contains=None, evidence=None, error=None,
        excluded=None, backend=backend, condition=condition, answer_model=None, compressor_id=art["compressor_id"],
        compression_artifact=art["artifact_id"], context_ids_sha=art["context_ids_sha"], server_input_check=None,
        gpu_peak_mem_mb=None, hit_max_new_tokens=None, evidence_manual_ai=None, em_semantic_review=None, edit=None)
    try:
        header = dict(v=1, assembly="server_template", prompt_spec=prompt.spec_id, tokenizer=tokenizer_ref,
                      tok_sha=tok.sha, req=P.request_id(run_id + condition, item["sample_id"]), question=q)
        t0 = time.perf_counter()
        buf = P.encode(header, art["context_ids"])
        rec["t_serialize_s"] = time.perf_counter() - t0
        rec["payload_bytes"] = len(buf)
        out = server(buf)
        rec["t_ipc_roundtrip_s"] = out.get("t_roundtrip_s", 0.0)
        if out.get("error"):
            rec["error"] = "server: " + out["error"]
            return rec
        expect = prompt.assemble(tok, art["context_ids"], q)
        n_in = out["server_input_len"]
        rec.update(gen_profile=out.get("gen_profile"), final_server_input_tokens=n_in, server_input_sha=out["input_sha"],
                   server_input_check=(out["input_sha"] == ids_sha(expect) and n_in == len(expect)),
                   t_server_s=out["t_prefill_decode_s"], n_out_tokens=out["n_out_tokens"], prediction_raw=out["answer"],
                   answer_model=out.get("answer_model"), gpu_peak_mem_mb=out.get("gpu_peak_mem_mb"),
                   hit_max_new_tokens=out.get("hit_max_new_tokens"))
        rec["budget_ok"] = True if budget is None else n_in <= budget
        rec["budget_slack"] = None if budget is None else budget - n_in
        pred = format_prediction(out["answer"])
        rec.update(prediction=pred, qa_f1=best_f1(pred, item["answers"]), em=exact_match(pred, item["answers"]),
                   contains=answer_contained(pred, item["answers"]))
        if auto_ann is not None and art["kept_intervals"] is not None:
            rec["evidence"] = score_preservation(item["context"], auto_ann, art["kept_intervals"])
        if manual_ann is not None and art["kept_intervals"] is not None:
            rec["evidence_manual_ai"] = evaluate_manual(item["context"], manual_ann, art["kept_intervals"],
                                                        tok.decode(art["context_ids"]))
    except Exception as e:  # recorded per item
        rec["error"] = f"client: {type(e).__name__}: {e}"
    return rec


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
