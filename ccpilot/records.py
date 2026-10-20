"""Per-item result record (one JSONL row per sample x method x budget x question-access).

Server-visible data and evaluation-only data are stored in separate files:
  results.jsonl     : RESULT_FIELDS (no provenance)
  provenance.jsonl  : kept intervals / selected positions / scores (eval-only sidecar)
"""
from __future__ import annotations

import json
from pathlib import Path

# name: (type(s), nullable, description)
RESULT_FIELDS: dict[str, tuple[tuple, bool, str]] = {
    "schema_version": ((int,), False, "record schema version"),
    "run_id": ((str,), False, "run identifier"),
    "config_id": ((str,), False, "sha256[:12] of the frozen run config"),
    "split": ((str,), False, "dev | eval | diag"),
    "sample_id": ((str,), False, "LongBench _id"),
    "hotpot_id": ((str,), True, "HotpotQA id (join key), null if unmatched"),
    "data_revision": ((str,), False, "LongBench HF revision + file sha256[:12]"),
    "method": ((str,), False, "full | specpc_pilot | specpc_repro | llmlingua2 | sent_bm25 | sent_uniform | diag_*"),
    "question_access": ((bool,), False, "selection signal could depend on the question"),
    "question_in_server_input": ((bool,), False, "question appended to server input"),
    "flow": ((str,), False, "pilot | official_repro"),
    "gen_profile": ((str,), True, "server generation profile (server_worker.GEN_PROFILES)"),
    "budget_nominal": ((int,), True, "final server-input token budget, null for full"),
    "orig_context_tokens": ((int,), False, "server-tokenizer tokens of full context"),
    "orig_server_input_tokens": ((int,), False, "server input length with full context"),
    "fixed_tokens": ((int,), False, "prefix+suffix (template, instruction, question, specials)"),
    "context_budget": ((int,), True, "budget_nominal - fixed_tokens"),
    "pre_fit_context_tokens": ((int,), False, "context tokens before budget fitting"),
    "final_context_tokens": ((int,), False, "context tokens transmitted"),
    "final_server_input_tokens": ((int,), False, "measured by the server after assembly"),
    "budget_ok": ((bool,), False, "final_server_input_tokens <= budget_nominal (or full)"),
    "budget_slack": ((int,), True, "budget_nominal - final_server_input_tokens"),
    "fit_iterations": ((int,), False, "compression re-calls / unit drops during fitting"),
    "fit_truncated": ((bool,), False, "hard tail truncation used"),
    "fit_failed": ((bool,), False, "could not meet budget"),
    "payload_bytes": ((int,), False, "serialized payload size (excl. transport framing)"),
    "payload_text_bytes": ((int,), True, "UTF-8 bytes of decoded context (reference only)"),
    "wire_bytes": ((int,), True, "measured on-wire bytes; null = not measured"),
    "t_compress_s": ((float,), False, "compressor time (excl. model load)"),
    "t_fit_s": ((float,), False, "budget fitting time"),
    "t_serialize_s": ((float,), False, "payload encode time"),
    "t_transfer_s": ((float,), True, "network transfer; null = not measured (estimate separately)"),
    "t_ipc_roundtrip_s": ((float,), False, "local process round-trip incl. server compute"),
    "t_server_s": ((float,), False, "server prefill+decode (GPU-synchronised for hf)"),
    "server_input_sha": ((str,), True, "sha256[:16] of server input IDs"),
    "n_out_tokens": ((int,), True, "generated tokens"),
    "prediction_raw": ((str,), True, "server answer"),
    "prediction": ((str,), True, "after format_prediction"),
    "answers": ((list,), False, "gold answers"),
    "qa_f1": ((float,), True, "LongBench qa_f1 (max over answers)"),
    "em": ((bool,), True, "normalised exact match (primary correctness)"),
    "contains": ((bool,), True, "normalised gold contained in prediction (secondary)"),
    "evidence": ((dict,), True, "preservation summary (see evidence.score_preservation)"),
    "backend": ((str,), True, "mock | hf"),
    "condition": ((str,), True, "e.g. S-full, L-compressed, L-diag_restore"),
    "answer_model": ((str,), True, "answer model id@revision (server-reported)"),
    "compressor_id": ((str,), True, "compressor id incl. model id@revision / selector origin"),
    "compression_artifact": ((str,), True, "artifact id (file#sample#method#budget)"),
    "context_ids_sha": ((str,), True, "sha256[:16] of transmitted context token IDs"),
    "server_input_check": ((bool,), True, "server input sha == client-side sha(prefix+payload ctx+suffix)"),
    "gpu_peak_mem_mb": ((float,), True, "torch max_memory_allocated during this request (answer model)"),
    "hit_max_new_tokens": ((bool,), True, "generation stopped by max_new_tokens"),
    "evidence_manual_ai": ((dict,), True, "AI-written annotation (not human-verified): char rate + key terms"),
    "em_semantic_review": ((str,), True, "manual (AI) review label for EM-ambiguous answers"),
    "edit": ((dict,), True, "diagnostic edit: removed/added source spans, token counts, rule version"),
    "error": ((str,), True, "error message"),
    "excluded": ((str,), True, "exclusion reason (rule id)"),
}
SCHEMA_VERSION = 1


def validate(rec: dict) -> list[str]:
    errs = []
    for k, (types, nullable, _) in RESULT_FIELDS.items():
        if k not in rec:
            errs.append(f"missing {k}")
            continue
        v = rec[k]
        if v is None:
            if not nullable:
                errs.append(f"null {k}")
        elif not isinstance(v, types) or (isinstance(v, bool) and bool not in types):
            errs.append(f"type {k}: {type(v).__name__}")
    extra = set(rec) - set(RESULT_FIELDS)
    if extra:
        errs.append(f"extra fields {sorted(extra)}")
    return errs


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a")

    def write(self, rec: dict, validate_result=False):
        if validate_result:
            errs = validate(rec)
            if errs:
                raise ValueError(f"invalid record: {errs}")
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
