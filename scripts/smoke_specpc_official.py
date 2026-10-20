"""SpecPC GPU smoke (NOT executed in the current stage; requires approval + GPU env).

A. Official path, untouched: official eval.py load_model / patch_model / InputProcessor /
   run_sample / LongBenchDataset.compute_metrics, official cfg yaml. The only addition is a
   pass-through hook that records what specpc_compress_prompt returns (selected input_ids).
B. Harness reproduction flow (SpecPCRepro, official selection function imported, official
   generation profile) on the same items, same loaded draft model.
Report: per item, A vs B equality of (1) full input ids, (2) selected ids, (3) generated text,
and A run twice for determinism. Any mismatch is reported, not fixed in place.

usage (after approval):
  CCPILOT_ALLOW_MODELS=1 CUDA_VISIBLE_DEVICES=<ids> python scripts/smoke_specpc_official.py \
      --cfg external/specpc/cfg/paper/specpc/longbench/qwen25_05b_14b/cmax_1024/hotpotqa.yaml --n 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "external/specpc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    if os.environ.get("CCPILOT_ALLOW_MODELS") != "1":
        sys.exit("model loading disabled in this stage")
    import torch
    import yaml

    os.chdir(OFFICIAL)  # official code opens dataset/longbench/*.json relative to repo root
    sys.path.insert(0, str(OFFICIAL))
    import draft_approx_llm.specpc.specpc as M
    from dataset import load_dataset as official_load_dataset
    from draft_approx_llm import DraftApproxLLMConfig, patch_model
    from eval import load_model, run_sample
    from transformers import AutoTokenizer, __version__ as tv
    from utils.tokenize_utils import InputProcessor

    sys.path.insert(0, str(ROOT))
    from ccpilot.adapters.specpc_draft import make_draft_score_fn
    from ccpilot.compressors.base import CompressRequest
    from ccpilot.compressors.specpc import SpecPCParams, SpecPCRepro, load_official_select
    from ccpilot.prompt import HFTokenizer, PromptSpec

    cfg = yaml.safe_load(open(ROOT / a.cfg if not Path(a.cfg).is_absolute() else a.cfg))
    sc = dict(cfg["sparse_config"]); sc.pop("sparse_type"); cap = sc["max_capacity_prompt"]
    rows = {json.loads(x)["_id"]: json.loads(x) for x in (ROOT / "data/raw/longbench/data/hotpotqa.jsonl").read_text().splitlines()}
    dev = json.loads((ROOT / "data/processed/splits.json").read_text())["dev"][: a.n]

    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"]["model_name_or_path"], trust_remote_code=True)
    t0 = time.perf_counter()
    draft = load_model(cfg["draft_model"], tokenizer)
    target = load_model(cfg["target_model"], tokenizer)
    t_load = time.perf_counter() - t0
    model = patch_model(target, draft, DraftApproxLLMConfig.from_dict(cfg["sparse_config"]))
    ds = official_load_dataset(tokenizer, **cfg["dataset"], model_cfg=cfg["target_model"])  # official dispatcher
    ds.dataset = [rows[s] for s in dev]  # local pinned rows instead of HF download; same fields
    ip = InputProcessor(model, tokenizer, ds)

    captured = []
    orig = M.specpc_compress_prompt

    def hook(*args, **kw):  # pass-through recorder
        out = orig(*args, **kw)
        captured.append(out["input_ids"][0].tolist())
        return out
    M.specpc_compress_prompt = hook

    sel_fn, origin = load_official_select()
    assert origin == "import"
    score_fn = make_draft_score_fn(model=draft, params=SpecPCParams(**{k: v for k, v in sc.items() if k != "max_capacity_prompt"}))
    repro = SpecPCRepro(score_fn, SpecPCParams(**{k: v for k, v in sc.items() if k != "max_capacity_prompt"}),
                        score_fn_id="draft:" + cfg["draft_model"]["model_name_or_path"], selector=sel_fn, selector_origin=origin)
    repro.total_budget = cap
    tok = HFTokenizer(ROOT / "data/raw/tokenizers/Qwen2.5-14B-Instruct/tokenizer.json", "Qwen/Qwen2.5-14B-Instruct@cf98f3b")
    report = dict(transformers=tv, torch=torch.__version__, cfg=a.cfg, t_load_s=t_load, cap=cap,
                  draft_device_map=str(getattr(draft, "hf_device_map", draft.device)),
                  target_device_map=str(getattr(target, "hf_device_map", target.device)),
                  official_select_dtype_device="official compress: AttentionCache output tensor as-is", items=[])
    for k, s in enumerate(ds):
        sid = dev[k]
        enc = ip.encode(s["input"], device=model.device)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t1 = time.perf_counter()
        out_a = run_sample(model, ip, s["input"], ds.max_new_tokens)
        torch.cuda.synchronize(); t_a = time.perf_counter() - t1
        peak_a = torch.cuda.max_memory_allocated() / 2**20
        sel_a = captured[-1]
        if k == 0:
            out_a2 = run_sample(model, ip, s["input"], ds.max_new_tokens)
            det_sel, det_out = captured[-1] == sel_a, out_a2 == out_a
        res = repro.compress(CompressRequest(rows[sid]["context"], rows[sid]["input"], cap, tok, PromptSpec()))
        x = torch.tensor([res.info["raw_ids"]], device=target.device)
        with torch.inference_mode():  # official profile: same kwargs as run_sample, unpatched target generate
            g = target.__class__.generate(target, input_ids=x, attention_mask=torch.ones_like(x),
                                          max_new_tokens=ds.max_new_tokens, do_sample=False, temperature=None, top_p=None)
        out_b = tokenizer.decode(g[0, x.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        report["items"].append(dict(
            sample_id=sid, full_ids_equal=enc["input_ids"][0].tolist() == tok.encode(PromptSpec().prefix_text() + rows[sid]["context"] + PromptSpec().suffix_text(rows[sid]["input"])),
            selected_equal=sel_a == res.info["raw_ids"], n_sel_a=len(sel_a), n_sel_b=len(res.info["raw_ids"]),
            selected_overlap=len(set(map(tuple, enumerate(sel_a))) & set(map(tuple, enumerate(res.info["raw_ids"])))) / max(1, len(sel_a)),
            output_a=out_a, output_b=out_b, output_equal=out_a.strip() == out_b.strip(), t_official_s=t_a,
            peak_mem_official_mb=peak_a, harness_score_dtype=res.info.get("score_dtype"),
            harness_select_device=res.info.get("select_device"),
            suffix_kept=res.info["suffix_kept"], suffix_len=res.info["suffix_len"],
            prefix_kept=res.info["prefix_kept"], prefix_len=res.info["prefix_len"]))
        if k == 0:
            report["items"][-1]["official_rerun_selected_equal"] = det_sel
            report["items"][-1]["official_rerun_output_equal"] = det_out
    metrics, _ = ds.compute_metrics(predictions=[i["output_a"] for i in report["items"]],
                                    references=[{"answers": rows[s]["answers"], "all_classes": None} for s in dev])
    report["official_score_on_smoke_items"] = metrics
    out = ROOT / "runs" / "real" / f"smoke-specpc-official-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
