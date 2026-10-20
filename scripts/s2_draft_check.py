"""S2: draft-only check. Official aggregated attention -> official selection on (a) the tensor as
produced (GPU, model dtype), (b) CPU float64 copy (old wrapper behaviour), (c) numpy port."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
from ccpilot.adapters.specpc_draft import make_draft_score_fn
from ccpilot.compressors.specpc import SpecPCParams, load_official_select, official_select, pilot_select
from ccpilot.prompt import HFTokenizer, PromptSpec

rows = {json.loads(x)["_id"]: json.loads(x) for x in (ROOT / "data/raw/longbench/data/hotpotqa.jsonl").read_text().splitlines()}
ids = json.loads((ROOT / "data/processed/splits.json").read_text())["dev"][:3]
tok = HFTokenizer(ROOT / "data/raw/tokenizers/Qwen2.5-14B-Instruct/tokenizer.json", "qwen25")
P, p = PromptSpec(), SpecPCParams()
sel, origin = load_official_select()
t0 = time.perf_counter(); sfn = make_draft_score_fn(device="cuda:0"); t_load = time.perf_counter() - t0
out = dict(selector_origin=origin, t_load_s=t_load, items=[])
for sid in ids:
    r = rows[sid]
    full_text = P.prefix_text() + r["context"] + P.suffix_text(r["input"])
    full = tok.encode(full_text)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    attn = sfn(full); torch.cuda.synchronize(); t_score = time.perf_counter() - t1
    a = sel(attn, 2048, p); info_a = dict(sel.last)
    b = sel(attn.float().cpu().double(), 2048, p); info_b = dict(sel.last)
    c = official_select(attn.float().cpu().numpy().astype(np.float64), 2048, p)
    x = attn.float().cpu().numpy()
    ties = int(len(x) - len(np.unique(x)))
    jac = lambda u, v: len(set(u) & set(v)) / len(set(u) | set(v))
    out["items"].append(dict(sample_id=sid, L=len(full), dtype=str(attn.dtype), device=str(attn.device),
        n_duplicate_raw_scores=ties, t_score_s=t_score, lookahead=sfn.last_lookahead,
        gpu_native=info_a, cpu64=info_b,
        equal_gpu_vs_cpu64=bool(np.array_equal(a, b)), jaccard_gpu_vs_cpu64=jac(a, b),
        equal_gpu_vs_port=bool(np.array_equal(a, c)), jaccard_gpu_vs_port=jac(a, c),
        peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20))
print(json.dumps(out, indent=1))
d = ROOT / "runs/real/s2_draft_check"; d.mkdir(parents=True, exist_ok=True)
(d / "report.json").write_text(json.dumps(out, indent=1))
