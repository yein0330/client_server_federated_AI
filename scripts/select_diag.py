"""Ordered annotation queue for the restoration-diagnostic sample (rules v1 section 6).
Uses only eval sample ids, HotpotQA type and a fixed seed."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261020
TARGET = {"bridge": 10, "comparison": 2}

splits = json.loads((ROOT / "data/processed/splits.json").read_text())
types = {r["sample_id"]: r["type"] for r in map(json.loads, (ROOT / "data/processed/manifest.jsonl").read_text().splitlines())}
queue = {}
for t in sorted(TARGET):
    ids = sorted(s for s in splits["eval"] if types[s] == t)
    random.Random(f"{SEED}-{t}").shuffle(ids)
    queue[t] = ids
out = dict(seed=SEED, target_eligible=TARGET, queue=queue)
(ROOT / "data/annotations/diag_queue.json").write_text(json.dumps(out, indent=1))
for t, ids in queue.items():
    print(t, len(ids), " ".join(ids[:16]))
