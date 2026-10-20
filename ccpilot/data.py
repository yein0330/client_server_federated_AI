"""LongBench v1 hotpotqa loading, HotpotQA join, and frozen dev/eval selection."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"  # THUDM/LongBench (HF dataset)
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"  # hotpotqa/hotpot_qa (HF dataset)


def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_longbench(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_hotpot_dev(parquet_path: str | Path) -> dict[str, dict]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    out = {}
    for r in df.to_dict("records"):
        out[r["question"].strip()] = dict(
            hotpot_id=r["id"], answer=r["answer"], type=r["type"], level=r["level"],
            supporting_facts=list(zip(list(r["supporting_facts"]["title"]),
                                      [int(x) for x in r["supporting_facts"]["sent_id"]])),
            paragraphs={t: list(s) for t, s in zip(r["context"]["title"], r["context"]["sentences"])},
        )
    return out


def join_items(longbench: list[dict], hotpot: dict[str, dict]) -> list[dict]:
    items = []
    for row in longbench:
        h = hotpot.get(row["input"].strip())
        items.append(dict(
            sample_id=row["_id"], question=row["input"], context=row["context"],
            answers=list(row["answers"]), lb_length_words=row["length"],
            hotpot=h,  # None if unmatched; never given to compressors
        ))
    return items


def select_splits(sample_ids: list[str], strata: dict[str, str], n_dev: int, n_eval: int,
                  seed: int) -> dict[str, list[str]]:
    """Stratified (by HotpotQA type) seeded draw, independent of any model output.
    Proportional allocation with largest-remainder rounding; dev drawn first."""
    rng = random.Random(seed)
    by: dict[str, list[str]] = {}
    for sid in sorted(sample_ids):
        by.setdefault(strata.get(sid, "unknown"), []).append(sid)
    for v in by.values():
        rng.shuffle(v)
    total = len(sample_ids)

    def alloc(n):
        raw = {k: n * len(v) / total for k, v in by.items()}
        base = {k: int(x) for k, x in raw.items()}
        rest = n - sum(base.values())
        for k in sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))[:rest]:
            base[k] += 1
        return base

    a_dev, a_eval = alloc(n_dev), alloc(n_eval)
    dev, ev = [], []
    for k in sorted(by):
        dev += by[k][: a_dev[k]]
        ev += by[k][a_dev[k] : a_dev[k] + a_eval[k]]
    return {"dev": sorted(dev), "eval": sorted(ev)}
