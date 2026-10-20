import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOK_PATH = ROOT / "data/raw/tokenizers/Qwen2.5-14B-Instruct/tokenizer.json"
LB_PATH = ROOT / "data/raw/longbench/data/hotpotqa.jsonl"
PROC = ROOT / "data/processed"


@pytest.fixture(scope="session")
def tok():
    from ccpilot.prompt import HFTokenizer

    if not TOK_PATH.exists():
        pytest.skip("tokenizer.json not downloaded")
    return HFTokenizer(TOK_PATH, "Qwen/Qwen2.5-14B-Instruct@cf98f3b")


@pytest.fixture(scope="session")
def lb_items():
    if not LB_PATH.exists():
        pytest.skip("LongBench data not downloaded")
    return [json.loads(x) for x in LB_PATH.read_text().splitlines()]


@pytest.fixture(scope="session")
def joined():
    from ccpilot.data import join_items, load_hotpot_dev, load_longbench

    hp = ROOT / "data/raw/hotpotqa/distractor/validation-00000-of-00001.parquet"
    if not (LB_PATH.exists() and hp.exists()):
        pytest.skip("data not downloaded")
    return join_items(load_longbench(LB_PATH), load_hotpot_dev(hp))


@pytest.fixture(scope="session")
def anns():
    p = PROC / "annotations_auto.jsonl"
    if not p.exists():
        pytest.skip("run scripts/prepare_data.py first")
    return {a["sample_id"]: a for a in map(json.loads, p.read_text().splitlines())}


def server_args(gen_profile="pilot_common", **kw):
    return dict(tokenizer_path=str(TOK_PATH), tokenizer_name="Qwen/Qwen2.5-14B-Instruct@cf98f3b",
                max_input_tokens=32736, gen_profile=gen_profile, **kw)


@pytest.fixture(scope="session")
def manual_dev():
    from ccpilot.data import load_longbench
    from ccpilot.manual import load_manual

    p = ROOT / "data/annotations/manual_v1_dev.json"
    if not (p.exists() and LB_PATH.exists()):
        pytest.skip("manual annotations / data missing")
    ctx = {r["_id"]: r["context"] for r in load_longbench(LB_PATH)}
    return load_manual(p, ctx)
