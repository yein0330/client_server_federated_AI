"""Server worker process. Imports only payload/prompt/textutil - never the dataset.

Backends:
  mock : model-free. Answers from its own input with a fixed heuristic (no gold access),
         so plumbing/recording can be exercised end to end.
  hf   : Qwen2.5-14B-Instruct greedy decoding (UNVERIFIED; guarded by CCPILOT_ALLOW_MODELS=1).
Protocol: frames on stdin/stdout (see server.py). First frame out = {"t_load_s": ...}.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import time

from ccpilot import payload as P
from ccpilot.prompt import HFTokenizer, PromptSpec
from ccpilot.textutil import word_tokens

# Generation profiles (fixed per server process; recorded in every result row).
#  official_repro: exactly the kwargs of official eval.py::run_sample (+ LongBench maxlen 32);
#                  everything else comes from the 14B generation_config (repetition_penalty=1.05,
#                  top_k=20 - inert under greedy). pad_token_id = tokenizer eos, as official load_model.
#  pilot_common  : one explicit setting shared by ALL pilot comparison conditions.
GEN_PROFILES = {
    "official_repro": dict(max_new_tokens=32, do_sample=False, temperature=None, top_p=None),
    "pilot_common": dict(max_new_tokens=32, do_sample=False, temperature=None, top_p=None, top_k=None,
                         repetition_penalty=1.0),
}


def _read(f):
    n = f.read(4)
    if len(n) < 4:
        return None
    return f.read(struct.unpack("<I", n)[0])


def _write(f, obj):
    b = json.dumps(obj).encode()
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


class MockBackend:
    def __init__(self, args):
        self.gen = GEN_PROFILES[args["gen_profile"]]
        # Real server tokenizer (local tokenizer.json, no weights) so IDs round-trip exactly.
        self.tok = HFTokenizer(args["tokenizer_path"], args["tokenizer_name"])
        self.debug_visible = bool(args.get("debug_return_visible_text"))

    def generate(self, ids):
        t0 = time.perf_counter()
        text = self.tok.decode(ids)
        q = text.rsplit("Question:", 1)[-1].split("\n")[0]
        if re.match(r"\s*(is|are|was|were|do|does|did|can|could|has|have)\b", q, re.I):
            ans = "yes"
        else:
            titles = re.findall(r"Passage \d+:\n([^\n]+)", text)
            qs = set(word_tokens(q))
            ans = max(titles, key=lambda t: (len(qs & set(word_tokens(t))), -titles.index(t)), default="unknown")
        out = dict(answer=ans, n_out_tokens=len(self.tok.encode(ans)), t_prefill_decode_s=time.perf_counter() - t0)
        if self.debug_visible:
            out["visible_text"] = text
        return out


class HFBackend:
    def __init__(self, args):
        if os.environ.get("CCPILOT_ALLOW_MODELS") != "1":
            raise RuntimeError("model loading disabled in this stage")
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.tok = HFTokenizer(args["tokenizer_path"], args["tokenizer_name"])
        self.gen = GEN_PROFILES[args["gen_profile"]]
        self.model = AutoModelForCausalLM.from_pretrained(
            args["model"], revision=args["revision"], torch_dtype=torch.bfloat16,
            attn_implementation=args.get("attn_implementation", "flash_attention_2"), device_map="auto").eval()
        self.eos = [151645, 151643]

    def generate(self, ids):
        torch = self.torch
        x = torch.tensor([ids], device=self.model.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), pad_token_id=151645, **self.gen)
        torch.cuda.synchronize()
        new = out[0, len(ids):].tolist()
        while new and new[-1] in self.eos:
            new.pop()
        return dict(answer=self.tok.decode(new).replace("<|im_end|>", "").strip(), n_out_tokens=len(new),
                    t_prefill_decode_s=time.perf_counter() - t0)


def main():
    backend, args = sys.argv[1], json.loads(sys.argv[2])
    t0 = time.perf_counter()
    be = {"mock": MockBackend, "hf": HFBackend}[backend](args)
    prompt = PromptSpec()
    max_input = int(args["max_input_tokens"])
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    _write(stdout, {"t_load_s": time.perf_counter() - t0})
    while (buf := _read(stdin)) is not None:
        try:
            t1 = time.perf_counter()
            h, ctx_ids = P.decode(buf, vocab_size=be.tok.vocab_size, max_ctx=max_input)
            if h["tok_sha"] != be.tok.sha:
                raise P.BoundaryError("tokenizer mismatch")
            if h["assembly"] == "server_template":
                if h["prompt_spec"] != prompt.spec_id:
                    raise P.BoundaryError("prompt_spec mismatch")
                ids = prompt.assemble(be.tok, ctx_ids, h["question"])
            else:
                ids = ctx_ids
            if len(ids) > max_input:
                raise P.BoundaryError(f"server input {len(ids)} > limit {max_input}")
            t_asm = time.perf_counter() - t1
            r = be.generate(ids)
            r.update(req=h["req"], gen_profile=args["gen_profile"], server_input_len=len(ids), t_server_assemble_s=t_asm,
                     input_sha=hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()[:16], error=None)
        except Exception as e:  # reported per item, never crashes the loop
            r = dict(error=f"{type(e).__name__}: {e}")
        _write(stdout, r)


if __name__ == "__main__":
    main()
