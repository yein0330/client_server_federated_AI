"""Server prompt assembly with segmented tokenization.

Protocol (pilot flow, fixed for every method):
    ids = enc(PREFIX) + context_ids + enc(SUFFIX(question))
PREFIX/SUFFIX are the Qwen2.5 chat template around the LongBench v1 hotpotqa prompt
(SpecPC dataset2prompt.json, identical to LongBench). Segmented tokenization differs
from tokenizing the joined string at the context boundaries in 73/200 items; the
official SpecPC eval tokenizes the joined string (documented deviation).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

LONGBENCH_HOTPOTQA = (
    "Answer the question based on the given passages. Only give me the answer and do not "
    "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
    "question based on the given passages. Only give me the answer and do not output any "
    "other words.\n\nQuestion: {input}\nAnswer:"
)
# System prompt used by SpecPC's get_conv_template for Qwen2.5 (also the template default).
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def qwen_chat(user: str, system: str = QWEN_SYSTEM) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")


class TokenizerLike(Protocol):
    name: str
    sha: str
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class HFTokenizer:
    """`tokenizers` backend loaded from a local tokenizer.json (no model weights)."""

    def __init__(self, path: str | Path, name: str):
        from tokenizers import Tokenizer

        path = Path(path)
        self._tok = Tokenizer.from_file(str(path))
        self.name = name
        self.sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        self.vocab_size = self._tok.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self._tok.encode(text, add_special_tokens=False).offsets


class MockTokenizer:
    """Whitespace-ish tokenizer for model-free tests: each word/punct/whitespace run is a token."""

    import re as _re

    _pat = _re.compile(r"\s+|\w+|[^\w\s]")

    def __init__(self):
        self.name, self.sha, self._vocab, self._inv = "mock", "mock", {}, []
        self.vocab_size = 1 << 20

    def _id(self, t: str) -> int:
        if t not in self._vocab:
            self._vocab[t] = len(self._inv)
            self._inv.append(t)
        return self._vocab[t]

    def encode(self, text: str) -> list[int]:
        return [self._id(m.group(0)) for m in self._pat.finditer(text)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._inv[i] for i in ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in self._pat.finditer(text)]


@dataclass
class PromptSpec:
    template: str = LONGBENCH_HOTPOTQA
    system: str = QWEN_SYSTEM
    spec_id: str = "lb1_hotpotqa_qwen25chat_seg_v1"
    _cache: dict = field(default_factory=dict, repr=False)

    def prefix_text(self) -> str:
        return qwen_chat(self.template, self.system).split("{context}")[0]

    def suffix_text(self, question: str) -> str:
        return qwen_chat(self.template, self.system).split("{context}")[1].replace("{input}", question)

    def prefix_ids(self, tok: TokenizerLike) -> list[int]:
        key = ("prefix", tok.name, tok.sha)
        if key not in self._cache:
            self._cache[key] = tok.encode(self.prefix_text())
        return self._cache[key]

    def suffix_ids(self, tok: TokenizerLike, question: str) -> list[int]:
        return tok.encode(self.suffix_text(question))

    def fixed_len(self, tok: TokenizerLike, question: str) -> int:
        return len(self.prefix_ids(tok)) + len(self.suffix_ids(tok, question))

    def assemble(self, tok: TokenizerLike, context_ids: list[int], question: str) -> list[int]:
        return self.prefix_ids(tok) + list(context_ids) + self.suffix_ids(tok, question)


# Question-only variant (no context). Minimal edit of LONGBENCH_HOTPOTQA:
#  - removed "based on the given passages" (x2), "The following are given passages.\n{context}\n\n"
#    and the duplicated instruction; kept "Only give me the answer and do not output any other words."
#  - same chat template/system prompt, same "Question: {input}\nAnswer:" ending.
#  The {context} slot is kept (always empty) so segmented assembly is unchanged.
QUESTION_ONLY = ("Answer the question. Only give me the answer and do not output any other words.\n\n"
                 "{context}Question: {input}\nAnswer:")


def question_only_spec() -> "PromptSpec":
    return PromptSpec(template=QUESTION_ONLY, spec_id="qonly_qwen25chat_seg_v1")
