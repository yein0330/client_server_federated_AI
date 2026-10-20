"""Client->server payload: the ONLY object that crosses the transmission boundary.

Wire format v1 (bytes):
    b"CCP1" | uint32_le header_len | header (UTF-8 JSON, sorted keys, no spaces) | body
    body = context token IDs as uint32 little-endian (len = header.n_ctx)
Header keys (whitelist, all required):
    v            protocol version (1)
    assembly     "server_template": server wraps context with its own PromptSpec
                 "client_raw"     : body is the complete server input (SpecPC repro flow)
    prompt_spec  PromptSpec.spec_id the server must use ("" for client_raw)
    tokenizer    "<hf repo>@<revision>"
    tok_sha      sha256[:16] of tokenizer.json
    req          opaque request id (sha256 of run_id/sample_id, 16 hex) - not the sample id
    question     question text (UTF-8) or "" for client_raw
    n_ctx        number of uint32 IDs in body
Attention mask: not transmitted (batch size 1, no padding -> all ones; the server builds it).
Positions: not transmitted; the server re-indexes contiguously (as the official SpecPC flow).
Not transmitted (by construction, checked by validate_header / tests): original context,
dropped tokens, selection indices, scores, method id, sample id, gold answers, evidence.
"""
from __future__ import annotations

import hashlib
import json
import struct

MAGIC = b"CCP1"
HEADER_KEYS = {"v", "assembly", "prompt_spec", "tokenizer", "tok_sha", "req", "question", "n_ctx"}
ASSEMBLY = {"server_template", "client_raw"}


class BoundaryError(ValueError):
    pass


def request_id(run_id: str, sample_id: str) -> str:
    return hashlib.sha256(f"{run_id}/{sample_id}".encode()).hexdigest()[:16]


def encode(header: dict, context_ids: list[int]) -> bytes:
    header = dict(header, n_ctx=len(context_ids))
    validate_header(header)
    hb = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    body = struct.pack(f"<{len(context_ids)}I", *context_ids)
    return MAGIC + struct.pack("<I", len(hb)) + hb + body


def decode(buf: bytes, *, vocab_size: int, max_ctx: int) -> tuple[dict, list[int]]:
    if buf[:4] != MAGIC:
        raise BoundaryError("bad magic")
    (hl,) = struct.unpack_from("<I", buf, 4)
    header = json.loads(buf[8 : 8 + hl].decode())
    validate_header(header)
    body = buf[8 + hl :]
    if len(body) != 4 * header["n_ctx"]:
        raise BoundaryError("body length mismatch")
    ids = list(struct.unpack(f"<{header['n_ctx']}I", body))
    if header["n_ctx"] > max_ctx:
        raise BoundaryError(f"n_ctx {header['n_ctx']} exceeds server limit {max_ctx}")
    if any(i >= vocab_size for i in ids):
        raise BoundaryError("token id out of vocabulary")
    return header, ids


def validate_header(h: dict) -> None:
    keys = set(h)
    if keys != HEADER_KEYS:
        raise BoundaryError(f"header keys mismatch: extra={sorted(keys - HEADER_KEYS)} "
                            f"missing={sorted(HEADER_KEYS - keys)}")
    if h["v"] != 1 or h["assembly"] not in ASSEMBLY:
        raise BoundaryError("bad version/assembly")
    if not (isinstance(h["req"], str) and len(h["req"]) == 16 and all(c in "0123456789abcdef" for c in h["req"])):
        raise BoundaryError("req must be 16 hex chars")
    if h["assembly"] == "client_raw" and (h["question"] or h["prompt_spec"]):
        raise BoundaryError("client_raw carries no question/prompt_spec")
    if not isinstance(h["question"], str) or not isinstance(h["n_ctx"], int):
        raise BoundaryError("bad field types")
