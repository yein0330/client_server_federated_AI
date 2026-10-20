"""Server side. Runs in a separate OS process and receives payload bytes only.

ServerProcess spawns `python -m ccpilot.server_worker <backend>` and talks over stdin/stdout
with length-prefixed frames. The worker never imports the dataset module, so the only
information it holds about an item is what arrives in the payload.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_frame(f, b: bytes):
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


def _read_frame(f) -> bytes:
    n = f.read(4)
    if len(n) < 4:
        raise EOFError
    (k,) = struct.unpack("<I", n)
    return f.read(k)


class ServerProcess:
    def __init__(self, backend: str, backend_args: dict):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        self.p = subprocess.Popen([sys.executable, "-m", "ccpilot.server_worker", backend,
                                   json.dumps(backend_args)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  cwd=str(ROOT), env=env)
        self.t_load_s = json.loads(_read_frame(self.p.stdout))["t_load_s"]

    def __call__(self, payload: bytes) -> dict:
        t0 = time.perf_counter()
        _write_frame(self.p.stdin, payload)
        out = json.loads(_read_frame(self.p.stdout))
        out["t_roundtrip_s"] = time.perf_counter() - t0  # local IPC, NOT network transfer time
        return out

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            self.p.wait(timeout=30)
