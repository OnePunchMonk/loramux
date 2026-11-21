"""Helpers shared by the Triton wrappers."""
from __future__ import annotations

import os

import triton

from .torch_ref import kernel_rb  # noqa: F401  (re-exported)


def interpreting() -> bool:
    """TRITON_INTERPRET=1 (must be set before triton is imported; tests/conftest.py does it on CPU-only hosts)."""
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


# Single config used under the interpreter (autotune is disabled there).
INTERP_BM, INTERP_BN, INTERP_BK = 32, 64, 32


def configs(*, bm: bool, bn: bool, bk: bool) -> list:
    """Autotune space of TASKS L2.1: BM {32,64,128}, BN {64,128,256}, BK {32,64}, warps {4,8}, stages {3,4}.
    bm=False drops BM (varlen mode: BM is fixed by the tile table)."""
    out = []
    for m in ((32, 64, 128) if bm else (None,)):
        for n in ((64, 128, 256) if bn else (None,)):
            for k in ((32, 64) if bk else (None,)):
                for w in (4, 8):
                    for s in (3, 4):
                        kw = {}
                        if m is not None:
                            kw["BM"] = m
                        if n is not None:
                            kw["BN"] = n
                        if k is not None:
                            kw["BK"] = k
                        out.append(triton.Config(kw, num_warps=w, num_stages=s))
    return out
