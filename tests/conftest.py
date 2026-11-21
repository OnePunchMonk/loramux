"""Test setup.

Triton's interpreter (TRITON_INTERPRET=1) is chosen when triton is imported, so it must be set before anything
imports triton. On a host without CUDA this file sets it (importing torch does not import triton), so the
`interp` tests run the real kernels on CPU in the default `pytest -m "not gpu"` command. On a CUDA host the
compiled kernels are wanted, so the interpreter is left off and `interp` tests are skipped unless the run sets
TRITON_INTERPRET=1 itself (e.g. `TRITON_INTERPRET=1 pytest -m interp tests`).
"""
import os
import sys

import pytest
import torch

if not torch.cuda.is_available() and "triton" not in sys.modules:
    os.environ.setdefault("TRITON_INTERPRET", "1")


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no CUDA")
    skip_interp = pytest.mark.skip(reason="needs TRITON_INTERPRET=1 (set automatically on CPU-only hosts)")
    for it in items:
        if "gpu" in it.keywords and not torch.cuda.is_available():
            it.add_marker(skip_gpu)
        if "interp" in it.keywords and not _interpreting():
            it.add_marker(skip_interp)
