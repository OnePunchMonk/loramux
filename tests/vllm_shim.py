"""Lets the real vllm_omni source (scheduler, request and key dataclasses) import without vllm installed.

`install(src)` puts a permissive stand-in `vllm` package (and zmq / aenum / msgspec / cloudpickle when they are
missing) on sys.meta_path and a bare `vllm_omni` package object that points at `src/vllm_omni` (so its heavy
`__init__` never runs). `uninstall()` removes every trace so other tests never see the stubs. The stand-ins only
have to survive import; none of vllm's behaviour is emulated.
"""
from __future__ import annotations

import enum
import importlib.abc
import importlib.machinery
import os
import sys
import types

_STUBBED = ("vllm", "zmq", "aenum", "msgspec", "cloudpickle")
_FINDER = None


class _Mod(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name == "PlatformEnum":
            v = enum.Enum(name, {n: n for n in ("CUDA", "ROCM", "TPU", "XPU", "CPU", "NEURON", "OOT", "UNSPECIFIED",
                                                "NPU", "MUSA")})
        elif name.isupper():
            v = []
        else:
            v = type(name, (), {"__init__": lambda self, *a, **k: None, "__call__": lambda self, *a, **k: None,
                                "__init_subclass__": classmethod(lambda cls, **kw: None),
                                "__class_getitem__": classmethod(lambda cls, item: cls)})
        setattr(self, name, v)
        return v


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top in _STUBBED and (top == "vllm" or importlib.machinery.PathFinder.find_spec(top) is None):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _Mod(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


def install(src: str, version: str = "0.32.0") -> None:
    global _FINDER
    _FINDER = _Finder()
    sys.meta_path.insert(0, _FINDER)
    import vllm
    vllm.__version__, vllm.__version_tuple__ = version, tuple(int(p) for p in version.split("."))
    pkg = types.ModuleType("vllm_omni")
    pkg.__path__ = [os.path.join(src, "vllm_omni")]
    pkg.__version__ = version
    sys.modules["vllm_omni"] = pkg


def uninstall() -> None:
    global _FINDER
    if _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
    for name in [n for n in sys.modules if n.split(".")[0] in ("vllm", "vllm_omni") or
                 (n.split(".")[0] in _STUBBED and isinstance(sys.modules[n], _Mod))]:
        del sys.modules[name]
    _FINDER = None
