"""The vLLM-Omni integration against the REAL vllm_omni scheduler, request and output classes.

Needs the vllm-omni source tree: set LORAMUX_VLLM_OMNI_SRC to a directory that contains `vllm_omni/` (an unpacked
wheel is enough; vllm itself is stubbed by tests/vllm_shim.py). Skipped otherwise. The worker module is not imported
(it needs a platform and zmq at import time); its LoRA contract is checked from its source instead.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import vllm_shim

SRC = os.environ.get("LORAMUX_VLLM_OMNI_SRC")
pytestmark = pytest.mark.skipif(not SRC or not (Path(SRC or "") / "vllm_omni").is_dir(),
                                reason="set LORAMUX_VLLM_OMNI_SRC to a dir containing vllm_omni/")


@pytest.fixture(scope="module")
def real():
    vllm_shim.install(SRC)
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.sched import base_scheduler, request_scheduler
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput, NewRequestData
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from loramux.integrations.vllm_omni import patch_scheduler_keys
    ns = SimpleNamespace(base=base_scheduler, req=request_scheduler, Request=OmniDiffusionRequest,
                         Params=OmniDiffusionSamplingParams, Output=DiffusionSchedulerOutput, New=NewRequestData,
                         patch=patch_scheduler_keys)
    yield ns
    vllm_shim.uninstall()


def _request(real, rid, lora_id, scale):
    lr = SimpleNamespace(lora_int_id=lora_id, lora_path=f"/a/{lora_id}") if lora_id is not None else None
    return real.Request(prompt="p", sampling_params=real.Params(lora_request=lr, lora_scale=scale), request_id=rid)


def test_scheduler_keys_ignore_the_adapter_after_patching(real):
    a, b, base = _request(real, "a", 1, 1.0), _request(real, "b", 2, .5), _request(real, "c", None, 1.0)
    step = lambda r: real.base.BaseScheduler._build_sampling_params_key(None, r)
    reqk = real.req.build_request_batch_sampling_params_key
    assert step(a) != step(b) and reqk(a) != reqk(b)                      # stock: adapters split the batch
    other = SimpleNamespace(build_request_batch_sampling_params_key=reqk)   # a module that imported the name early
    import sys
    sys.modules["_early_importer"] = other
    try:
        real.patch()
        real.patch()                                                      # idempotent: no double wrapping
        assert step(a) == step(b) == step(base)
        assert real.req.build_request_batch_sampling_params_key(a) == real.req.build_request_batch_sampling_params_key(b)
        assert other.build_request_batch_sampling_params_key(a) == other.build_request_batch_sampling_params_key(base)
        assert step(a).lora_int_id is None and step(a).lora_scale == 1.0
    finally:
        del sys.modules["_early_importer"]


def test_attach_with_real_scheduler_output_dataclasses(real):
    from test_vllm_omni import D, Model, _peft
    from torch import nn

    from loramux import AdapterPool
    from loramux.integrations.vllm_omni import MixedLoRA

    sd, cfg = _peft(1)
    model, base = Model(), Model()
    base.load_state_dict(model.state_dict())
    pool = AdapterPool({}, slots=2, r_max=16, device="cpu", dtype=torch.float32)
    mux = MixedLoRA(model, pool, [r"attn\.(to_qkv|to_out)$"], load_fn=lambda p: (sd, cfg))
    pipe = SimpleNamespace(denoise_step=lambda ib: model(ib.x), forward=lambda rb: model(rb.x))
    worker = SimpleNamespace(lora_manager=nn.Identity(), model_runner=SimpleNamespace(pipeline=pipe))
    mux.attach(worker)

    req = _request(real, "r0", 5, 1.0)
    out = real.Output(step_id=0, scheduled_new_reqs=[real.New(request_id="r0", req=req)],
                      scheduled_cached_reqs=SimpleNamespace(request_ids=[]), finished_req_ids=set(), num_running_reqs=1, num_waiting_reqs=0)
    worker._activate_step_lora(out)
    assert "lora:5" in pool.slot_of                      # prefetched when the request is first scheduled
    x = torch.randn(1, 3, D)
    y = pipe.denoise_step(SimpleNamespace(x=x, request_ids=list(out.scheduled_request_ids)))
    assert pool.refs["lora:5"] == 1                      # pinned from the first denoise step until the request finishes
    assert (y - base(x)).abs().max() > 0.1               # and the adapter was applied
    worker._activate_step_lora(real.Output(step_id=1, scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
                                           finished_req_ids={"r0"}, num_running_reqs=0, num_waiting_reqs=0))
    assert pool.refs["lora:5"] == 0


def test_worker_source_still_has_the_hooks_we_replace():
    """tests/vllm_shim cannot import the worker, so pin the contract MixedLoRA.attach relies on from its source."""
    tree = ast.parse((Path(SRC) / "vllm_omni/diffusion/worker/diffusion_worker.py").read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "DiffusionWorker")
    fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert [a.arg for a in fns["_activate_step_lora"].args.args] == ["self", "scheduler_output"]
    text = ast.unparse(fns["execute_stepwise"])
    assert "self._activate_step_lora(scheduler_output)" in text and "self.model_runner.execute_stepwise" in text
    runner = ast.parse((Path(SRC) / "vllm_omni/diffusion/worker/diffusion_model_runner.py").read_text())
    assert "denoise_step" in ast.unparse(runner) and "self.pipeline" in ast.unparse(runner)
