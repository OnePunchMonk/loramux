import itertools
import re

import pytest
import torch

from bench.common import read_csv
from bench.serve_loop import (
    CostModel,
    RealClock,
    ServeLoop,
    ToyStepModel,
    VirtualClock,
    flow_sigmas,
    select_admits,
)
from bench.step_bench import linear_shapes
from bench.synth_adapters import make_adapter
from bench.trace.synth_trace import Request
from loramux import AdapterPool, patch_model
from loramux.latency import AdapterSource

KEYS = ("a", "b", "c")


def _setup(policy, slots=4, patch=None):
    model = ToyStepModel(seed=0)
    shapes = linear_shapes(model.module, "attn")
    pool = AdapterPool(shapes, slots=slots, r_max=8, device="cpu", dtype=torch.float32)
    for i, k in enumerate(KEYS):
        ad = make_adapter(shapes, 8, "all", seed=i)
        pool.register(k, {n: (a, b * 5.0) for n, (a, b) in ad.items()})  # big enough to change outputs
    if patch if patch is not None else policy == "mixed":
        patch_model(model.module, ["^(?:" + "|".join(re.escape(k) for k in shapes) + ")$"], pool)
    return model, pool


def _trace(n=10, steps=4, gap=0.03):
    out = []
    for i in range(n):
        ad = [] if i % 4 == 3 else [KEYS[i % 3]]
        out.append(Request(id=i, t_arrive=gap * i, adapters=ad, scales=[1.0] * len(ad), steps=steps + i % 2,
                           prompt_id=i))
    return out


def _loop(policy, model, pool, **kw):
    kw.setdefault("cost", CostModel(h2d_s=0.07, swap_s=0.2))
    return ServeLoop(model, pool, policy=policy, max_bs=kw.pop("max_bs", 4), clock=VirtualClock(), **kw)


@pytest.mark.parametrize("policy", ["mixed", "keyed"])
def test_every_request_completes_exactly_once(policy):
    model, pool = _setup(policy)
    tr = _trace(12)
    res = _loop(policy, model, pool).run(tr)
    ids = [tl.request_id for tl in res.log.timelines]
    assert sorted(ids, key=int) == [str(r.id) for r in tr]
    assert len(set(ids)) == len(ids) == len(res.latents) == len(res.images)
    assert all(v[:2] == b"\xff\xd8" for v in res.images.values())      # JPEG
    if policy == "mixed":
        assert all(n == 0 for n in pool.refs.values())                 # every slot released


def test_mixed_outputs_equal_solo():
    model, pool = _setup("mixed")
    tr = _trace(10, gap=0.01)
    res = _loop("mixed", model, pool).run(tr)
    assert max(r["bs"] for r in res.step_rows) > 1 and max(r["k_distinct"] for r in res.step_rows) > 1
    for req in tr:
        solo = _loop("mixed", model, pool, max_bs=1).run([req])
        torch.testing.assert_close(res.latents[req.id], solo.latents[req.id], rtol=1e-5, atol=1e-5)


def test_keyed_never_mixes_and_matches_mixed():
    model, pool = _setup("keyed")
    w0 = {n: p.clone() for n, p in model.module.state_dict().items()}
    tr = _trace(12, gap=0.01)
    loop = _loop("keyed", model, pool)
    res = loop.run(tr)
    for groups in res.step_groups:
        assert len(set(groups)) == 1
    assert res.swaps >= 3 and sum(r["swap_ms"] for r in res.step_rows) == pytest.approx(200 * res.swaps)
    # merged weights give the same result as the mixed kernel path
    mm, mp = _setup("mixed")
    ref = _loop("mixed", mm, mp).run(tr)
    for req in tr:
        torch.testing.assert_close(res.latents[req.id], ref.latents[req.id], rtol=1e-3, atol=1e-2)  # |x| up to ~30
    loop.restore()
    for n, p in model.module.state_dict().items():
        torch.testing.assert_close(p, w0[n], rtol=0, atol=1e-5)


@pytest.mark.parametrize("policy", ["mixed", "keyed"])
def test_timelines_monotonic(policy):
    model, pool = _setup(policy, slots=2)       # 3 adapters, 2 slots: evictions and adapter waits
    res = _loop(policy, model, pool).run(_trace(16, gap=0.02))
    for tl in res.log.timelines:
        seq = [tl.t_arrive, tl.t_admit_ready_text, tl.t_first_step, tl.t_last_step, tl.t_vae_done,
               tl.t_encode_done, tl.t_sent]
        assert all(x <= y + 1e-12 for x, y in itertools.pairwise(seq)), seq
        assert tl.t_arrive <= tl.t_adapter_resident <= tl.t_first_step + 1e-12
        assert (tl.adapter_source is AdapterSource.NONE) == (tl.adapter_key is None)
    if policy == "mixed":
        assert pool.evictions > 0
        assert any(r["adapter_wait"] > 0 for r in res.log.to_rows())


def test_step_boundary_admission_and_post_off_path():
    model, pool = _setup("mixed")
    cost = CostModel(step_base_s=0.1, step_per_req_s=0.0, text_s=0.0, vae_s=1.0, encode_s=0.0)
    tr = [Request(id=0, t_arrive=0.0, steps=3), Request(id=1, t_arrive=0.05, steps=3)]
    res = _loop("mixed", model, pool, cost=cost).run(tr)
    tl = {t.request_id: t for t in res.log.timelines}
    assert tl["1"].t_first_step == pytest.approx(0.1)            # waits for the step boundary
    assert tl["0"].t_last_step == pytest.approx(0.3)
    assert tl["1"].t_last_step == pytest.approx(0.4)             # a 1 s VAE does not delay denoising
    assert tl["0"].t_vae_done == pytest.approx(1.3)
    assert tl["1"].t_vae_done == pytest.approx(2.3)              # one post worker: queued behind request 0
    assert [r["bs"] for r in res.step_rows] == [1, 2, 2, 1]


def test_csv_outputs(tmp_path):
    model, pool = _setup("mixed")
    res = _loop("mixed", model, pool).run(_trace(6))
    res.write(tmp_path / "lat.csv", tmp_path / "steps.csv", env={"x": 1})
    _, lat = read_csv(tmp_path / "lat.csv")
    _, st = read_csv(tmp_path / "steps.csv")
    assert len(lat) == 6 and {"e2e", "queue_wait", "t_sent", "policy"} <= set(lat[0])
    assert list(st[0])[:4] == ["t", "bs", "k_distinct", "step_ms"]


def test_real_clock_smoke():
    model, pool = _setup("mixed")
    tr = [Request(id=i, t_arrive=0.005 * i, adapters=[KEYS[i % 3]], scales=[1.0], steps=2) for i in range(4)]
    res = ServeLoop(model, pool, policy="mixed", max_bs=4, clock=RealClock()).run(tr)
    assert len(res.log) == 4 and all(r["step_ms"] > 0 for r in res.step_rows)
    assert all(r["e2e"] > 0 for r in res.log.to_rows())


def test_helpers():
    s = flow_sigmas(4, shift=3.0)
    assert s[0] == 1.0 and s[-1] == 0.0 and all(a > b for a, b in itertools.pairwise(s))

    class R:
        def __init__(self, g):
            self.group = g
    el = [R(("a", 1.0)), R(("b", 1.0)), R(("a", 1.0))]
    adm, g = select_admits("keyed", el, 0, None, 4)
    assert g == ("a", 1.0) and adm == [el[0], el[2]]
    adm, _ = select_admits("mixed", el, 3, None, 4)
    assert adm == [el[0]]


def test_diffusers_step_model_per_row_timesteps():
    pytest.importorskip("diffusers")
    from bench.serve_loop import DiffusersStepModel
    from tests.tiny_models import tiny_flux2, tiny_qwen_image

    for kind, tr in (("qwen_image", tiny_qwen_image()), ("flux2", tiny_flux2())):
        enc = lambda ps: [(torch.randn(6, 24, generator=torch.Generator().manual_seed(i)), None)
                          for i, _ in enumerate(ps)]
        m = DiffusersStepModel(kind, tr, encode_fn=enc, resolution=64, device="cpu", dtype=torch.float32)
        e = m.encode_text(["x", "y"])
        lat = torch.stack([m.init_latents(Request(id=i, t_arrive=0)) for i in range(2)])
        ts = torch.tensor([0.9, 0.3])
        with torch.no_grad():
            both = m.denoise(lat, ts, e)
            for i in range(2):
                solo = m.denoise(lat[i:i + 1], ts[i:i + 1], e[i:i + 1])
                torch.testing.assert_close(both[i:i + 1], solo, rtol=1e-4, atol=1e-4)
            assert not torch.allclose(both[0], m.denoise(lat[:1], ts[1:2], e[:1])[0])


def test_mixed_pins_in_flight_load_and_admits_once_ready():
    """GPU semantics on CPU: ensure_resident returns an unfinished event; the loop must pin it and wait."""
    model, pool = _setup("mixed")

    class Ev:
        polls = 0

        def query(self):
            Ev.polls += 1
            return Ev.polls > 3

    load = pool._load_into
    pool._load_into = lambda slot, key: (load(slot, key), Ev())[1]
    tr = _trace(3)
    res = _loop("mixed", model, pool).run(tr)
    assert len(res.log.timelines) == 3 and Ev.polls > 3
    assert all(v == 0 for v in pool.refs.values())
