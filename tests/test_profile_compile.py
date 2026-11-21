"""profile_steps compile path: one graph per (bs, patched or not), reuse across k, regional compile, streaming."""
import torch
from torch import nn

from bench.profile_steps import count_shapes, profile_target, setup_compile
from bench.step_bench import ToyDiTTarget, _ToyBlock


def _graphs():
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


def test_count_shapes_matches_grid():
    # bs (1,2,4,8,16), k (2,4,8,16), one adapter set: 5 base + 15 adapter configs, 10 compiles
    assert count_shapes((1, 2, 4, 8, 16), (2, 4, 8, 16), 1) == (20, 10)


def test_full_compile_reuses_graph_across_k():
    """bs 2 and 4, k up to 4: 7 configs but only 4 graphs (base and LoRA per bs). The old path compiled 7."""
    t = ToyDiTTarget()
    before = _graphs()
    rows = profile_target(t, batch_sizes=(2, 4), k_values=(2, 4), rank=4, modes=("compile",),
                          compile_backend="eager", compile_scope="full", rounds=1, warmup=1, iters=1)
    assert len(rows) == 2 + 2 + 3     # base x2, bs2: k1,k2, bs4: k1,k2,k4
    assert _graphs() - before == 4, f"compiled {_graphs() - before} graphs, want 4 (one per bs, base and LoRA)"
    assert {r["compile_scope"] for r in rows} == {"full"}


def test_regional_compile_used_when_model_has_repeated_blocks():
    t = ToyDiTTarget()
    called = {}

    def compile_repeated_blocks(**kw):
        called.update(kw)
        for m in t.model.modules():
            if isinstance(m, _ToyBlock):
                m.compile(**kw)
    t.model._repeated_blocks = ["_ToyBlock"]
    t.model.compile_repeated_blocks = compile_repeated_blocks
    fwd, scope = setup_compile(t, "regional", "eager")
    assert scope == "regional" and called == {"backend": "eager"} and fwd == t.forward
    torch._dynamo.reset()


def test_rows_stream_as_measured():
    seen = []
    rows = profile_target(ToyDiTTarget(), batch_sizes=(2,), k_values=(2,), rank=4, rounds=1, warmup=0, iters=1,
                          on_row=seen.append)
    assert seen == rows and len(rows) == 3 and all("first_call_s" in r for r in rows)


def test_patch_set_found_before_regional_compile():
    """With blocks compiled before the harness runs, every row must still patch the eager patch set."""
    from bench.step_bench import linear_shapes, token_linears
    t = ToyDiTTarget()
    want = len(token_linears(t, linear_shapes(t.model, "all")))
    t.model._repeated_blocks = ["_ToyBlock"]
    t.model.compile_repeated_blocks = lambda **kw: [m.compile(**kw) for m in t.model.modules()
                                                    if isinstance(m, _ToyBlock)]
    rows = profile_target(t, batch_sizes=(2,), k_values=(2,), rank=4, modes=("compile",), compile_backend="eager",
                          compile_scope="regional", rounds=1, warmup=1, iters=1)
    lora_rows = [r for r in rows if r["kind"] != "base"]
    assert lora_rows and all(r["n_patched"] == want for r in lora_rows), [r["n_patched"] for r in lora_rows]
    torch._dynamo.reset()
