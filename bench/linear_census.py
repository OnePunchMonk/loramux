"""Which linears each adapter target set patches, and their share of linear FLOPs (meta device, no weights).
python -m bench.linear_census --kind flux2 --model-id black-forest-labs/FLUX.2-klein-4B"""
import argparse
import collections
import re

import diffusers
import torch

from bench.step_bench import linear_shapes

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="flux2")
ap.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-4B")
a = ap.parse_args()
cls = {"qwen_image": "QwenImageTransformer2DModel", "flux2": "Flux2Transformer2DModel"}[a.kind]
cfg = getattr(diffusers, cls).load_config(a.model_id, subfolder="transformer")
with torch.device("meta"):
    model = getattr(diffusers, cls).from_config(cfg)
allsh = linear_shapes(model, "all")
attn = linear_shapes(model, "attn")
tot = sum(i * o for i, o in allsh.values())
print(f"config: { {k: cfg[k] for k in cfg if 'layers' in k or 'dim' in k or 'heads' in k} }")
for name, sh in (("all", allsh), ("attn", attn)):
    g = collections.defaultdict(lambda: [0, 0, set()])
    for k, (i, o) in sh.items():
        pat = re.sub(r"\.\d+\.", ".N.", k)
        g[pat][0] += 1
        g[pat][1] += i * o
        g[pat][2].add((i, o))
    fl = sum(v[1] for v in g.values())
    print(f"== {name}: {len(sh)} linears, {fl / tot:.0%} of all linear weight-FLOPs (per token)")
    for pat, (n, f, shapes) in sorted(g.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"   {n:3d}x {pat:55s} {sorted(shapes)}  {f / tot:.1%}")
