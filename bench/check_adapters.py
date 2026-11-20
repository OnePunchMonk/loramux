"""CPU dry run of the format normalizer on real adapter files against the model's module names (meta device,
no weights): python -m bench.check_adapters --kind qwen_image --model-id Qwen/Qwen-Image a.safetensors b.safetensors"""
import argparse

import diffusers
import torch

from bench.k1_correctness import _keymap
from loramux.formats import load_lora, normalize

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="qwen_image")
ap.add_argument("--model-id", default="Qwen/Qwen-Image")
ap.add_argument("--assume-alpha-eq-rank", action="store_true")
ap.add_argument("adapters", nargs="+")
a = ap.parse_args()
cls = {"qwen_image": "QwenImageTransformer2DModel", "flux2": "Flux2Transformer2DModel"}[a.kind]
cfg = getattr(diffusers, cls).load_config(a.model_id, subfolder="transformer")
with torch.device("meta"):
    model = getattr(diffusers, cls).from_config(cfg)
km = _keymap(a.kind, model)
for path in a.adapters:
    name = path.split("/")[-1]
    try:
        sd, c = load_lora(path)
        out, rep = normalize(sd, km, config=c, assume_alpha_eq_rank=a.assume_alpha_eq_rank)
        ranks = sorted(set(rep["ranks"].values()))
        print(f"OK   {name}: format={rep['format']} matched {rep['matched']}/{rep['total']} ranks={ranks} "
              f"engine modules={len(out)}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}")
