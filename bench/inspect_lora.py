"""Print key patterns of LoRA files (header only, no tensors loaded): python -m bench.inspect_lora a.safetensors ..."""
import collections
import re
import sys

from safetensors import safe_open

for path in sys.argv[1:]:
    with safe_open(path, "pt") as f:
        keys = list(f.keys())
        pat = collections.Counter(re.sub(r"\.\d+(?=\.|$)", ".N", k) for k in keys)
        print(f"== {path.split('/')[-1]}: {len(keys)} keys, metadata {list((f.metadata() or {}))[:6]}")
        for k, c in pat.most_common(10):
            print(f"   {c:4d}  {k}")
