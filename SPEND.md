# GPU spend log

| date | run | GPU | minutes | est. USD | actual USD | cumulative | phase |
|---|---|---|---|---|---|---|---|
| 2026-11-20 | killtest smoke | H100 | 10 | 0.6600 |  | 0.6600 | 1 |
| 2026-11-20 | profile B4 overhead | H100 | 4 | 0.2600 |  | 0.9200 | 1 |
| 2026-11-20 | profile B4 graph + fused | H100 | 5 | 0.3300 |  | 1.2500 | 1 |
| 2026-11-20 | tune expand sweep | H100 | 5 | 0.3300 |  | 1.5800 | 1 |
| 2026-11-20 | k2b qwen_image 1024 eager+compile | H100 | 15 | 0.9900 |  | 2.5700 | 1 |
| 2026-11-20 | k2b retry (meta outside compile) | H100 | 15 | 0.9900 |  | 3.5600 | 1 |
| 2026-11-20 | k2b qwen_image 1024 rank64 | H100 | 15 | 0.9900 |  | 4.5500 | 1 |
| 2026-11-20 | k2b flux2 klein4b 1024 rank16 | H100 | 10 | 0.6600 |  | 5.2100 | 1 |
| 2026-11-20 | k1bc qwen_image 4 adapters | H100 | 25 | 1.6500 |  | 6.8600 | 1 |
| 2026-11-20 | k1bc retry (alpha=rank) | H100 | 25 | 1.6500 |  | 8.5100 | 1 |
| 2026-11-20 | k1bc retry 2 (dotted lora.down support) | H100 | 25 | 1.6500 |  | 10.1600 | 1 |
| 2026-11-20 | profile_b4 after FastLaunch | H100 | 5 | 0.3300 |  | 10.4900 | 1 |
| 2026-11-20 | k2b klein 1024 eager,compile,cudagraph | H100 | 15 | 0.9900 |  | 11.4800 | 1 |
| 2026-11-20 | K1b controls (ctrl_base, ctrl_same) | H100 | 20 | 1.3200 |  | 12.8000 | 1 |
| 2026-11-20 | k2b klein 1024 attn slices fix | H100 | 12 | 0.7900 |  | 13.5900 | 1 |
| 2026-11-20 | K1b fp32 klein + controls | H100 | 20 | 1.3200 |  | 14.9100 | 1 |
