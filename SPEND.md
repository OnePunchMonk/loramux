# GPU spend log

| date | run | GPU | minutes | est. USD | actual USD | cumulative | phase |
|---|---|---|---|---|---|---|---|
| 2026-11-20 | killtest smoke | H100 | 10 | 0.6600 |  | 0.6600 | 1 |
| 2026-11-20 | profile B4 overhead | H100 | 4 | 0.2600 |  | 0.9200 | 1 |
| 2026-11-20 | profile B4 graph + fused | H100 | 5 | 0.3300 |  | 1.2500 | 1 |
| 2026-11-20 | tune expand sweep | H100 | 5 | 0.3300 |  | 1.5800 | 1 |
| 2026-11-20 | k2b qwen_image 1024 eager+compile | H100 | 15 | 0.9900 |  | 2.5700 | 1 |
| 2026-11-20 | k2b retry (meta outside compile) | H100 | 15 | 0.9900 |  | 3.5600 | 1 |
