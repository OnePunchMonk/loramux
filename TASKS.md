# loramux — remaining work, by complexity level

Spec: `HANDOFF.md` (section numbers below refer to it). Benchmark rules: `BENCH_CHECKLIST.md`.
Status keys: `todo` · `in progress` · `review` · `done`. **Approval:** all levels approved (2026-11-20). Everything lands in PR #3 (`l0/plumbing`).

## Levels and who does what

| Level | What belongs here | Writer | Reviewer / fixer | Exit criteria |
|---|---|---|---|---|
| **L0** | Plumbing with no numerics or GPU risk. Fully specified, verifiable by CPU unit tests alone. | Codex **luna** | Haiku subagent, **medium** effort: reviews and fixes | CPU tests pass; reviewer finds nothing in a full pass |
| **L1** | Moderate CPU logic where correctness matters but there is no perf tuning and no concurrency: format parsing, key maps, math utilities, bench harnesses. | **Sonnet, medium** (terra unreachable) | Sonnet, **low** | Same as L0 + tests cover every branch named in the task |
| **L2** | GPU kernels, CUDA streams/events, `torch.compile` integration, and system code whose bugs would flip a kill-test verdict (scheduler loop, simulator, correctness harness). | **Opus, low** | **Opus low**, iterated until a full review pass finds nothing | GPU tests pass on Modal; reviewer pass clean; no open TODOs |
| **L3** | Research-critical or cross-system work: fused GEMM epilogue, CUDA graphs, vLLM-Omni integration, running the kill test, and the verdict. | Claude **Opus high** | Owner | Owner sign-off |

Rules for every level:
- One branch per task (`l<level>/<id>-<slug>`), one PR per task. Exception: L0 ships as one branch (`l0/plumbing`) with one commit per task and a single PR. Nothing merges to `main` without the owner.
- Tests run on Modal (`modal run modal_app.py::cpu_tests`), never on a laptop. For now no Modal runs: CPU tests run in the cloud workspace, and every PR carries an unticked checklist item for the Modal run.
- GPU spend: estimate in `SPEND.md` before the run, actual after. Phase 1 cap 25 USD, total 200 USD.

---

## Existing skeleton (commit `56f670e`) — audit

The skeleton was written before this pipeline existed, so every file still needs review at the level it belongs to. Known issues are listed so the reviewers start from them.

| File | Level | Known issues |
|---|---|---|
| `context.py`, `metadata.py` | L1 | Builds tiles from host lists every step (fine for v1). No fixed-capacity buffers yet (needed for CUDA graphs, L3.2). |
| `layers.py` (`MuxLinear`, `patch_model`) | L1 | Only wraps `nn.Linear`; engines with custom linear classes need an adapter hook. |
| `formats/normalize.py` | L1 | PEFT `alpha` silently defaults to `r` when missing (PEFT stores it in `adapter_config.json`). Kohya `lora_unet_` keys use underscores and need per-model key maps. No ComfyUI-specific handling. (→ L1.1, L1.2) |
| `ops/torch_ref.py` | L1 | Fine as the oracle. Python loop, so it's slow (expected). |
| `ops/triton_shrink.py`, `ops/triton_expand_add.py` | L2 | No autotune. The rank loop always spans `r_max` (masking only, no early cut at the slot's rank). `shrink` allocates a zeroed fp32 `Z [T, r_max]` on every call. No grouped shrink. (→ L2.1) |
| `ops/custom_op.py` | L2 | OK for eager/compile. Not checked on a real DiT for graph breaks yet. |
| `pool.py` | L2 | Creates a new CUDA stream on every load. Zeroes the full slot (every layer, full `r_max`) per load. One small copy per tensor instead of one bulk copy. Pins memory per tensor. Slot rank = max over layers (wasted FLOPs for composed QKV). (→ L2.3) |
| `modal_app.py` | L0 | No bench entrypoints; versions not recorded in results. (→ L0.7) |
| `tests/*` | per file | Cover the basics; extended by each task. |

---

## L0 — approved, ready to dispatch

Each task is self-contained: file paths, behavior and tests are all given here. Use only numpy, torch, safetensors, matplotlib and the standard library unless a task says otherwise.

### L0.1 Project config — `review`
- `pyproject.toml`: add extras `bench = ["pandas", "matplotlib", "scipy"]` and `modal = ["modal"]`. Add `[tool.ruff]` with `line-length = 120`, target `py310`.
- `.gitignore`: add `results/*` with exceptions `!results/.gitkeep`. Create `results/.gitkeep`.
- Accept: `pip install -e .[dev,bench]` works; `ruff check loramux bench tests` passes on new files.

### L0.2 Timing and results utilities — `bench/common.py` — `review`
- `collect_env() -> dict`: python, torch, triton (if importable), diffusers (if importable), safetensors versions; `torch.cuda.get_device_name(0)` and driver version (`torch.cuda.driver_version()` if present, else `nvidia-smi` query, else `None`); hostname; UTC timestamp. Never raises.
- `time_cuda(fn, warmup=3, iters=20) -> list[float]` (ms per iteration, CUDA events; falls back to `time.perf_counter` with a `"timer": "cpu"` flag when CUDA is absent).
- `summarize(ms: list[float]) -> dict`: `median`, `p90`, `mean`, `n`.
- `interleave(configs: dict[str, Callable], rounds: int, warmup=3, iters=20) -> dict[str, list[float]]`: alternate config order each round (ABAB, then BABA) to cancel drift; concatenate per-config samples.
- `write_csv(path, rows: list[dict], env: dict)`: writes `# key=value` env lines as a header comment block, then the CSV. `read_csv(path) -> (env, rows)` round-trips it.
- `seed_all(seed)`: python, numpy, torch.
- Tests `tests/test_bench_common.py` (CPU): summarize on a known list; interleave order alternates (record call order with fakes); CSV round-trip including env.

### L0.3 Request latency record — `loramux/latency.py` — `review`
- `AdapterSource` enum: `NONE, GPU_HIT, CPU_HIT, DISK_HIT, MISS`.
- `RequestTimeline` dataclass with float timestamps (seconds, `time.perf_counter` domain): `t_arrive, t_admit_ready_text, t_adapter_resident, t_first_step, t_last_step, t_vae_done, t_encode_done, t_sent`, plus `adapter_source`, `n_steps`, `request_id`, `adapter_key | None`.
- `breakdown(tl) -> dict` returns the §11.3 terms in ms:
  - `queue_wait = t_first_step − max(t_admit_ready_text, t_adapter_resident)`
  - `adapter_wait = max(0, t_adapter_resident − t_admit_ready_text)`
  - `text_encode = t_admit_ready_text − t_arrive`
  - `denoise = t_last_step − t_first_step`
  - `vae_decode = t_vae_done − t_last_step`
  - `image_encode = t_encode_done − t_vae_done`
  - `transfer = t_sent − t_encode_done`
  - `e2e = t_sent − t_arrive`
  - Base-model requests (`adapter_key is None`): `adapter_wait = 0` and `t_adapter_resident` is ignored.
- `LatencyLog`: append timelines; `to_rows()` → list of dicts (timestamps, breakdown, source); `percentiles(term, by=("adapter_source",)) -> dict` with P50/P90/P99.
- Tests: hand-built timelines with known numbers; base-request path; percentiles on a small known set; the sum of terms equals `e2e`.

### L0.4 Prompt-embedding cache — `loramux/textcache.py` — `review`
- `TextEmbedCache(max_entries: int, max_bytes: int | None = None)`. Key = `(encoder_id: str, prompt: str, max_len: int)`. Values = a tensor or a tuple/dict of tensors; size = sum of `numel * element_size`.
- `get(key)` (moves to MRU), `put(key, value)` (evicts LRU until both limits hold; an item larger than `max_bytes` alone is not cached), `stats()` → hits, misses, evictions, entries, bytes.
- Optional `bypass(adapter_has_text_encoder: bool)` helper returns True when the request carries a text-encoder LoRA (those must not use the cache).
- Tests: LRU order, byte-limit eviction, oversize item, nested value sizes, stats.

### L0.5 Packed adapter blob — `loramux/blob.py` — `review`
File format `.lmx` (§11.1), little-endian:
```
bytes 0..7     magic b"LMXBLOB1"
bytes 8..15    uint64 header_len (bytes of UTF-8 JSON, before padding)
header JSON    {"version": 1, "base_model": str, "dtype": "bfloat16",
                "layers": [{"key", "d_in", "d_out", "r", "off_A", "off_B"}, ...],
                "meta": {...}, "payload_sha256": str}
padding        zero bytes up to the next multiple of 256 → payload_start
payload        per layer, in the given order: A[r, d_in] then B[d_out, r], bf16 row-major,
               each array starting at a 256-byte-aligned offset relative to payload_start
```
- `write_blob(path, adapter: dict[str, tuple[Tensor, Tensor]], base_model: str, meta: dict | None = None, order: list[str] | None = None) -> dict` (returns the header). Casts to bf16; native rank (no padding); validates `A.shape[0] == B.shape[1]`.
- `read_header(path) -> dict`.
- `read_blob(path, mmap=True, verify=False) -> dict[str, tuple[Tensor, Tensor]]` using `numpy.memmap` (or `torch.frombuffer` over an mmap) without copying when `mmap=True`; `verify=True` checks `payload_sha256`.
- `blob_nbytes(path) -> int` (payload bytes only).
- Tests: round-trip equality (bitwise, bf16) for mixed ranks and shapes; offsets aligned to 256; corrupted payload fails `verify=True`; bad magic raises; empty adapter raises.

### L0.6 Synthetic perf adapters — `bench/synth_adapters.py` — `review`
- `make_adapter(layer_shapes: dict[str, tuple[int,int]], rank: int, targets: str, seed: int) -> dict` where `targets ∈ {"attn", "all"}`; `"attn"` keeps keys whose last component matches `to_q|to_k|to_v|to_out(\.0)?|attn` (regex given as a module constant `ATTN_PAT`). `A ~ N(0, 1/d_in)`, `B ~ N(0, 1e-2)`, float32.
- `make_perf_set(layer_shapes, n=64, ranks=(8,16,32,64), seed=0) -> dict[str, dict]`: half `attn`, half `all`, ranks cycled; keys `perf_{i:03d}_{targets}_r{rank}`.
- CLI: `python -m bench.synth_adapters --shapes shapes.json --out dir/ --n 64` writes one `.lmx` per adapter via `loramux.blob.write_blob`, plus `index.json`.
- Tests: determinism by seed; rank/target split; attention filter; statistics roughly match (std of A ≈ 1/sqrt(d_in) within 10% on a large layer).

### L0.7 Modal entrypoints and spend guard — `modal_app.py`, `bench/spend.py` — `review`
- `bench/spend.py`: `estimate(gpu: str, minutes: float, usd_per_hour: float) -> float`; `append_row(path="SPEND.md", date, run, gpu, minutes, est_usd, actual_usd)` appends a markdown table row and recomputes the `cumulative` column; `check_caps(path, phase_cap=25.0, total_cap=200.0, phase="1", new_est=…)` raises `SpendCapError` if adding the estimate would exceed a cap. Phase is tracked by a `phase` column (add it to the table header if missing; keep the existing rows valid).
- `modal_app.py`: add `bench_cpu(script: str, args: str = "")` (CPU, runs `python -m <script> <args>` in `/work`) and `bench_gpu(script, args, est_minutes: float)` (H100!, calls `check_caps` before running and fails fast if exceeded; prints `collect_env()` first). Keep the existing test entrypoints unchanged.
- Tests (CPU): estimate math; append + cumulative; cap error; parsing an existing table with no rows.

### L0.8 Plots from CSVs — `bench/plots.py`, `bench/latency_breakdown.py` — `review`
Input CSVs use `bench/common.write_csv`. Pure matplotlib, Agg backend, PNG + SVG output.
- `plot_overhead_vs_tokens(csv, out)`: columns `baseline, tokens, overhead_pct` → one line per baseline (B0–B5), log-x tokens.
- `plot_sim(csv, out)`: columns `policy, rho, throughput, p50_ms, p99_ms` → two panels: throughput vs ρ and P99 vs ρ, one line per policy (P1/P2/P3).
- `plot_latency_breakdown(csv, out)` in `latency_breakdown.py`: rows from `LatencyLog.to_rows()` plus `policy, regime` → stacked bars of mean (and a second figure of P99) per term, grouped by policy × regime.
- CLI for each: `python -m bench.plots overhead in.csv out_prefix`, etc.
- Tests: run each on a tiny synthetic CSV into a tmp dir; files exist and are non-empty.

---

## Design decisions that L1 and L2 must follow

Found while auditing the skeleton. They change the data model, so they come first.

1. **Per-linear segmentation (MMDiT).** In Qwen-Image / FLUX-style MMDiT blocks, image-stream linears see `N_img` tokens per row and text-stream linears see `N_txt`. A single per-batch tile table is wrong. The batch metadata must hold **per-request** `(slot, scale)` only. Each `MuxLinear` derives its segments from its own input shape: `x: [rows, N, d]` → `rows` segments of length `N`. Rows map to requests directly (`rows == n_req`) or through CFG duplication (`rows == 2·n_req`, layout `concat` = diffusers' `torch.cat([x]*2)`, or `interleaved`). Anything else raises. 2-D packed inputs `[T, d]` need explicit `seg_len`. Derived views are cached per `(rows, N, bm, layout)` for the life of the meta object (one denoise step).
2. **Uniform-row kernel mode.** When every segment has the same length `N`, the kernels compute `row = pid // tiles_per_row`, `r0 = row·N + (pid % tiles_per_row)·BM` arithmetically. They need only `row_slot[rows]` and `row_scale[rows]` on device, not tile tables. Varlen mode (tile tables) is kept for packed inputs.
3. **Per-layer rank.** One adapter can have different ranks per layer (composed QKV has `3r`). The pool keeps `rank[layer][slot]`, and kernels mask with the **layer's** rank. `rank == 0` (layer not targeted by that adapter) exits early. Stale data beyond a layer's rank is never read, so slots never need zeroing on load.
4. **Rank bucket.** The host knows the ranks of the slots in a batch when it forms the batch. `meta.r_bucket = next_pow2(max active rank)`, clamped to `[16, r_max]`, is passed as a `constexpr` (variants 16/32/64/128) instead of always `r_max`.
5. **No host syncs in `forward`.** Everything the forward pass branches on (`any_lora`, `r_bucket`, row lists) is decided on the host when the batch is formed. Device copies of metadata are made once per step from pinned buffers with `non_blocking=True`.
6. **Testing kernels without a GPU.** `TRITON_INTERPRET=1` runs Triton kernels on CPU. In the interpreter fp32 works; **bf16 `tl.dot` returns garbage**. So interpreter tests run in fp32 (mark them `interp`, and use small shapes so they finish in seconds), and bf16 tests are marked `gpu`.

## L1 — Sonnet medium writes, Sonnet low reviews

Environment for every L1/L2 task: `. /home/claude/.venv-loramux/bin/activate`. CPU tests: `python -m pytest -q -m "not gpu" tests`. Install extra packages with pip into that venv if a task needs them (`diffusers`, `peft`, `transformers`, `accelerate`); add them to the right `pyproject.toml` extra. Tests that need diffusers are skipped when it isn't importable.

### L1.0 Metadata redesign — `context.py`, `metadata.py`, `layers.py` — `review`
- `BatchLoRAMeta` (new shape): `req_slot: list[int]`, `req_scale: list[float]`, `cfg_layout: str = "concat"`, `device`, `any_lora: bool`, `r_bucket: int`, `seg_len: list[int] | None` (packed mode only). Method `view_for(x) -> SegView`, cached as in decision 1.
- `SegView`: `rows, N (or None for varlen), uniform: bool, row_slot (int32 device), row_scale (fp32 device), seg_start, seg_len (int32 device), tiles: TileMeta (built lazily, varlen or for the reference path), slot_host: list[int], scale_host: list[float]`.
- `build_meta(req_slots, req_scales, *, cfg_layout="concat", seg_len=None, rank_of_slot=None, r_max=None, device="cpu")`. `rank_of_slot(slot) -> int` gives the max rank over layers so `r_bucket` can be computed; default `r_bucket = r_max or 64`. Keep `expand_cfg` and `build_tiles` (tests use them).
- `MuxLinear.forward`: `v = meta.view_for(x)` then call the op with the view. Until L2 adds the uniform kernel path, the op is called in varlen mode with `v.tiles`.
- Update `tests/test_metadata.py`, `tests/test_compile.py` and `tests/test_ops.py` to the new API without weakening them. Add a test with a toy MMDiT block whose two streams have different token counts (e.g. 10 image, 4 text tokens) and different adapters per request: the mixed batch must equal per-request solo runs. Add CFG tests: `rows == 2·n_req` for both layouts, and `rows` mismatches raise.

### L1.1 Format readers — `formats/peft.py`, `kohya.py`, `comfy.py`, `normalize.py` — `review`
- `load_lora(path_or_dir) -> (state_dict, config | None)`: `.safetensors` file, or a directory containing `adapter_model.safetensors` + `adapter_config.json` (PEFT).
- PEFT: alpha comes from `config["lora_alpha"]` (and `alpha_pattern` / `rank_pattern` per module when present). Missing alpha → `ValueError`, unless `assume_alpha_eq_rank=True`.
- Kohya: `lora_down` / `lora_up` / `alpha` per module; `lora_unet_` prefix with underscores. Resolve underscores against the engine's module names (see L1.2), never by guessing.
- ComfyUI: `diffusion_model.` prefix; also accept `lora_A/lora_B` and `lora_down/lora_up` under it.
- `normalize(sd, keymap, *, config=None, assume_alpha_eq_rank=False, rank_cap=None)` → `(canonical, report)`; report has `format, matched, total, unmatched (list), ranks (dict)`. Unmatched still raises unless `strict=False`, in which case the report lists them.
- Reject LoKr / LoHa / DoRA keys (`lokr_`, `hada_`, `dora_scale`, `lora_magnitude_vector`) with a clear message.
- Tests: synthetic state dicts for each format; alpha from config; alpha pattern; missing alpha raises; LoKr rejected; `strict=False` report.

### L1.2 Key maps — `formats/keymaps/__init__.py`, `generic.py`, `qwen_image.py`, `flux2.py` — `review`
- `generic.make_keymap(engine_linear_names: list[str]) -> Callable[[str], str | list[tuple[str, slice]] | None]`. It resolves a source module name, after prefix stripping, to an engine key by exact dotted match or by underscore form (`a.b_c.0.d` ↔ `a_b_c_0_d`, built from the engine names so ambiguity is detected and raises).
- A keymap may return a **split**: a list of `(engine_key, out_slice)` for a source module that is fused in the file but separate in the engine. Example: BFL/Comfy FLUX `double_blocks.0.img_attn.qkv` → diffusers `to_q`, `to_k`, `to_v`. A shares across the parts; `B` is sliced by output rows. The normalizer applies splits. The reverse case (separate in the file, fused in the engine) uses `compose_fused`.
- `qwen_image.py` and `flux2.py`: rename tables from original/Comfy names to diffusers 0.43 module names. Get the real names by instantiating `QwenImageTransformer2DModel` and the FLUX.2 transformer class from diffusers 0.43 with a tiny config (2 layers, small dims, CPU) and listing `nn.Linear` modules. Read diffusers' own LoRA conversion utilities (`diffusers/loaders/lora_conversion_utils.py`) for the source naming, and cite the functions you mirrored in a module docstring.
- `engine_linear_names(model) -> list[str]` helper.
- Tests (skip without diffusers): the tiny models' linear names; a fake Kohya and a fake Comfy adapter for each model maps 100%; the fused-qkv split reproduces `x @ (B@A).T` for q, k and v.

### L1.3 Traces — `bench/trace/__init__.py`, `synth_trace.py`, `load_gentd26.py` — `review`
- `trace_stats(adapter_ids_per_request) -> dict`: `lora_frac`, `multi_frac`, `n_unique`, `gini` (over adapter use counts), `top5pct_share`, `median_uses`.
- `fit_zipf_exponent(n_adapters=874, target_gini=0.749) -> float` by bisection on the expected Gini of Zipf(s) weights.
- `synth_trace(n_requests, rate_rps, *, n_adapters=874, lora_frac=0.168, multi_frac=0.016, gini=0.749, seed=0) -> list[Request]`. `Request(id, t_arrive, adapters: list[str], scales: list[float], steps=28, width=1024, height=1024, prompt_id)`. Poisson arrivals. Multi requests draw 2 distinct adapters.
- `load_gentd26(path) -> list[Request]`: the trace's real schema isn't known here. Accept a CSV with documented columns (`timestamp, request_id, adapters` with `;` separators, optional `steps, width, height`) and a `--column-map` JSON so a real file can be mapped without code changes. Document it in the module docstring.
- Tests: Gini of a known distribution; the fitted exponent reproduces Gini 0.749 ± 0.01; on 20k synthetic requests, lora_frac, multi_frac, top-5% share (target 53.5%, report the error) and median reuse are close to target; the CSV loader with a column map.

### L1.4 Blob ↔ normalizer ↔ pool (CPU path) — `blob.py`, `pool.py`, `formats/normalize.py` — `review`
- `convert(src, out_path, keymap, base_model, *, fused_groups=None, **normalize_kwargs) -> header`: load → normalize → optional fused composition (`fused_groups`: `{engine_key: [member source keys in output order]}`) → `write_blob`. Writes to a temp file and `os.replace` (fixes the SIGBUS-on-overwrite risk).
- `AdapterPool.register_blob(key, path)`: mmap-backed CPU tier entry (lazy; nothing is copied until `ensure_resident`). Keep `register(key, tensors)` working.
- `composite_from_blobs([(path, strength), ...]) -> canonical` using `compose_multi` per layer (layers missing in some adapters are handled).
- Tests: convert → register_blob → ensure_resident on CPU gives the same slot contents as `register`; composite equals the sum of parts on a random input.

### L1.5 Lossy compression reference — `loramux/compress.py` — `review`
- `svd_truncate(A, B, energy=0.995, max_rank=None) -> (A', B', k)` via `QR(B)`, `QR(Aᵀ)`, `SVD(R₁R₂ᵀ)` (§11.2.1). Exact when `energy=1.0` (up to fp tolerance).
- `effective_rank_report(canonical, energy) -> dict[layer, (r, k)]`.
- `quantize_fp8_rows(t) -> (q: float8_e4m3fn, scale: fp32 per row)`, `dequantize_fp8_rows(q, scale) -> bf16`. Per-row amax scaling; zero rows stay zero.
- `prune_small(canonical, base_weights, rel_thresh=1e-4) -> (canonical', pruned_keys)`.
- Tests: exactness at energy 1.0; the k chosen hits the energy target on a synthetic low-rank-plus-noise ΔW; FP8 round-trip relative error < 7% per row (e4m3); pruning.

### L1.6 Linear-chain microbench — `bench/linear_chain.py` — `todo`
- Chain of `L` linears (`d → d`), bf16 on CUDA (fp32 on CPU for smoke tests), tokens ∈ a list, rank, number of distinct adapters across rows. Baselines: `B0` no LoRA; `B1` merged ΔW (single adapter); `B2` reimplementation of the vLLM-Omni wrapper pattern (full-width delta + slice-assign copy; labelled as a reimplementation in the CSV); `B3` `torch_ref`; `B4` loramux op; `B5` vLLM's `lora_shrink` / `lora_expand` if importable, otherwise a row with `status=unavailable`.
- Uses `bench.common` timing and CSV; outputs `overhead_pct` vs B0. CLI `python -m bench.linear_chain --tokens 1024,4096,16384 --rank 64 --layers 40 --dim 3072 --adapters 8 --out results/k2a.csv`.
- Test: a CPU smoke run with tiny shapes produces the expected rows and columns.

### L1.7 Step bench and profiler — `bench/step_bench.py`, `bench/profile_steps.py` — `todo`
- Model-agnostic: a `StepTarget` protocol `(make_inputs(bs, rows_per_req) -> kwargs, forward(**kwargs))` plus a diffusers implementation for Qwen-Image / FLUX.2 transformers (random latents and text embeddings of the right shapes; real weights loaded by id).
- `step_bench`: one denoise forward, bs = 8 mixed, attention-only vs all-linear adapter sets, eager vs `torch.compile`; overhead vs base.
- `profile_steps`: `T_base(bs)`, `T_uniform(bs)`, `T_mixed(bs, k)` for bs ∈ {1,2,4,8,16}, resolutions 512² and 1024² → CSV consumed by the simulator (L2.6). Column contract documented in the module docstring.
- Test: CPU smoke run with a toy DiT `StepTarget`.

### L1.8 diffusers integration — `loramux/integrations/__init__.py`, `diffusers.py` — `review`
- `enable(pipe, pool, targets="attn"|"all"|list[regex]) -> list[str]` patches `pipe.transformer` with `patch_model`, after registering the pool's layer shapes from the model.
- `mixed_lora(pipe, req_slots, req_scales)` context manager: installs a forward pre-hook on the transformer that builds the step's `BatchLoRAMeta`. CFG is detected from the batch dimension at call time (decision 1); works both for pipelines that call the transformer twice (cond, uncond) and for those that concatenate.
- `disable(pipe)` restores the original linears.
- Tests (skip without diffusers): a tiny Qwen-Image transformer; with adapters A/B/none per request, the mixed batch equals solo runs; the twice-called-CFG path; disable restores outputs bit-exactly.

## L2 — Opus low writes, Opus low reviews (iterate until a full pass finds nothing)

### L2.1 Kernel hardening — `ops/triton_shrink.py`, `ops/triton_expand_add.py`, `ops/custom_op.py` — `todo`
- Uniform-row mode (decision 2) and varlen mode, per-layer rank with `rank == 0` early exit (decision 3), `r_bucket` constexpr (decision 4).
- `Z` workspace via `torch.empty` (never `zeros`): every row of an active tile is written, and expand never reads rows of slot −1 or columns ≥ rank. Option `z_dtype` fp32 (default) or bf16.
- `@triton.autotune` over BM ∈ {32, 64, 128}, BN ∈ {64, 128, 256}, BK ∈ {32, 64}, warps {4, 8}, stages {3, 4}, keyed on `(d_in, d_out, RB)`. Autotune is disabled under `TRITON_INTERPRET=1` (single config).
- Grouped shrink: `shrink_grouped(x, A_list, ...)` computes Z for several linears that share `x` in one pass (A concatenated along rank, via the pool's group buffers). MuxLinear uses it in eager mode only; under `torch.compiler.is_compiling()` it falls back to per-linear shrink.
- Custom op signature updated; `register_fake` kept.
- Tests: interpreter (fp32) sweep of ragged and uniform shapes, ranks {8, 16, 64, 128} in one batch, rank-0 layers, slot −1, `N % BM != 0`, `d_out % BN != 0`, `d_in % BK != 0`; GPU-marked bf16 versions with rel err ≤ 2e-3.

### L2.2 FP8-weight kernels — `ops/triton_*_fp8w.py` — `todo`
- A and B stored `float8_e4m3fn` with per-row fp32 scales (L1.5 format); upcast in registers. Same modes and masks as L2.1. Selected by the pool's storage dtype.
- Tests: GPU-marked accuracy vs the bf16 path (FP8 isn't reliable in the interpreter; try it, and keep an interp test only if it passes deterministically).

### L2.3 Pool H2D rewrite — `pool.py`, `ops/triton_scatter.py` — `todo`
- **Arena:** one tensor `arena[S, slot_elems]` per dtype. Per-layer `A_k = arena[:, offA:offA+R·d_in].view(S, R, d_in)` and `B_k` likewise `[S, d_out, R]` (views; kernels use strides). One slot's whole adapter is `arena[slot]`, contiguous.
- **Per-layer rank** `rank[layer] → int32[S]` (decision 3). No zeroing on load.
- **Staging ring:** `n_staging` pinned CPU buffers sized to the largest registered payload, reused round-robin; a buffer is reused only after its event completes.
- **Load path:** payload (native rank, blob order, from the mmap or an in-memory registration) → one memcpy into a pinned staging buffer → **one** `copy_(non_blocking=True)` into a GPU staging buffer on a **persistent** side stream → **one** `triton_scatter` launch driven by a descriptor table `(src_off, dst_off, rows, cols, src_ld, dst_ld)` for every (layer, A/B) region → record event. The rank tensors are updated on the same stream.
- `is_ready(key) -> bool` (`event.query()`), `wait(key)`. The default stream must not read a slot before its event completes: the scheduler admits only ready keys, and `acquire()` asserts readiness.
- CPU device path: the same descriptor table applied with torch copies (tests).
- Tests: CPU path equals the old per-tensor semantics; interpreter test of `triton_scatter`; LRU, refcount and readiness rules; staging reuse ordering; per-layer rank masks give correct results when a layer is missing from the adapter or has a different rank.

### L2.4 Compile and no-sync — `ops/custom_op.py`, `layers.py`, `tests/test_compile.py` — `todo`
- Zero graph breaks with `torch.compile` on (a) the toy model and (b) the tiny diffusers Qwen-Image transformer from L1.2 (CPU, `backend="aot_eager"` and `"eager"`), with mixed adapters and CFG. Inside the compiled region, MuxLinear must not touch Python lists of the meta in ways that create guards per value (pass device tensors; keep the host-side branch on `any_lora` outside or as a guard that changes once per batch).
- Test that `forward` does no device-to-host copy: run under `torch.cuda.set_sync_debug_mode("error")` (GPU-marked).

### L2.5 Serve loop — `bench/serve_loop.py` — `todo`
- `StepModel` protocol: `encode_text(prompts) -> embeds`, `init_latents(req) -> tensor`, `denoise(latents[rows], timesteps[rows], embeds[rows]) -> velocity` (meta set by the loop), `decode(latents) -> images`. Implementations: `ToyStepModel` (CPU, tests) and `DiffusersStepModel` (Qwen-Image / FLUX.2; real weights; per-row timesteps).
- Loop: arrival queue from a trace (L1.3) replayed in real time (or as fast as possible with a virtual clock for tests); `ensure_resident` at arrival; admission at step boundaries only for requests whose text is encoded and whose adapter `is_ready`; continuous batching up to `max_bs`; per-row flow-matching Euler step with each request's own sigma schedule; finished rows leave immediately; VAE decode + image encode (`torchvision.io.encode_jpeg` on CUDA if available, else PIL) on a separate stream / worker; `RequestTimeline` filled for every request.
- Policies: `mixed` (P2) and `keyed` (P1: batches only requests with the same adapter; switching adapters merges/unmerges ΔW into the base weights for real, so the swap cost is measured, not assumed).
- Output: latency CSV (L0.3 rows) + per-step CSV `(t, bs, k_distinct, step_ms)`.
- Tests (CPU, toy model, virtual clock): every request completes exactly once; outputs of the mixed loop equal solo generation per request; the keyed policy never mixes adapters in a step; timelines are monotonic.

### L2.6 Simulator — `bench/sim/__init__.py`, `replay_sim.py` — `todo`
- Discrete-event replay of a trace with step-time tables from L1.7 (`T_base`, `T_uniform`, `T_mixed`, interpolated over bs and k). Policies P1 (adapter-keyed, `max_num_seqs`, measured swap cost), P2 (mixed; H2D time = bytes / bandwidth, hidden behind text encode as modelled), P3 (mixed with zero LoRA overhead). Same admission rules as the serve loop.
- Outputs: throughput, P50/P90/P99 E2E per policy and ρ; per-request timelines in the L0.3 format.
- `validate(sim_csv, serve_csv) -> dict` (P50 error; pass if ≤ 10%).
- Tests with analytic answers: one request → `steps × T(1)` + fixed costs; two simultaneous same-adapter requests under P1 batch together; under P1 alternating adapters incur swap costs; P3 ≤ P2 ≤ P1 latency on a contrived trace; determinism by seed.

### L2.7 vLLM-Omni switch cost — `bench/vllm_omni_swap_cost.py` — `todo`
- Read the installed vllm-omni source (`pip download vllm-omni==0.32.0 --no-deps`, unpack, read only; do not install it into the venv) to find the diffusion LoRA API (`LoRARequest`, the diffusion LoRA manager). Cite the file paths in the docstring.
- Script: start an offline vLLM-Omni diffusion engine (Qwen-Image), register N adapters, time adapter activation/deactivation and per-request latency when alternating adapters vs repeating one. Output a CSV for the simulator's swap cost. GPU-only; there is no CPU test beyond import-guarding and argument parsing.

### L2.8 K1 correctness harness — `bench/k1_correctness.py` — `todo`
- K1a: per-linear rel. error vs fp32 `torch_ref` over the shape sweep (CPU-runnable, interpreter or ref).
- K1b: mixed batch vs solo through loramux on a real pipeline, final image PSNR.
- K1c: loramux vs diffusers `load_lora_weights` + `set_adapters` (merged reference), same seed, 28 steps; PSNR + LPIPS (`lpips` package, optional extra).
- Bars from HANDOFF §5 as constants in `bench/bars.py` (shared with L3.4). Writes `results/k1.csv` and a pass/fail summary.
- Tests: PSNR/LPIPS helper math on synthetic images; K1a on CPU tiny shapes.

## L3 — Claude (Opus high)

| ID | Task |
|---|---|
| L3.1 | v2 kernel: LoRA expand fused into the base GEMM epilogue (uniform-row mode), interpreter-tested, GPU-marked perf test |
| L3.2 | CUDA graphs per batch-size bucket with fixed-capacity metadata buffers |
| L3.3 | vLLM-Omni integration: `--diffusion-lora-mode mixed` patch against vllm-omni 0.32 + loramux-side adapter; drafted PR + RFC #6239 comment (not posted) |
| L3.4 | Kill-test orchestration on Modal: K1/K2/K3 runner, bar evaluation, roofline comparison, `RESULTS_killtest.md` writer (verdict logic tested on synthetic CSVs; GPU run left for the owner) |
| L3.5 | Phase 2b evaluation: bars L1–L5 / P1–P3 from CSVs, lossy-gate decisions, `RESULTS_latency.md` writer |

## Dependency order

```
L0.1 → L0.2 → (L0.3, L0.4, L0.5, L0.7, L0.8 in parallel) → L0.6 (needs L0.5)
L1.0 → everything else in L1 and L2
L0.5 → L1.4 → L2.3
L1.1 → L1.2 → (L1.8, L2.8)
L1.3 → L2.6 ← L2.5 ← (L2.3, L2.4, L1.8)
L1.5 → L2.2
L1.6, L1.7 → L2.1 → L3.1, L3.4 ← L2.6, L2.8 → L3.5
L2.4 → L3.2;  L2.7 → L3.3
```
