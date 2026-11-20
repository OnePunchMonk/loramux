# loramux — remaining work, by complexity level

Spec: `HANDOFF.md` (section numbers below refer to it). Benchmark rules: `BENCH_CHECKLIST.md`.
Status keys: `todo` · `in progress` · `review` · `done`. **Approval:** L0 is approved. L1, L2 and L3 each need the owner's explicit go before any work starts.

## Levels and who does what

| Level | What belongs here | Writer | Reviewer / fixer | Exit criteria |
|---|---|---|---|---|
| **L0** | Plumbing with no numerics or GPU risk. Fully specified, verifiable by CPU unit tests alone. | Codex **luna** | Haiku subagent, **medium** effort: reviews and fixes | CPU tests pass; reviewer finds nothing in a full pass |
| **L1** | Moderate CPU logic where correctness matters but there is no perf tuning and no concurrency: format parsing, key maps, math utilities, bench harnesses. | **terra**, medium | Sonnet, **low** | Same as L0 + tests cover every branch named in the task |
| **L2** | GPU kernels, CUDA streams/events, `torch.compile` integration, and system code whose bugs would flip a kill-test verdict (scheduler loop, simulator, correctness harness). | **Sonnet medium** + **terra medium** (one writes, the other cross-checks or writes an independent version for kernels) | **Opus low**, iterated until a full review pass finds nothing | GPU tests pass on Modal; reviewer pass clean; no open TODOs |
| **L3** | Research-critical or cross-system work: fused GEMM epilogue, CUDA graphs, vLLM-Omni integration, running the kill test, and the verdict. | Claude **Opus high** | Owner | Owner sign-off |

Rules for every level:
- One branch per task (`l<level>/<id>-<slug>`), one PR per task. Nothing merges to `main` without the owner.
- Tests run on Modal (`modal run modal_app.py::cpu_tests`), never on a laptop.
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

### L0.1 Project config — `todo`
- `pyproject.toml`: add extras `bench = ["pandas", "matplotlib", "scipy"]` and `modal = ["modal"]`. Add `[tool.ruff]` with `line-length = 120`, target `py310`.
- `.gitignore`: add `results/*` with exceptions `!results/.gitkeep`. Create `results/.gitkeep`.
- Accept: `pip install -e .[dev,bench]` works; `ruff check loramux bench tests` passes on new files.

### L0.2 Timing and results utilities — `bench/common.py` — `todo`
- `collect_env() -> dict`: python, torch, triton (if importable), diffusers (if importable), safetensors versions; `torch.cuda.get_device_name(0)` and driver version (`torch.cuda.driver_version()` if present, else `nvidia-smi` query, else `None`); hostname; UTC timestamp. Never raises.
- `time_cuda(fn, warmup=3, iters=20) -> list[float]` (ms per iteration, CUDA events; falls back to `time.perf_counter` with a `"timer": "cpu"` flag when CUDA is absent).
- `summarize(ms: list[float]) -> dict`: `median`, `p90`, `mean`, `n`.
- `interleave(configs: dict[str, Callable], rounds: int, warmup=3, iters=20) -> dict[str, list[float]]`: alternate config order each round (ABAB, then BABA) to cancel drift; concatenate per-config samples.
- `write_csv(path, rows: list[dict], env: dict)`: writes `# key=value` env lines as a header comment block, then the CSV. `read_csv(path) -> (env, rows)` round-trips it.
- `seed_all(seed)`: python, numpy, torch.
- Tests `tests/test_bench_common.py` (CPU): summarize on a known list; interleave order alternates (record call order with fakes); CSV round-trip including env.

### L0.3 Request latency record — `loramux/latency.py` — `todo`
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

### L0.4 Prompt-embedding cache — `loramux/textcache.py` — `todo`
- `TextEmbedCache(max_entries: int, max_bytes: int | None = None)`. Key = `(encoder_id: str, prompt: str, max_len: int)`. Values = a tensor or a tuple/dict of tensors; size = sum of `numel * element_size`.
- `get(key)` (moves to MRU), `put(key, value)` (evicts LRU until both limits hold; an item larger than `max_bytes` alone is not cached), `stats()` → hits, misses, evictions, entries, bytes.
- Optional `bypass(adapter_has_text_encoder: bool)` helper returns True when the request carries a text-encoder LoRA (those must not use the cache).
- Tests: LRU order, byte-limit eviction, oversize item, nested value sizes, stats.

### L0.5 Packed adapter blob — `loramux/blob.py` — `todo`
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

### L0.6 Synthetic perf adapters — `bench/synth_adapters.py` — `todo`
- `make_adapter(layer_shapes: dict[str, tuple[int,int]], rank: int, targets: str, seed: int) -> dict` where `targets ∈ {"attn", "all"}`; `"attn"` keeps keys whose last component matches `to_q|to_k|to_v|to_out(\.0)?|attn` (regex given as a module constant `ATTN_PAT`). `A ~ N(0, 1/d_in)`, `B ~ N(0, 1e-2)`, float32.
- `make_perf_set(layer_shapes, n=64, ranks=(8,16,32,64), seed=0) -> dict[str, dict]`: half `attn`, half `all`, ranks cycled; keys `perf_{i:03d}_{targets}_r{rank}`.
- CLI: `python -m bench.synth_adapters --shapes shapes.json --out dir/ --n 64` writes one `.lmx` per adapter via `loramux.blob.write_blob`, plus `index.json`.
- Tests: determinism by seed; rank/target split; attention filter; statistics roughly match (std of A ≈ 1/sqrt(d_in) within 10% on a large layer).

### L0.7 Modal entrypoints and spend guard — `modal_app.py`, `bench/spend.py` — `todo`
- `bench/spend.py`: `estimate(gpu: str, minutes: float, usd_per_hour: float) -> float`; `append_row(path="SPEND.md", date, run, gpu, minutes, est_usd, actual_usd)` appends a markdown table row and recomputes the `cumulative` column; `check_caps(path, phase_cap=25.0, total_cap=200.0, phase="1", new_est=…)` raises `SpendCapError` if adding the estimate would exceed a cap. Phase is tracked by a `phase` column (add it to the table header if missing; keep the existing rows valid).
- `modal_app.py`: add `bench_cpu(script: str, args: str = "")` (CPU, runs `python -m <script> <args>` in `/work`) and `bench_gpu(script, args, est_minutes: float)` (H100!, calls `check_caps` before running and fails fast if exceeded; prints `collect_env()` first). Keep the existing test entrypoints unchanged.
- Tests (CPU): estimate math; append + cumulative; cap error; parsing an existing table with no rows.

### L0.8 Plots from CSVs — `bench/plots.py`, `bench/latency_breakdown.py` — `todo`
Input CSVs use `bench/common.write_csv`. Pure matplotlib, Agg backend, PNG + SVG output.
- `plot_overhead_vs_tokens(csv, out)`: columns `baseline, tokens, overhead_pct` → one line per baseline (B0–B5), log-x tokens.
- `plot_sim(csv, out)`: columns `policy, rho, throughput, p50_ms, p99_ms` → two panels: throughput vs ρ and P99 vs ρ, one line per policy (P1/P2/P3).
- `plot_latency_breakdown(csv, out)` in `latency_breakdown.py`: rows from `LatencyLog.to_rows()` plus `policy, regime` → stacked bars of mean (and a second figure of P99) per term, grouped by policy × regime.
- CLI for each: `python -m bench.plots overhead in.csv out_prefix`, etc.
- Tests: run each on a tiny synthetic CSV into a tmp dir; files exist and are non-empty.

---

## L1 — needs approval

| ID | Task | Files | Notes |
|---|---|---|---|
| L1.1 | Split format readers; PEFT alpha from `adapter_config.json` (error if absent unless `assume_alpha_eq_rank=True`); ComfyUI prefixes; coverage report | `formats/peft.py`, `kohya.py`, `comfy.py`, `normalize.py` | Fixes the silent-alpha bug |
| L1.2 | Key maps for Qwen-Image and FLUX.2 Klein (diffusers names ↔ Kohya/Comfy names, fused QKV groups) | `formats/keymaps/*.py` | Start from diffusers' own LoRA conversion utilities |
| L1.3 | Synthetic trace with Zipf exponent fitted to Gini 0.749 (bisection), 16.8% LoRA, 1.6% multi, Poisson arrivals; `trace_stats()`; GenTD26 loader | `bench/trace/synth_trace.py`, `load_gentd26.py` | Fit check: Gini, top-5% share 53.5%, median reuse 5 |
| L1.4 | Blob ↔ normalizer ↔ pool (CPU path): write fused/composite adapters; `AdapterPool.register_blob(path)` | `blob.py`, `pool.py` | Bulk GPU copy itself is L2.3 |
| L1.5 | Lossy compression reference math: SVD truncation via QR (§11.2.1), FP8 per-row quant/dequant (CPU reference) | `compress.py` | Kernels for FP8 weights are L2.2 |
| L1.6 | Linear-chain microbench harness, baselines B0–B5 (B2 = labelled reimplementation of the vLLM-Omni wrapper pattern unless importable) | `bench/linear_chain.py` | GPU runs need spend approval |
| L1.7 | Step bench + step-time profiler on diffusers pipelines (`T_base`, `T_uniform`, `T_mixed`) | `bench/step_bench.py`, `profile_steps.py` | |
| L1.8 | diffusers integration: `enable(pipe, pool, targets)`, CFG layout detection per pipeline | `integrations/diffusers.py` | |

## L2 — needs approval

| ID | Task | Files |
|---|---|---|
| L2.1 | Kernel hardening: autotune, per-slot rank loop bound, grouped shrink, reusable Z buffer, fp32 vs bf16 Z option; full shape-sweep tests | `ops/triton_*.py`, `tests/test_ops.py` |
| L2.2 | FP8-weight kernel variants (`*_fp8w`) | `ops/` |
| L2.3 | Pool H2D rewrite: persistent side stream, pinned staging ring, one bulk copy + `triton_scatter`, zero only the stale rank region, event-gated admission | `pool.py`, `ops/triton_scatter.py` |
| L2.4 | Compile on real DiTs: zero graph breaks, no host syncs in the forward | `ops/custom_op.py`, `layers.py`, `tests/test_compile.py` |
| L2.5 | `serve_loop.py`: step-wise mixed batching, admission on events, VAE + GPU JPEG on a side stream, latency logging, P1 emulation | `bench/serve_loop.py` |
| L2.6 | Discrete-event simulator P1/P2/P3 + validation against `serve_loop` (P50 within 10%) | `bench/sim/replay_sim.py` |
| L2.7 | vLLM-Omni adapter switch-cost measurement | `bench/vllm_omni_swap_cost.py` |
| L2.8 | K1 correctness harness: K1a/K1b/K1c with PSNR + LPIPS on real adapters | `bench/k1_correctness.py` |

## L3 — needs approval (Claude, Opus high)

| ID | Task |
|---|---|
| L3.1 | v2 kernel: LoRA expand fused into the base GEMM epilogue (only if K2b all-linear > 8%) |
| L3.2 | CUDA graphs per batch bucket with fixed-capacity metadata buffers |
| L3.3 | vLLM-Omni branch: `--diffusion-lora-mode mixed`, tests, benchmarks, drafted PR + RFC #6239 comment (not posted) |
| L3.4 | Run the kill test end to end; roofline vs measured; `ncu` where off; `RESULTS_killtest.md` verdict |
| L3.5 | Phase 2b: latency bars L1–L5 / P1–P3, lossy-gate decisions, `RESULTS_latency.md` |

## Dependency order

```
L0.1 → L0.2 → (L0.3, L0.4, L0.5, L0.7, L0.8 in parallel) → L0.6 (needs L0.5)
L0.5 → L1.4 → L2.3
L1.1 → L1.2 → L2.8
L1.3 → L2.6 ← L2.5 ← (L2.3, L2.4, L1.8)
L1.6, L1.7 → L2.1 → L3.4 ← L2.6, L2.8 → (L3.1, L3.2, L3.3, L3.5 as the verdict allows)
```
