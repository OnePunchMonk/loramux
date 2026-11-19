# Benchmark checklist

Bars live in HANDOFF.md §5 and §11.5. They are fixed. Do not edit them after seeing data.
Tick a box only when the result CSV is under `results/` and the number is in the results file.

## 0. Before any GPU run
- [ ] Cost estimate written in `SPEND.md` (GPU, expected minutes, USD) before launch
- [ ] Phase 1 total ≤ 25 USD, overall ≤ 200 USD
- [ ] Modal GPU set to `"H100!"` (plain `"H100"` can land on H200)
- [ ] Image pinned: torch, triton, diffusers, safetensors, vllm-omni versions recorded
- [ ] Results file header has GPU name, driver, all library versions
- [ ] CPU tests pass on Modal CPU first (never local pytest)
- [ ] Smoke run on tiny shapes before the full sweep
- [ ] Spend row appended to `SPEND.md` after every run

## 1. Timing protocol (every benchmark)
- [ ] CUDA events, not wall clock
- [ ] 3 warmup + 20 timed iterations
- [ ] Report median and p90
- [ ] A/B configs interleaved in alternating order (clock drift)
- [ ] Fixed seeds
- [ ] No host sync inside the timed region (`any_lora` decided at batch formation)
- [ ] Compiled runs: compile time excluded, graph breaks checked (`TORCH_LOGS=graph_breaks`)
- [ ] Raw per-iteration data saved as CSV, every plot regenerable by one script (`bench/plots.py`)

## 2. K1 correctness (gate, run first)
- [ ] K1a per-linear rel. error vs fp32 `torch_ref` ≤ 2e-3 (ranks 8/16/64/128, slot −1, ragged segments)
- [ ] K1b mixed batch vs same request solo through loramux, final image PSNR ≥ 45 dB
- [ ] K1c loramux vs diffusers merged adapter, same seed, 28 steps: mean PSNR ≥ 35 dB and mean LPIPS ≤ 0.05
- [ ] 16 prompts × 4 real adapters, at least 2 file formats (diffusers + Kohya/Comfy)
- [ ] Scale-convention test pinned against diffusers `set_adapters`
- [ ] If K1c fails and K1b passes: check scale semantics and key mapping before numerics
- [ ] If K1b fails: try fp32 Z storage before asking to relax anything

## 3. K2a linear-chain microbench
- [ ] 40 linears, dim = model d_model, bf16, rank 64, 8 distinct adapters across rows
- [ ] Tokens 1024 / 4096 / 16384
- [ ] B0 no LoRA
- [ ] B1 merged weights (lower bound)
- [ ] B2 vLLM-Omni wrapper (real import, or labelled reimplementation)
- [ ] B3 `torch_ref` per-segment bmm
- [ ] B4 loramux v1
- [ ] B5 vLLM `lora_shrink` / `lora_expand`
- [ ] Pass: B4 overhead vs B0 at 4096 tokens ≤ ½ of B2 overhead
- [ ] Sanity: B2 reproduces issue 5053 trend (+12 / +20 / +30%) within reason
- [ ] If B5 meets the bar, use it instead of the new kernel
- [ ] Per-linear overhead vs roofline table (§3.6); `ncu` if > 1.3× predicted

## 4. K2b end-to-end step
- [ ] DiT at 1024², bs = 8 rows, 8 distinct adapters
- [ ] Attention-only and all-linear adapter sets
- [ ] Eager and compiled
- [ ] Pass (compiled): attention-only ≤ 5%, all-linear ≤ 12%
- [ ] Reported per model (Qwen-Image, Klein 4B) and per regime, not averaged
- [ ] One honest optimization pass before any NO-GO: autotune, grouped shrink, B5

## 5. K3 system value
- [ ] Profile `T_base(bs)`, `T_uniform(bs)`, `T_mixed(bs, k)` for bs 1/2/4/8/16 at 512² and 1024²
- [ ] Measure vLLM-Omni adapter switch cost (activate/deactivate)
- [ ] Trace: GenTD26 downloaded, or synthetic (874 adapters, Gini 0.749, 16.8% LoRA, 1.6% multi)
- [ ] Synthetic trace fit checked: Gini, top-5% share (53.5%), median reuse
- [ ] Arrivals Poisson at ρ ∈ {0.5, 0.7, 0.9} of measured base capacity
- [ ] Simulator policies P1 (adapter-keyed + swap cost, max_num_seqs 8), P2 (mixed), P3 (ideal)
- [ ] Throughput and P50/P99 vs ρ plotted
- [ ] Simulator validated on a 10-minute slice with `serve_loop.py`: predicted P50 within 10% of measured
- [ ] Pass: at ρ = 0.7 and 0.9, P2 vs P1 throughput ≥ 1.5× or P99 ≤ 0.5× for at least one model × resolution

## 6. Latency breakdown (§11.3, instrument from day one)
- [ ] Per-request timestamps: queue_wait, adapter_wait, text_encode, steps, vae_decode, image_encode, response_transfer
- [ ] adapter_wait split: GPU-hit / CPU-hit / disk-hit / miss
- [ ] P50/P90/P99 of each term, per policy and per regime, at ρ 0.5/0.7/0.9
- [ ] Stacked-bar plot from CSV (`bench/latency_breakdown.py`)

## 7. Phase 2b bars (only after GO or GREY)
- [ ] L1 LoRA P50 E2E ≤ 1.10× base, ρ = 0.7, mixed
- [ ] L2 P99 E2E P2 vs P1 ≤ 0.5×, ρ = 0.9
- [ ] L3 `adapter_wait` P90 for CPU-hit ≤ 1 denoise step
- [ ] L4 rank-32 attention-only adapter, pinned → GPU slot ≤ 10 ms and ≥ 70% of measured H2D bandwidth
- [ ] L5 VAE + encode on the critical path = 0
- [ ] P1 payload bytes vs source safetensors (expect ≤ 0.5× for fp32 sources)
- [ ] P2 FP8 storage passes K1c vs exact path, slots per GB ≈ 2×
- [ ] P3 SVD truncation passes K1c vs exact path, median bytes saved reported
- [ ] Each optimization (§11.4) measured before and after, one commit each

## 8. Reporting
- [ ] Verdict line GO / GREY / NO-GO with one-paragraph reason
- [ ] Every bar marked pass/fail with its number
- [ ] Plots: overhead vs tokens (K2a); throughput and P99 vs ρ for P1/P2/P3 (K3)
- [ ] Roofline predicted vs measured
- [ ] Threats to validity and surprises (baseline reimplementations named)
- [ ] Spend total
- [ ] Stop after `RESULTS_killtest.md` and wait for the owner
