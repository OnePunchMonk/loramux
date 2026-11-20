# Kill test on Modal: handoff and cost estimate

Goal: one GO / GREY / NO-GO / INCOMPLETE verdict for loramux (mixed-adapter LoRA batching for DiT serving), from the
bars in `HANDOFF.md` §5 (fixed; do not edit after seeing data). Everything here runs from PR #3 (`l0/plumbing`).
Never run pytest or benches locally (checklist rule): use Modal.

## What is and is not verified

- Verified on CPU: 293 tests, the kill-test verdict logic on synthetic CSVs, the Triton kernels in the interpreter
  (fp32 only), the fused kernel, the vLLM-Omni scheduler-key patch against the real vllm_omni source (vllm stubbed).
- **Never run on a GPU:** every bf16 kernel, the CUDA load path, the fused kernel, CUDA graphs, torch.compile of the
  real Qwen-Image transformer, all K1/K2/K3 stages. Expect a first-run bug or two; the smoke step below is there to
  find them cheaply.
- Modal wiring (`modal_app.py`) is new and was never executed (modal is not installed in my sandbox; the file parses
  and passes the dep-guard test). Treat step 0 as the test of it.

## Prerequisites

1. `pip install modal && modal setup` on your machine; check out `l0/plumbing`.
2. **Adapters:** K1b needs at least 2 real Qwen-Image LoRAs; the checklist asks for 4, in at least 2 file formats
   (diffusers/PEFT plus Kohya or Comfy). Pick Hub repos yourself. Each must be a plain LoRA (no DoRA/LoKr/LoHa).
3. `Qwen/Qwen-Image` is ungated, so no HF token is needed for it. If an adapter repo is gated, add a Modal secret and
   attach it to `prefetch` (not wired).
4. Re-check the H100 price on Modal's pricing page. The repo uses $3.95/h (`modal_app.H100_USD_PER_HOUR`), taken from
   third-party trackers on 2026-11-20, not from Modal.

## Steps

```bash
# 0. CPU tests on Modal (tests the new image + wiring; ~minutes, pennies)
modal run modal_app.py::cpu_tests

# 1. Weights + adapters into the persistent volume (CPU only, ~60 GB for Qwen-Image; ~10-20 min)
modal run modal_app.py::prefetch --repos "Qwen/Qwen-Image,<adapter repo 1>,<adapter repo 2>"
#    -> prints {repo: [/hf/.../x.safetensors]}; use those paths as --adapter below.

# 2. Log the estimate BEFORE launching (the spend guard reads SPEND.md inside the container: it ships with the code)
python -c "from bench.spend import append_row; append_row(date='2026-11-20', run='killtest smoke', gpu='H100', minutes=10, est_usd=0.66)"

# 3. GPU smoke: the two cheapest stages only (k1a, k2a need no model weights)
modal run modal_app.py::bench_gpu --script bench.killtest \
  --args "run --results /results/killtest --only k1a,k2a" --est-minutes 10
#    Stop here if either stage fails; read /results/killtest/runlog.csv.

# 4. Full run (log a 120 GPU-min / 7.90 USD estimate in SPEND.md first, same command as step 2)
modal run modal_app.py::bench_gpu --script bench.killtest \
  --args "run --results /results/killtest --adapter /hf/<a>.safetensors --adapter /hf/<b>.safetensors --adapter /hf/<c>.safetensors --adapter /hf/<d>.safetensors --only k2b,profile,k1bc,k3_512,k3_1024" \
  --est-minutes 120
#    Safer: launch k2b, profile and k1bc as three separate runs (--only k2b, then --only profile, then --only k1bc)
#    so a late failure does not lose the earlier results; K3 stages are CPU and run on the saved profile CSV.

# 5. Verdict (CPU) and fetch
modal run modal_app.py::bench_cpu --script bench.killtest --args "evaluate --results /results/killtest --out /results/RESULTS_killtest.md"
modal volume get loramux-results killtest ./results/killtest
modal volume get loramux-results RESULTS_killtest.md .

# 6. Log actual spend from the Modal dashboard (append_row with actual_usd=...), commit RESULTS_killtest.md.
```

Stages (`python -m bench.killtest plan` prints them): k1a (per-linear rel. error), k2a (40-linear chain, B0..B5),
k2b (Qwen-Image step overhead, eager + compiled), profile (T_base/T_uniform/T_mixed at bs 1..16, 512² and 1024²),
k1bc (mixed == solo PSNR, loramux vs diffusers PSNR + LPIPS), k3_512 / k3_1024 (simulator, P1 vs P2 at ρ 0.7 and 0.9).
The optional `swap` stage (real vLLM-Omni switch cost, `--vllm-omni`) is **not** runnable in this image (vllm-omni is
not installed); without it K3 uses the simulator's default switch cost, which weakens K3. Add a vllm-omni image first
if you want it.

## Reading the result

- NO-GO needs K1 or K2 to fail. K1 failing is a correctness problem: if K1c fails but K1b passes check scale
  semantics and key mapping before numerics; if K1b fails try fp32 Z storage. K2 failing: one honest optimisation pass
  (autotune, the fused kernel `MuxLinear(fused=True)`) before accepting NO-GO.
- GREY: kernel fine, system value (K3) fails: upstream the kernel only.
- INCOMPLETE: a deciding bar was not measured (any N/A row, an unmeasured regime). Not a pass.
- If Qwen-Image OOMs on the 80 GB card in step_bench, rerun that stage with `--bs 4` and say so in the results.

## Cost estimate (H100 at $3.95/h = $0.0658/min; GPU only, CPU/memory billed on top)

| scenario | GPU-min | USD | basis |
|---|---|---|---|
| Code's own stage estimates (k1a 2, k2a 4, k2b 8, profile 15, k1bc 20) | 49 | 3.23 | `bench.killtest.stages`; these assume warm models and no compile time |
| Plus optional vLLM-Omni swap stage (15) | 64 | 4.21 | not runnable in the current image |
| **Expected (log this)** | **100–120** | **6.6–7.9** | adds three model loads from the volume and torch.compile of a 60-block DiT, incl. the profile stage's ~10 compiled shapes |
| Pessimistic single launch | 180 | 11.85 | equals the 3 h function timeout, the most one launch can bill |
| One full re-run after a fix | +100–120 | +6.6–7.9 | |

Totals: smoke ($0.66) + full run ($7.90) + one re-run ($7.90) ≈ **$16**, under the phase-1 cap of **$25**
(`bench.spend.check_caps` refuses a launch that would pass it; overall cap $200). Prefetch and the CPU stages cost
pennies; Volume storage was not billed per GB when the repo's notes were written, but confirm on your plan.

How firm: the 49-minute row is the code's guess, not a measurement. The expected row is my judgement, mostly
uncertain about compile time and model-load time on Qwen-Image. Treat $6–8 as the planning number and $12 as the
ceiling per launch.
