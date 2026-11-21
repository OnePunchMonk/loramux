# DRAFT, not posted: [Diffusion] Mixed-adapter LoRA batching (`--diffusion-lora-mode mixed`)

## Problem
`lora_int_id` and `lora_scale` are part of `StepBatchSamplingParamsKey` / `RequestBatchSamplingParamsKey`, and the
manager holds one active adapter (`max_loras=1`). Requests with different adapters therefore never share a denoise
batch; under a skewed adapter distribution the batch degenerates to size ~1 and adapter swaps dominate.

## Change
- Opt-in mode `mixed`: the two batch keys drop the adapter identity (`dataclasses.replace(key, lora_int_id=None, lora_scale=1.0)`).
- The worker does not call `set_request_adapter`. Each request's `LoRARequest` is resolved to an entry in a GPU slot pool
  (loaded once, LRU, pinned while a request is in flight); per-row slot/scale tensors are built from request order.
- Linears get a wrapper that adds `scale * (x @ A[slot].T) @ B[slot].T` per row by segment (Triton, per-layer rank).
  Fused `to_qkv` / `add_kv_proj` use a composed adapter (concat A, block-diagonal B) from `packed_modules_mapping`.
- Default mode is unchanged.

## Out of scope
Tensor parallel > 1, quantized base weights, DoRA/LoKr, LoRA on the text encoder.

## Evidence (to be filled from the Modal runs)
- [ ] K1: mixed == solo per request (PSNR >= 45 dB), equals diffusers `set_adapters` (PSNR >= 35, LPIPS <= 0.05)
- [ ] K2: step overhead vs plain, attention-only <= 5%, all linears <= 12%
- [ ] K3: simulator and live serve loop, P2 vs P1 at rho 0.7 / 0.9
- [ ] Unit tests: stubs for linears, pipeline and worker (no GPU)
