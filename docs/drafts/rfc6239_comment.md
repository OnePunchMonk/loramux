# DRAFT, not posted: comment for RFC #6239

Before proposing more per-adapter machinery: the scheduler currently splits batches on `lora_int_id` / `lora_scale`, so
the useful question is how much throughput a mixed batch buys on a skewed adapter mix. We built a prototype that keeps
the keys adapter-free and applies per-row LoRA through a segmented kernel, behind a `mixed` mode that leaves the default
untouched. Numbers (correctness vs solo and vs diffusers, step overhead, simulated and live P99) will follow once the GPU
runs are done; we would rather post them than argue from the design. If the maintainers prefer a different seam (for
example a scheduler hook instead of patched keys), say so and we will adapt the prototype.
