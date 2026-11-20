"""Plain PyTorch reference: loop over segments. Used as the correctness oracle, baseline B3 and the CPU fallback.

Same per-layer-rank semantics as the Triton kernels (TASKS decision 3): `rank` is the layer's int32 [S];
slot -1 and rank 0 are skipped, and only A[slot, :r] / B[slot, :, :r] are read (stale data beyond the rank
never is).
"""
import torch


def kernel_rb(r_bucket: int) -> int:
    """Rank tile width used by the kernels: next pow2 of the rank bucket, at least 16 (tl.dot needs K >= 16)."""
    r = max(16, int(r_bucket))
    return 1 << (r - 1).bit_length()


def lora_apply_ref_(y, x, A, B, rank, slot_id, scale, seg_start=None, seg_len=None, fp32=False, *, N: int = 0,
                    a_scale=None, b_scale=None, r_bucket: int | None = None):
    """y[rows] += scale * (x[rows] @ A[slot,:r].T) @ B[slot,:,:r].T, in place.

    y: [T, d_out]  x: [T, d_in]  A: [S, R, d_in]  B: [S, d_out, R]  rank: int32 [S] for this layer.
    Segments: seg_start / seg_len (varlen) or, when they are None, uniform rows of N tokens (segment i =
    [i*N, (i+1)*N)). a_scale [S, R] / b_scale [S, d_out]: per-row fp32 scales of FP8-stored A / B.
    r_bucket: when given, a live rank above it raises (the kernels would silently truncate)."""
    if seg_start is None and N <= 0:
        raise ValueError("pass seg_start/seg_len (varlen) or N > 0 (uniform rows)")
    for i in range(slot_id.numel()):
        s = int(slot_id[i])
        if s < 0:
            continue
        r = int(rank[s])
        if r == 0:
            continue
        if r_bucket is not None and r > kernel_rb(r_bucket):
            raise ValueError(f"slot {s} has rank {r} at this layer but the batch rank bucket is {r_bucket}")
        if seg_start is None:
            a, n = i * N, N
        else:
            a, n = int(seg_start[i]), int(seg_len[i])
        dt = torch.float32 if (fp32 or a_scale is not None) else x.dtype
        xs = x[a:a + n].to(dt)
        As = A[s, :r].to(dt)
        Bs = B[s, :, :r].to(dt)
        if a_scale is not None:
            As = As * a_scale[s, :r, None].to(dt)
        if b_scale is not None:
            Bs = Bs * b_scale[s, :, None].to(dt)
        z = xs @ As.T
        d = (z @ Bs.T) * float(scale[i])
        y[a:a + n] += d.to(y.dtype)
    return y
