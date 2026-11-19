"""Plain PyTorch reference: loop over segments. Used as the correctness oracle and baseline B3."""
import torch


def lora_apply_ref_(y, x, A, B, rank, slot_id, scale, seg_start, seg_len, fp32=False):
    """y[rows] += scale * (x[rows] @ A[slot,:r].T) @ B[slot,:,:r].T, in place.

    y: [T, d_out]  x: [T, d_in]  A: [S, R, d_in]  B: [S, d_out, R]
    """
    for i in range(slot_id.numel()):
        s = int(slot_id[i])
        if s < 0:
            continue
        r = int(rank[s])
        a, n = int(seg_start[i]), int(seg_len[i])
        dt = torch.float32 if fp32 else x.dtype
        xs = x[a:a + n].to(dt)
        z = xs @ A[s, :r].to(dt).T
        d = (z @ B[s, :, :r].to(dt).T) * float(scale[i])
        y[a:a + n] += d.to(y.dtype)
    return y
