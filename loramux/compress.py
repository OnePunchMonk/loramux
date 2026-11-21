"""Lossy adapter compression references (HANDOFF §11.2). All opt-in; none of this runs on the serving path.

Canonical adapters are `{layer_key: (A [r, d_in], B [d_out, r])}` with delta W = B @ A.
"""
from __future__ import annotations

import torch

FP8_MAX = 448.0  # largest finite float8_e4m3fn value


def svd_truncate(A: torch.Tensor, B: torch.Tensor, energy: float = 0.995,
                 max_rank: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Rank-truncate delta W = B @ A without forming the d_out x d_in matrix (§11.2.1).

    QR(B) = Q1 R1, QR(A^T) = Q2 R2, SVD(R1 R2^T) = U S V^T, so delta W = Q1 U S V^T Q2^T.
    k is the smallest rank whose singular values hold >= `energy` of the squared Frobenius norm (energy >= 1.0
    keeps every non-zero direction, so the result is exact up to fp error), capped by `max_rank`.
    Returns (A' [k, d_in], B' [d_out, k], k) with B' = Q1 U_k S_k^(1/2), A' = (Q2 V_k S_k^(1/2))^T, in the
    input dtype. An all-zero delta returns k = 1 with zero factors.
    """
    if A.shape[0] != B.shape[1]:
        raise ValueError(f"rank mismatch: A {tuple(A.shape)} vs B {tuple(B.shape)}")
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    dt = A.dtype
    A64, B64 = A.to(torch.float64), B.to(torch.float64)
    Q1, R1 = torch.linalg.qr(B64)
    Q2, R2 = torch.linalg.qr(A64.T)
    U, S, Vh = torch.linalg.svd(R1 @ R2.T)
    e = S * S
    total = float(e.sum())
    if total == 0.0:
        k = 1
    elif energy >= 1.0:
        k = max(1, int((S > S[0] * torch.finfo(torch.float64).eps * max(S.numel(), 1)).sum()))
    else:
        cum = torch.cumsum(e, 0) / total
        k = int((cum < energy).sum()) + 1
    if max_rank is not None:
        k = min(k, max_rank)
    k = max(1, min(k, S.numel()))
    root = S[:k].sqrt()
    B_new = Q1 @ (U[:, :k] * root)
    A_new = ((Q2 @ Vh[:k].T) * root).T
    if A_new.shape[0] < k:  # not reachable (k <= S.numel()), kept as a guard for degenerate shapes
        raise AssertionError("internal rank error")
    return A_new.to(dt).contiguous(), B_new.to(dt).contiguous(), k


def effective_rank_report(canonical: dict[str, tuple[torch.Tensor, torch.Tensor]],
                          energy: float = 0.995) -> dict[str, tuple[int, int]]:
    """{layer: (original rank r, rank k kept at this energy)}."""
    out = {}
    for key, (A, B) in canonical.items():
        _, _, k = svd_truncate(A, B, energy)
        out[key] = (int(A.shape[0]), k)
    return out


def quantize_fp8_rows(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row amax scaling to float8_e4m3fn. Returns (q, scale fp32 [rows]) with t ~ q * scale[:, None].

    All-zero rows get scale 1.0 and stay exactly zero.
    """
    if t.dim() != 2:
        raise ValueError(f"expected a 2-D tensor, got shape {tuple(t.shape)}")
    tf = t.to(torch.float32)
    amax = tf.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (tf / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def dequantize_fp8_rows(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.to(torch.float32) * scale[:, None]).to(torch.bfloat16)


def prune_small(canonical: dict[str, tuple[torch.Tensor, torch.Tensor]], base_weights: dict[str, torch.Tensor],
                rel_thresh: float = 1e-4) -> tuple[dict, list[str]]:
    """Drop layers whose ||B A||_F / ||W||_F < rel_thresh (§11.2.3). Returns (kept, pruned_keys).

    ||B A||_F^2 = tr((B^T B)(A A^T)) is computed without forming B A. A layer with no entry in
    `base_weights` cannot be judged and is kept.
    """
    kept, pruned = {}, []
    for key, (A, B) in canonical.items():
        W = base_weights.get(key)
        if W is None:
            kept[key] = (A, B)
            continue
        Af, Bf = A.to(torch.float64), B.to(torch.float64)
        d2 = ((Bf.T @ Bf) * (Af @ Af.T)).sum()  # tr(X Y) for symmetric X, Y
        w = W.to(torch.float64).norm()
        if float(d2.sqrt()) < rel_thresh * float(w):
            pruned.append(key)
        else:
            kept[key] = (A, B)
    return kept, pruned
