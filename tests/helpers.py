"""Shared test fixtures."""
import torch


class FixedPool:
    """Minimal pool: fixed (A, B, rank) buffers for every layer."""
    def __init__(self, A, B, rank):
        self._b = (A, B, rank)

    def has_layer(self, k):
        return True

    def buffers(self, k):
        return self._b

    def scales(self, k):
        return None


def rand_adapter(layers, seed=0, dt=torch.float32) -> dict:
    """{key: (A[r, d_in], B[d_out, r])} from (key, d_in, d_out, r) tuples; randn drawn A then B per key."""
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(r, d_in, generator=g).to(dt), torch.randn(d_out, r, generator=g).to(dt))
            for k, d_in, d_out, r in layers}
