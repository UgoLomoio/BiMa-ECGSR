"""Fused CPU selective scan (Mamba v1) with numba: y = SSM(u, delta, A, B, C) + D*u, gated by silu(z).

Exactly the recursion of mamba_ssm's selective_scan_ref (float32 state, float64-free), computed in one pass per
(batch, channel) without materialising the [B, D, L, N] decay / input tensors. Parallel over batch x channel.
Optional: imported lazily by mamba_torch; if numba is missing the pure-PyTorch associative scan is used.
"""
import math

import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath=False, cache=True)
def _scan(u, delta, dbias, A, Bt, Ct, Dv, z, out):
    # Bt, Ct: [B, L, N] (time-major so the inner state loop is contiguous)
    Bsz, Dd, L = u.shape
    N = A.shape[1]
    for i in nb.prange(Bsz * Dd):
        b = i // Dd
        d = i % Dd
        h = np.zeros(N, np.float32)
        a_row = A[d]
        for t in range(L):
            dt = delta[b, d, t] + dbias[d]
            dt = dt if dt > 20.0 else math.log1p(math.exp(dt))       # softplus (torch threshold 20)
            ut = u[b, d, t]
            acc = np.float32(0.0)
            for n in range(N):
                h[n] = math.exp(dt * a_row[n]) * h[n] + dt * Bt[b, t, n] * ut
                acc += h[n] * Ct[b, t, n]
            y = acc + ut * Dv[d]
            zt = z[b, d, t]
            out[b, d, t] = y * (zt / (1.0 + math.exp(-zt)))


def selective_scan_numba(u, delta, A, B, C, D, z, delta_bias):
    """numpy float32 arrays: u, delta, z [B, D, L]; A [D, N]; B, C [B, N, L]; D, delta_bias [D] -> y [B, D, L]."""
    out = np.empty_like(u)
    _scan(u, delta, delta_bias, A, np.ascontiguousarray(B.transpose(0, 2, 1)), np.ascontiguousarray(C.transpose(0, 2, 1)),
          D, z, out)
    return out
