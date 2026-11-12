"""Explainability for the Space (app only; not part of the paper's analyses).

1. Integrated Gradients (IG) on the 12-lead classifier: x = the classifier's own per-window, per-lead z-scored input
   (its training-time input handling, applied once), f(x) = logit_c(network(x)) per 5-s window, baseline x0 = 0
   (= the per-lead mean in z units). The path runs in z space because the z-score makes the classifier invariant to
   the scale of its input (IG along alpha * raw input would be identically zero). 32-point Gauss-Legendre quadrature of the path integral on [0, 1] (exact for
   polynomial paths up to degree 63; the completeness residual sum(attr) - (f(x) - f(x0)) is reported).
2. Reconstruction attribution: gradient x input of a predicted lead's output (summed over a time window of each 5-s
   window) with respect to the normalised measured inputs that the network sees. One backward pass per predicted lead
   (batched), fp32.

Mamba backends and autograd: mamba_ssm's kernels have their own backward. The pure-PyTorch backend would keep every
[B, d_inner, L, d_state] scan tensor of all 64 Mamba blocks for the backward pass (~6 GB per window), so during an
attribution call (context manager `torch_backend_autograd`) its linear recurrence gets a memory-light custom backward
(reverse scan) and every Mamba block is gradient-checkpointed. Forward values are unchanged.
"""
from __future__ import annotations

import contextlib
import time

import numpy as np

IG_STEPS = 32


# ----------------------------------------------------------------------------- PyTorch-backend autograd helpers
def _linrec_fn():
    import torch
    from bima_ecgsr.models import mamba_torch as MT
    orig = MT.linear_recurrence

    class LinRec(torch.autograd.Function):
        """h_t = a_t h_{t-1} + b_t (h_{-1} = 0). Backward: g_t = dL/dh_t + a_{t+1} g_{t+1} (reverse scan);
        dL/db_t = g_t, dL/da_t = g_t h_{t-1}. Saves only a and h."""

        @staticmethod
        def forward(ctx, a, b):
            with torch.no_grad():
                h = orig(a, b)
            ctx.save_for_backward(a, h)
            return h

        @staticmethod
        def backward(ctx, gh):
            a, h = ctx.saved_tensors
            with torch.no_grad():
                a_next = torch.cat([a[..., 1:, :], torch.zeros_like(a[..., :1, :])], dim=-2)
                g = torch.flip(orig(torch.flip(a_next, [-2]), torch.flip(gh.contiguous(), [-2])), [-2])
                h_prev = torch.cat([torch.zeros_like(h[..., :1, :]), h[..., :-1, :]], dim=-2)
            return g * h_prev, g

    def linear_recurrence(a, b, h0=None, chunk=None):
        if h0 is not None or chunk is not None or not torch.is_grad_enabled():
            return orig(a, b, h0, chunk)
        return LinRec.apply(a, b)

    return MT, orig, linear_recurrence


@contextlib.contextmanager
def torch_backend_autograd(model):
    """Patch the PyTorch Mamba backend for memory-light input gradients (no effect on mamba_ssm models)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from bima_ecgsr.models.mamba_torch import TorchBlock
    MT, orig, patched = _linrec_fn()
    blocks = [m for m in model.modules() if isinstance(m, TorchBlock)]
    saved = []
    for b in blocks:
        f = b.forward
        saved.append((b, f))

        def wrapped(hidden_states, residual=None, inference_params=None, _f=f, **kw):
            if residual is None:
                return checkpoint(lambda h: _f(h, None, inference_params=None), hidden_states, use_reentrant=False)
            return checkpoint(lambda h, r: _f(h, r, inference_params=None), hidden_states, residual, use_reentrant=False)
        b.forward = wrapped
    MT.linear_recurrence = patched
    try:
        yield len(blocks)
    finally:
        MT.linear_recurrence = orig
        for b, f in saved:
            b.forward = f


# ----------------------------------------------------------------------------- classifier: Integrated Gradients
def _gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2                    # nodes / weights on [0, 1]


def integrated_gradients(clf, X, cls, device, steps=IG_STEPS, chunk=64):
    """X [W,12,L] (any units; z-scored per window and lead first, as the classifier does). Returns dict(attr [W,12,L],
    f_x [W], f_0 [W], sum_attr [W], rel_err [W] = |sum_attr - (f_x - f_0)| / |f_x - f_0|)."""
    import torch
    from bima_ecgsr.classification.predict import zscore_windows
    clf.eval()
    alphas, weights = _gauss_legendre(steps)
    x = zscore_windows(torch.as_tensor(np.asarray(X, np.float32), device=device))
    W = x.shape[0]
    path = (torch.as_tensor(alphas, device=device, dtype=torch.float32)[:, None, None, None] * x[None])  # [S,W,12,L]
    path = path.reshape(-1, *x.shape[1:])
    grads = torch.empty_like(path)
    for s in range(0, path.shape[0], chunk):
        p = path[s:s + chunk].clone().requires_grad_(True)
        out = clf(p)[:, cls].sum()
        grads[s:s + chunk] = torch.autograd.grad(out, p)[0]
    grads = grads.reshape(steps, W, *x.shape[1:])
    avg = (torch.as_tensor(weights, device=device, dtype=torch.float32)[:, None, None, None] * grads).sum(0)
    attr = (x * avg).detach()
    with torch.no_grad():
        f_x = clf(x)[:, cls]
        f_0 = clf(torch.zeros_like(x))[:, cls]
    sa = attr.sum((1, 2))
    rel = (sa - (f_x - f_0)).abs() / (f_x - f_0).abs().clamp_min(1e-6)
    c = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
    return dict(attr=c(attr), f_x=c(f_x), f_0=c(f_0), sum_attr=c(sa), rel_err=c(rel))


# ----------------------------------------------------------------------------- reconstruction: gradient x input
def recon_grad_x_input(model, inp, pred_leads, t0, t1, sampler=None, chunk=None):
    """model: LeadResidualSR (regression) or a callable f(inp) -> [B,12,L]; inp [W,k,L] normalised inputs on the model's
    device. For each predicted lead c: d sum_{t in [t0,t1)} out[:, c, t] / d inp, times inp.
    Returns gxi [n_pred, W, k, L] (numpy float32) and finite flag."""
    import torch
    f = sampler or model
    W = inp.shape[0]; n = len(pred_leads)
    chunk = chunk or n
    out = np.empty((n, W, inp.shape[1], inp.shape[2]), np.float32)
    for s in range(0, n, chunk):
        leads = pred_leads[s:s + chunk]
        x = inp.detach().repeat(len(leads), 1, 1).clone().requires_grad_(True)       # [len*W, k, L]
        y = f(x)                                                                       # [len*W, 12, L]
        sel = torch.as_tensor(np.repeat(leads, W), device=inp.device)
        obj = y[torch.arange(y.shape[0], device=inp.device), sel, t0:t1].sum()
        g = torch.autograd.grad(obj, x)[0]
        out[s:s + len(leads)] = (g * x).detach().float().cpu().numpy().reshape(len(leads), W, *inp.shape[1:])
    return out, bool(np.isfinite(out).all())


def integrated_gradients_adaptive(clf, X, cls, device, tol=0.05, steps=(IG_STEPS, 128, 512)):
    """IG with IG_STEPS Gauss-Legendre points; refined (128, 512) while the completeness error exceeds tol."""
    for n in steps:
        r = integrated_gradients(clf, X, cls, device, steps=n)
        r["steps"] = n
        if float(r["rel_err"].max()) <= tol:
            break
    return r


def lead_matrix(gxi):
    """Share of |grad x input| per measured lead, rows = predicted leads: [n_pred, k], rows sum to 1."""
    a = np.abs(gxi).sum((1, 3))
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def timed(fn, *a, sync=None, **k):
    t0 = time.perf_counter(); r = fn(*a, **k)
    if sync:
        sync()
    return r, time.perf_counter() - t0
