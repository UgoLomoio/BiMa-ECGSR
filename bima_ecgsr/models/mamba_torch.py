"""Pure-PyTorch Mamba (v1) building blocks, state_dict-compatible with ``mamba_ssm`` 2.x.

Used when the CUDA kernels of ``mamba_ssm`` are not available (CPU inference, e.g. a CPU-only Hugging Face
Space). The modules have exactly the parameter names and shapes of ``mamba_ssm.Mamba``,
``mamba_ssm.ops.triton.layer_norm.RMSNorm`` and ``mamba_ssm.modules.block.Block`` (fused_add_norm=False), so a
checkpoint trained with ``mamba_ssm`` loads with ``strict=True``. The math follows
``mamba_ssm.modules.mamba_simple.Mamba.forward`` (non-fused path) and
``mamba_ssm.ops.selective_scan_interface.selective_scan_ref``; the time recursion h_t = a_t h_{t-1} + b_t is
evaluated with a chunked Hillis-Steele associative scan (log-depth inside each chunk, sequential carry across
chunks), which only multiplies factors in (0, 1] and is therefore numerically stable.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

SCAN_CHUNK = int(os.environ.get("BIMA_SCAN_CHUNK", "64"))
# 'numba' (fused CPU kernel, default when numba imports and the tensors are on the CPU without autograd) or 'torch'
SCAN_IMPL = os.environ.get("BIMA_SCAN_IMPL", "auto").lower()


def _numba_scan():
    if SCAN_IMPL == "torch":
        return None
    try:
        from .scan_numba import selective_scan_numba
        return selective_scan_numba
    except Exception:
        if SCAN_IMPL == "numba":
            raise
        return None


def init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    """Copy of mamba_ssm.models.mixer_seq_simple._init_weights (used only at construction time)."""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class TorchRMSNorm(nn.Module):
    """RMSNorm with the parameters of mamba_ssm's Triton RMSNorm (weight only, no bias)."""

    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rstd * self.weight.float()).to(dtype)


def _shift(x, k, fill):
    """x [..., L, N] -> x shifted by k along L (positions < k filled with `fill`)."""
    pad = torch.full_like(x[..., :k, :], fill)
    return torch.cat([pad, x[..., :-k, :]], dim=-2)


def linear_recurrence(a, b, h0=None, chunk=None):
    """h_t = a_t * h_{t-1} + b_t over the time axis (dim -2). a, b: [B, D, L, N] float32. Returns h [B, D, L, N]."""
    chunk = chunk or SCAN_CHUNK
    L = a.shape[-2]
    out = []
    carry = h0
    for c0 in range(0, L, chunk):
        aa, bb = a[..., c0:c0 + chunk, :], b[..., c0:c0 + chunk, :]
        n, k = aa.shape[-2], 1
        while k < n:  # inclusive scan of (a, b) pairs, combine (a2,b2)o(a1,b1) = (a2 a1, a2 b1 + b2)
            bb = bb + aa * _shift(bb, k, 0.0)
            aa = aa * _shift(aa, k, 1.0)
            k *= 2
        if carry is not None:
            bb = bb + aa * carry.unsqueeze(-2)
        out.append(bb)
        carry = bb[..., -1, :]
    return torch.cat(out, dim=-2)


def selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, sequential=False):
    """Same contract as mamba_ssm selective_scan_fn for real A [D,N] and input-dependent B, C [B,N,L]."""
    dtype_in = u.dtype
    fused = _numba_scan() if (not sequential and u.device.type == "cpu" and z is not None and D is not None
                              and delta_bias is not None and delta_softplus and not torch.is_grad_enabled()) else None
    if fused is not None:
        f = lambda t: t.detach().float().contiguous().numpy()  # noqa: E731
        y = fused(f(u), f(delta), f(A), f(B), f(C), f(D), f(z), f(delta_bias))
        return torch.from_numpy(y).to(dtype_in)
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    B, C = B.float(), C.float()
    dA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A.float()))
    dBu = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    if sequential:  # reference recursion (slow; for tests)
        x = dA.new_zeros(dA.shape[0], dA.shape[1], dA.shape[3]); hs = []
        for i in range(u.shape[2]):
            x = dA[:, :, i] * x + dBu[:, :, i]; hs.append(x)
        h = torch.stack(hs, dim=2)
    else:
        h = linear_recurrence(dA, dBu)
    y = torch.einsum("bdln,bnl->bdl", h, C)
    out = y if D is None else y + u * D.float()[:, None]
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype_in)


class TorchMamba(nn.Module):
    """mamba_ssm.Mamba (v1) without CUDA kernels; identical parameters, initialisation and forward math."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True,
                 layer_idx=None, device=None, dtype=None):
        fk = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, d_state, d_conv, expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **fk)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, **fk)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **fk)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **fk)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner, **fk) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)[None, :].repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A).contiguous())
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **fk)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)                       # [B, 2*d_inner, L]
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(x.transpose(1, 2))                                 # [B, L, r + 2N]
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (dt @ self.dt_proj.weight.t()).transpose(1, 2)                    # [B, d_inner, L] (bias added in scan)
        y = selective_scan(x, dt, A, Bm.transpose(1, 2), Cm.transpose(1, 2), self.D.float(), z=z,
                           delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        return self.out_proj(y.transpose(1, 2))


class TorchBlock(nn.Module):
    """mamba_ssm.modules.block.Block with fused_add_norm=False (Add -> Norm -> Mixer [-> Add -> Norm -> MLP])."""

    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):
        super().__init__()
        assert not fused_add_norm, "fused_add_norm requires the Triton kernels"
        self.residual_in_fp32 = residual_in_fp32
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None

    def forward(self, hidden_states, residual=None, inference_params=None, **kw):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        if self.mlp is not None:
            residual = hidden_states + residual
            hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
