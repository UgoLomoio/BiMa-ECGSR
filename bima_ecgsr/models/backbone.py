"""Bidirectional-Mamba reconstruction backbone shared by BiMa-SR (regression) and DiBiMa-SR (diffusion).

    SignalEncoder (Conv1d k=5 -> BN -> SiLU -> Dropout 0.3, 64 ch)
      -> [DiBiMa-SR only] + sinusoidal timestep embedding -> MLP
      -> BidirectionalMamba (n_layers x [n_blocks forward + n_blocks reversed Mamba blocks, conv, residual])
      -> decoder Conv1d k=3 + BN, + encoder skip (residual_internal)
      -> SubPixel1D (Conv1d k=3 -> target channels; upscale 1 for spatial reconstruction)

State-dict keys are those of the trained checkpoints (best.pth). The Mamba implementation is chosen by
:mod:`bima_ecgsr.models.mamba_backend` ('mamba_ssm' CUDA kernels for training / GPU, 'torch' for CPU inference).
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import mamba_backend

# study hyper-parameters of the backbone (Mamba v1, dim 64, d_state 16, 4 blocks x 8 layers)
BEST_PARAMS = {
    "version": 1,
    "dim": 64,
    "d_state": 16,
    "n_mamba_blocks": 4,
    "n_mamba_layers": 8,
    "use_mamba": True,
    "use_diffusion": True,
}

# channel placement of a single input lead for the original lead-II residual path (decoder_forward(..., lr=...))
UNMASK_CHANNELS = {1: [1]}


def add_zero_channels(input_tensor, target_channels=12):
    """Place the input channels into a zero tensor with target_channels channels (UNMASK_CHANNELS layout)."""
    if input_tensor.ndim == 2:
        batch_size, nchs, length = None, input_tensor.size(0), input_tensor.size(1)
        out = torch.zeros((target_channels, length), device=input_tensor.device)
    else:
        batch_size, nchs, length = input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)
        out = torch.zeros((batch_size, target_channels, length), device=input_tensor.device)
    for i, ch in enumerate(UNMASK_CHANNELS[nchs]):
        if batch_size is None:
            out[ch, :] = input_tensor[i, :]
        else:
            out[:, ch, :] = input_tensor[:, i, :]
    return out


class SubPixel1D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, sr_type="temporal"):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv1d(in_channels, out_channels * upscale_factor, kernel_size=3, padding=1)
        self.sr_type = sr_type

    def forward(self, x):
        x = self.conv(x)
        B, C, T = x.size()
        r = self.upscale_factor
        if C % r != 0:
            raise ValueError(f"Channel dimension {C} not divisible by upscale factor {r}")
        if self.sr_type == "temporal":
            x = x.view(B, C // r, r, T).permute(0, 1, 3, 2).contiguous().view(B, C // r, T * r)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # x: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TemporalEmbedding(nn.Module):
    """Sinusoidal diffusion-timestep embedding followed by a 2-layer MLP -> [B, time_dim, 1]."""

    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

    def _timestep_embedding(self, t, dim):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half_dim = dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -emb_factor)
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t):
        return self.time_mlp(self._timestep_embedding(t, self.time_dim)).unsqueeze(-1)


class SignalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim, kernel_size=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, out_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_dim), nn.SiLU(), nn.Dropout(0.3))

    def forward(self, x):
        return self.encoder(x)


class BidirectionalMamba(nn.Module):
    """n_layers bidirectional layers; each runs n_mamba_blocks Mamba blocks forward in time and n_mamba_blocks on the
    time-reversed sequence, a Conv1d (k=3) per direction, sums both directions and adds the layer input."""

    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, n_layers=1, n_mamba_blocks=1, mamba_version=1,
                 merge_type="add", device="cuda:0" if torch.cuda.is_available() else "cpu", backend=None):
        super().__init__()
        be = mamba_backend.get(backend)
        self.backend = be.name
        self.d_model, self.n_layers, self.n_mamba_blocks, self.merge_type = d_model, n_layers, n_mamba_blocks, merge_type
        self.forward_layers, self.backward_layers = nn.ModuleList([]), nn.ModuleList([])
        self.forward_convs, self.backward_convs = nn.ModuleList([]), nn.ModuleList([])
        self.device = device
        # kept for state-dict compatibility with the trained checkpoints (not used in forward)
        self.residual_convs = nn.ModuleList([])
        in_channels = d_model
        for i in range(self.n_layers):
            out_channels = in_channels * 2 if self.merge_type == "concat" else in_channels
            self.residual_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1).to(self.device))
            in_channels = out_channels
        if mamba_version == 1:
            mamba_cls = be.Mamba
        else:
            if be.name != "mamba_ssm":
                raise NotImplementedError("Mamba v2/v3 blocks need the mamba_ssm backend")
            import mamba_ssm
            mamba_cls = {2: mamba_ssm.Mamba2, 3: getattr(mamba_ssm, "Mamba3", None)}[mamba_version]
        for i in range(n_layers):
            fwd_blocks, bwd_blocks = nn.ModuleList([]), nn.ModuleList([])
            fwd_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1).to(self.device)
            bwd_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1).to(self.device)
            if i > 0 and self.merge_type == "concat":
                d_model *= 2
            for j in range(n_mamba_blocks):
                for blocks in (fwd_blocks, bwd_blocks):
                    blocks.append(be.Block(
                        d_model,
                        mixer_cls=partial(mamba_cls, layer_idx=j, d_state=d_state, d_conv=d_conv, expand=expand),
                        mlp_cls=partial(nn.Linear, d_model, d_model),
                        norm_cls=partial(be.RMSNorm, eps=1e-5),
                        fused_add_norm=False).to(self.device))
            self.apply(partial(be.init_weights, n_layer=i))
            self.forward_layers.append(fwd_blocks)
            self.backward_layers.append(bwd_blocks)
            self.forward_convs.append(fwd_conv)
            self.backward_convs.append(bwd_conv)

    def _bimamba_layer(self, x, fwd_blocks, bwd_blocks, layer_idx):
        for_residual, forward_f = None, x.clone()
        for block in fwd_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        forward_f = self.forward_convs[layer_idx](forward_f.transpose(1, 2)).transpose(1, 2)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f
        if bwd_blocks is not None:
            back_residual, backward_f = None, torch.flip(x, [1])
            for block in bwd_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            backward_f = self.backward_convs[layer_idx](backward_f.transpose(1, 2)).transpose(1, 2)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f
            back_residual = torch.flip(back_residual, [1])
            if self.merge_type == "concat":
                residual = torch.cat([residual, back_residual], -1)
            else:
                residual += back_residual
        return residual

    def forward(self, x, debug=False):  # [B, L, D]
        residual_outer = x.clone()
        for i in range(self.n_layers):
            residual_inner = x.clone()
            x = self._bimamba_layer(x, self.forward_layers[i], self.backward_layers[i], layer_idx=i)
            x += residual_inner
        x += residual_outer
        return x


class BiMaSRNet(nn.Module):
    """Encoder - bidirectional Mamba - decoder network for spatial (lead-set) reconstruction.

    num_channels = k input leads; for the diffusion model (use_diffusion=True) the encoder input is
    [x_t (target_channels), inputs (k)].
    """

    def __init__(self, target_channels, ref_position=None, num_channels=12, fs_lr=50, fs_hr=500, seconds=10,
                 residual_global=True, use_mamba=True, use_diffusion=False, use_positional_encoding=False,
                 residual_internal=True, use_subpixel=True, sr_type="temporal", n_mamba_blocks=1, n_mamba_layers=1,
                 mamba_dim=64, mamba_d_state=16, mamba_version=1, backend=None, verbose=False):
        super().__init__()
        self.target_channels = target_channels
        self.target_length = fs_hr * seconds
        self.sr_type = sr_type
        self.fs_lr, self.fs_hr, self.seconds = fs_lr, fs_hr, seconds
        self.hr_len, self.lr_len = fs_hr * seconds, fs_lr * seconds
        self.use_mamba, self.n_mamba_layers, self.mamba_dim = use_mamba, n_mamba_layers, mamba_dim
        self.mamba_d_state, self.mamba_version, self.n_mamba_blocks = mamba_d_state, mamba_version, n_mamba_blocks
        self.ref_position = ref_position
        enc_out = self.mamba_dim if self.use_mamba else 256
        self.time_dim = enc_out
        self.use_diffusion = use_diffusion
        self.use_positional_encoding = use_positional_encoding
        if self.use_diffusion:
            self.time_mlp = TemporalEmbedding(self.time_dim)
            if use_positional_encoding:
                self.pos_enc = PositionalEncoding(d_model=enc_out)
        if self.sr_type == "spatial":
            self.num_channels = num_channels + self.target_channels if self.use_diffusion else num_channels
            self.input_length = self.hr_len
        else:
            self.num_channels = num_channels
            self.input_length = self.lr_len
        self.residual_global, self.residual_internal, self.use_subpixel = residual_global, residual_internal, use_subpixel
        self.encoder = SignalEncoder(self.num_channels, enc_out, kernel_size=5)
        if self.use_mamba:
            self.bottleneck = BidirectionalMamba(d_model=self.mamba_dim, d_state=self.mamba_d_state,
                                                 n_mamba_blocks=self.n_mamba_blocks, n_layers=self.n_mamba_layers,
                                                 mamba_version=self.mamba_version, backend=backend)
        else:
            self.bottleneck = nn.Sequential(nn.Conv1d(enc_out, enc_out, kernel_size=1, padding=0),
                                            nn.BatchNorm1d(enc_out), nn.SiLU(), nn.Dropout(0.3))
        upscale = 1 if self.sr_type == "spatial" else int(self.hr_len // self.lr_len)
        if verbose:
            print(f"BiMaSRNet: {num_channels} -> {self.target_channels} channels, mamba={use_mamba}, "
                  f"diffusion={use_diffusion}")
        self.decoder_sr = nn.Sequential(nn.Conv1d(enc_out, enc_out, kernel_size=3, padding=1), nn.BatchNorm1d(enc_out))
        self.subpixel = SubPixel1D(in_channels=enc_out, out_channels=self.target_channels, upscale_factor=upscale,
                                   sr_type=self.sr_type)

    def encoder_forward(self, x, pos=None, t=None, debug=False):
        x = self.encoder(x)
        residual_premamba = x.clone() if self.residual_internal else None
        if self.use_diffusion and t is not None:
            temb = self.time_mlp(t)
            x = x + temb.expand(-1, x.size(1), x.size(-1))
            if self.use_positional_encoding:
                x = self.pos_enc(x.transpose(1, 2)).transpose(1, 2)
                x = x + x.mean(2, keepdim=True)
        if self.use_mamba:
            x = self.bottleneck(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.bottleneck(x)
        return x, residual_premamba

    def decoder_forward(self, z, residual_premamba, lr=None, t=None, debug=False):
        sr = self.decoder_sr(z)
        if self.residual_internal and residual_premamba is not None:
            if residual_premamba.shape != sr.shape:
                raise ValueError(f"Residual shape {residual_premamba.shape} != decoder output shape {sr.shape}")
            sr += residual_premamba
        if self.use_subpixel:
            sr = self.subpixel(sr)
        if self.residual_global is not None and lr is not None:
            if self.sr_type == "spatial":
                lr_up = add_zero_channels(lr, self.target_channels)
            else:
                lr_up = F.interpolate(lr, size=self.target_length, mode="linear", align_corners=False)
            sr += lr_up
        return sr

    def forward_regression(self, lr, hr=None, debug=False):
        z, residual_premamba = self.encoder_forward(lr)
        return self.decoder_forward(z, residual_premamba=residual_premamba)

    def forward_diffusion(self, x_t_hr, t, lr, pos=None, debug=False):
        model_input = torch.cat([x_t_hr, lr], dim=1) if self.sr_type == "spatial" else lr
        z, residual_premamba = self.encoder_forward(model_input, t=t, pos=pos)
        return self.decoder_forward(z, residual_premamba=residual_premamba, lr=lr, t=t)

    def forward(self, *args, **kwargs):
        return self.forward_diffusion(*args, **kwargs) if self.use_diffusion else self.forward_regression(*args, **kwargs)
