"""Select the Mamba implementation.

``mamba_ssm``: the CUDA/Triton kernels of the ``mamba_ssm`` package (used for training and for every stored result).
``torch``    : the pure-PyTorch modules of :mod:`bima_ecgsr.models.mamba_torch` (CPU inference; same state_dict).

Selection: argument > environment variable BIMA_MAMBA_BACKEND ('auto' | 'mamba_ssm' | 'torch') > 'auto'
('auto' = mamba_ssm if it imports and CUDA is available, else torch).
"""
from __future__ import annotations

import os
from types import SimpleNamespace


def resolve(name: str | None = None) -> str:
    name = (name or os.environ.get("BIMA_MAMBA_BACKEND", "auto")).lower()
    if name not in ("auto", "mamba_ssm", "torch"):
        raise ValueError(f"unknown Mamba backend {name!r}")
    if name != "auto":
        return name
    import torch
    if not torch.cuda.is_available():
        return "torch"
    try:
        import mamba_ssm  # noqa: F401
        return "mamba_ssm"
    except Exception:
        return "torch"


def get(name: str | None = None) -> SimpleNamespace:
    """Return SimpleNamespace(name, Mamba, Block, RMSNorm, init_weights)."""
    name = resolve(name)
    if name == "mamba_ssm":
        from mamba_ssm import Mamba
        from mamba_ssm.models.mixer_seq_simple import _init_weights
        from mamba_ssm.modules.block import Block
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
        return SimpleNamespace(name=name, Mamba=Mamba, Block=Block, RMSNorm=RMSNorm, init_weights=_init_weights)
    from .mamba_torch import TorchBlock, TorchMamba, TorchRMSNorm, init_weights
    return SimpleNamespace(name=name, Mamba=TorchMamba, Block=TorchBlock, RMSNorm=TorchRMSNorm, init_weights=init_weights)
