"""1D-ResNet ECG classifier (xresnet1d-style): the 12-lead judge used to measure diagnostic readability.

Input  : [B, C, T] per-lead z-scored ECG windows (C = number of input leads, T = 2500 at 500 Hz).
Output : [B, 17] logits (apply sigmoid) in the order of ``LABELS``:
         5 PTB-XL diagnostic superclasses + 12 shared rhythm/morphology panel labels.

Architecture (follows the xresnet1d family of the PTB-XL benchmark, Strodthoff et al. 2021,
adapted to 500 Hz input with a wider kernel):
  stem   : 3 x (Conv1d k=7, BN, ReLU) with widths 32, 32, 64; first conv stride 2; MaxPool(3, s=2)
  stages : 4 stages of 2 pre-defined BasicBlocks (2 x Conv1d k=7 + BN, identity / 1x1-conv shortcut
           with AvgPool downsampling as in xresnet "ResNet-D"), widths 64, 128, 192, 256,
           strides 1, 2, 2, 2  (temporal resolution 2500 -> 625 -> 79)
  head   : concat(adaptive avg pool, adaptive max pool) -> BN -> Dropout(0.25) -> Linear(512,128)
           -> ReLU -> BN -> Dropout(0.5) -> Linear(128, 17)
~3.9 M parameters for 12 input leads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
PANEL = ("AFIB", "AFLT", "1AVB", "PAC", "PVC", "SBRAD", "STACH", "WPW", "LVH", "RVH", "CRBBB", "CLBBB")
LABELS = SUPERCLASSES + PANEL
N_LABELS = len(LABELS)  # 17
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def parse_leads(spec: str) -> list[int]:
    """'all' or comma-separated lead names (e.g. 'I,II,V2') -> sorted channel indices."""
    if spec.strip().lower() in ("all", "12"):
        return list(range(12))
    idx = []
    for name in spec.split(","):
        name = name.strip()
        if name not in LEADS:
            raise ValueError(f"unknown lead {name!r}; valid: {LEADS}")
        idx.append(LEADS.index(name))
    if len(set(idx)) != len(idx):
        raise ValueError("duplicate leads")
    return sorted(idx)


def conv_bn(cin, cout, k=7, stride=1, act=True, zero_bn=False):
    layers = [nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False), nn.BatchNorm1d(cout)]
    if zero_bn:
        nn.init.zeros_(layers[1].weight)
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class BasicBlock1d(nn.Module):
    def __init__(self, cin, cout, stride=1, k=7):
        super().__init__()
        self.convs = nn.Sequential(conv_bn(cin, cout, k, stride), conv_bn(cout, cout, k, 1, act=False, zero_bn=True))
        sc = []
        if stride != 1:
            sc.append(nn.AvgPool1d(stride, stride, ceil_mode=True))
        if cin != cout:
            sc += [nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout)]
        self.shortcut = nn.Sequential(*sc)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.convs(x)
        s = self.shortcut(x)
        if s.shape[-1] != y.shape[-1]:  # ceil/floor mismatch on odd lengths
            s = s[..., : y.shape[-1]]
        return self.act(y + s)


class ResNet1d(nn.Module):
    def __init__(self, in_channels: int = 12, n_classes: int = N_LABELS, widths=(64, 128, 192, 256),
                 blocks=(2, 2, 2, 2), kernel_size: int = 7, stem_widths=(32, 32, 64), dropout=(0.25, 0.5)):
        super().__init__()
        self.in_channels = in_channels
        stem, c = [], in_channels
        for i, w in enumerate(stem_widths):
            stem.append(conv_bn(c, w, kernel_size, stride=2 if i == 0 else 1))
            c = w
        stem.append(nn.MaxPool1d(3, stride=2, padding=1))
        self.stem = nn.Sequential(*stem)
        stages = []
        for si, (w, nb) in enumerate(zip(widths, blocks)):
            for bi in range(nb):
                stages.append(BasicBlock1d(c, w, stride=(1 if si == 0 else 2) if bi == 0 else 1, k=kernel_size))
                c = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2 * c), nn.Dropout(dropout[0]), nn.Linear(2 * c, 128), nn.ReLU(inplace=True),
            nn.BatchNorm1d(128), nn.Dropout(dropout[1]), nn.Linear(128, n_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        h = self.stages(self.stem(x))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h)


def build_model(in_channels: int) -> ResNet1d:
    return ResNet1d(in_channels=in_channels)


if __name__ == "__main__":
    m = build_model(12)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
    print(m(torch.randn(2, 12, 2500)).shape)
