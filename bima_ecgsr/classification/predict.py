"""Inference utilities shared by training, evaluation and the SR (reconstructed-ECG) experiments.

Input handling (identical at training and prediction time, implemented once in ``zscore_windows``):
each window's leads are z-scored *independently, per window*, with the population standard
deviation over time: x' = (x - mean_t(x)) / std_t(x). Leads whose std is below ``FLAT_EPS``
(flat / constant leads, e.g. a failed reconstruction) are only mean-centred (x' = x - mean), so
they enter the network as ~0 instead of being amplified or producing NaN.
Because the stored windows are already per-lead z-scores, recomputing the z-score is a
no-op on true ECGs (up to float16 rounding) but makes the classifiers invariant to the amplitude
scale / offset of reconstructed ECGs (mV or z units both work).

Usage
-----
    from bima_ecgsr.classification.predict import load_classifier, predict, aggregate_records
    from bima_ecgsr import config
    model, info = load_classifier(config.judge_ckpt())
    p_win = predict(model, X)                 # X: [N, 12, T] numpy/torch, any units -> [N, 17]
    rec_ids, p_rec = aggregate_records(p_win, meta)   # mean over the windows of each (cohort, record_id)

``X`` must contain exactly the model's input leads in canonical order (I, II, III, aVR, aVL, aVF,
V1..V6 restricted to ``info['input_leads']``); pass ``lead_idx=info['lead_idx']`` to select them
from a 12-lead array.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from .classifier import LABELS, build_model  # noqa: E402

FLAT_EPS = 1e-6  # in the units of the input after float32 conversion


def zscore_windows(x: torch.Tensor, eps: float = FLAT_EPS) -> torch.Tensor:
    """Per-window, per-lead z-score over the last (time) axis with a flat-lead guard. x: [B, C, T]."""
    x = x.float()
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    sd = xc.pow(2).mean(dim=-1, keepdim=True).sqrt()
    sd = torch.where(sd < eps, torch.ones_like(sd), sd)
    out = xc / sd
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def load_classifier(ckpt_path: str, device: str | torch.device = "cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = ck["info"]
    model = build_model(len(info["lead_idx"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, info


@torch.no_grad()
def predict(model: torch.nn.Module, X, lead_idx=None, batch_size: int = 256, device=None,
            amp: bool = True) -> np.ndarray:
    """Window-level sigmoid probabilities [N, 17]. X: [N, C, T] array (numpy, memmap or torch).

    lead_idx: optional channel indices to select from X before the z-score (when the model was trained on a
    subset of leads). Applies exactly the training-time input handling.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    n = X.shape[0]
    out = np.empty((n, len(LABELS)), dtype=np.float32)
    for s in range(0, n, batch_size):
        xb = X[s:s + batch_size]
        if lead_idx is not None:
            xb = xb[:, lead_idx]
        xb = torch.as_tensor(np.asarray(xb, dtype=np.float32) if not torch.is_tensor(xb) else xb)
        xb = zscore_windows(xb.to(device, non_blocking=True))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and xb.is_cuda):
            logits = model(xb)
        out[s:s + len(xb)] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def aggregate_records(p_win: np.ndarray, meta, cohort_col="cohort", record_col="record_id"):
    """Record-level probability = mean over the record's windows (both 5-s halves when available).

    meta: pandas DataFrame row-aligned with p_win. Returns (keys DataFrame [cohort, record_id] in
    first-appearance order, p_rec [R, K]).
    """
    import pandas as pd
    key = meta[cohort_col].astype(str) + "\x1f" + meta[record_col].astype(str)
    codes, uniq = pd.factorize(key, sort=False)
    R = len(uniq)
    p_rec = np.zeros((R, p_win.shape[1]), dtype=np.float64)
    np.add.at(p_rec, codes, p_win.astype(np.float64))
    cnt = np.bincount(codes, minlength=R).astype(np.float64)
    p_rec /= cnt[:, None]
    first = np.unique(codes, return_index=True)[1]
    keys = meta.iloc[first][[cohort_col, record_col]].reset_index(drop=True)
    return keys, p_rec.astype(np.float32), codes


def record_labels(meta, labels=LABELS, codes=None):
    """Record-level labels [R, K] (max over windows; labels are record-level so windows agree).

    Unknown = -1 is preserved; superclass columns are set to -1 where has_superclass == 0;
    columns absent in the meta (superclasses on external cohorts) are -1.
    """
    import pandas as pd
    if codes is None:
        key = meta["cohort"].astype(str) + "\x1f" + meta["record_id"].astype(str)
        codes, _ = pd.factorize(key, sort=False)
    R = codes.max() + 1
    Y = np.full((len(meta), len(labels)), -1, dtype=np.int8)
    for j, lab in enumerate(labels):
        if lab in meta.columns:
            Y[:, j] = meta[lab].values.astype(np.int8)
    if "has_superclass" in meta.columns:
        hs = meta["has_superclass"].values == 0
        for j, lab in enumerate(labels):
            if lab in ("NORM", "MI", "STTC", "CD", "HYP"):
                Y[hs, j] = -1
    Yr = np.full((R, len(labels)), -2, dtype=np.int8)
    np.maximum.at(Yr, codes, Y)
    return Yr
