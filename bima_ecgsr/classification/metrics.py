"""Record-level AUROC and record-bootstrap confidence intervals.

``auroc`` is the Mann-Whitney estimate with mid-ranks for ties (identical to sklearn's roc_auc_score).
``bootstrap_auroc`` resamples *records* with replacement (B draws, multinomial counts) and computes the
weighted Mann-Whitney AUROC for every label in every draw from one sort per label, so all labels and
all classifiers evaluated with the same ``seed`` and the same record order share identical resamples
(this makes paired comparisons between classifiers / input types possible later).
"""
from __future__ import annotations

import numpy as np


def auroc(y, s) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def bootstrap_counts(n_records: int, B: int = 1000, seed: int = 0) -> np.ndarray:
    """[B, n] int32 multiplicities of each record in each bootstrap resample."""
    rng = np.random.default_rng(seed)
    out = np.empty((B, n_records), dtype=np.int32)
    for b in range(B):
        out[b] = np.bincount(rng.integers(0, n_records, n_records), minlength=n_records)
    return out


def weighted_auroc_many(y, s, W) -> np.ndarray:
    """Weighted AUROC for each row of W ([B, n] non-negative weights). Ties count 1/2."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    s_o, y_o = s[order], y[order]
    # tie groups
    new = np.r_[True, s_o[1:] != s_o[:-1]]
    gid = np.cumsum(new) - 1
    G = gid[-1] + 1
    res = np.empty(W.shape[0])
    for b0 in range(0, W.shape[0], 64):
        Wb = W[b0:b0 + 64][:, order].astype(np.float64)  # [b, n]
        wpos = Wb * y_o
        wneg = Wb * (~y_o)
        # group sums via cumulative sums at tie-group boundaries
        ends = np.r_[np.flatnonzero(new[1:]), len(s_o) - 1]
        cp = np.cumsum(wpos, axis=1)[:, ends]; cn = np.cumsum(wneg, axis=1)[:, ends]
        gpos = np.diff(np.c_[np.zeros(len(cp)), cp], axis=1)
        gneg = np.diff(np.c_[np.zeros(len(cn)), cn], axis=1)
        neg_below = np.cumsum(gneg, axis=1) - gneg
        num = (gpos * (neg_below + 0.5 * gneg)).sum(1)
        P, N = gpos.sum(1), gneg.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[b0:b0 + 64] = np.where((P > 0) & (N > 0), num / (P * N), np.nan)
    return res


def bootstrap_auroc(Y, S, labels, counts, min_pos: int = 10, exclude=()):
    """Per-label and macro AUROC with percentile 95% CIs.

    Y: [R, K] int labels (1/0, <0 = unknown -> excluded per label); S: [R, K] scores;
    counts: [B, R] bootstrap multiplicities. Labels with < min_pos positive records (among known) or in
    ``exclude`` are skipped. Macro = mean over the evaluated labels within each resample.
    Returns (rows list of dicts, boot matrix [B, n_eval_labels], evaluated label names).
    """
    rows, boots, names = [], [], []
    for j, lab in enumerate(labels):
        known = Y[:, j] >= 0
        npos = int((Y[known, j] == 1).sum()); nneg = int((Y[known, j] == 0).sum())
        if lab in exclude:
            rows.append(dict(label=lab, n_pos=npos, n_neg=nneg, auroc=np.nan, ci_low=np.nan, ci_high=np.nan,
                             status="excluded_invalid_label")); continue
        if npos < min_pos or nneg < min_pos:
            rows.append(dict(label=lab, n_pos=npos, n_neg=nneg, auroc=np.nan, ci_low=np.nan, ci_high=np.nan,
                             status=f"skipped_lt_{min_pos}_pos")); continue
        y, s = Y[known, j] == 1, S[known, j]
        a = auroc(y, s)
        bb = weighted_auroc_many(y, s, counts[:, known])
        lo, hi = np.nanpercentile(bb, [2.5, 97.5])
        rows.append(dict(label=lab, n_pos=npos, n_neg=nneg, auroc=a, ci_low=lo, ci_high=hi, status="ok"))
        boots.append(bb); names.append(lab)
    if names:
        Bm = np.stack(boots, 1)
        mac = float(np.mean([r["auroc"] for r in rows if r["status"] == "ok"]))
        mb = np.nanmean(Bm, 1)
        lo, hi = np.nanpercentile(mb, [2.5, 97.5])
        rows.append(dict(label="macro", n_pos=len(names), n_neg=np.nan, auroc=mac, ci_low=lo, ci_high=hi,
                         status="ok:" + "|".join(names)))
    else:
        Bm = np.zeros((counts.shape[0], 0))
    return rows, Bm, names
