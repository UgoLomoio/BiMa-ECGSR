"""Supplementary Fig. S2 | Per-lead reconstruction accuracy on the external cohorts (Georgia, China).

Median per-window PCC per lead (results/final/sr_per_lead.csv) for BiMa-SR and the baseline (linear filter for I and
II; Einthoven + linear filter for configurations containing I and II); measured input leads are hatched.
Run:  python figures/figS2.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.plotting import style as S  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

NAME = "FigS2"
REPO = str(CFG.REPO)
SUP_PDF = os.path.join(REPO, "paper", "supplementary")
SUP_PNG = os.path.join(CFG.FIGURES_DIR, "supplementary")
CAP_DIR = os.path.join(CFG.FIGURES_DIR, "captions")
for d in (SUP_PDF, SUP_PNG, CAP_DIR, CFG.SOURCE_DATA_DIR):
    os.makedirs(d, exist_ok=True)
S.apply_style()
FS, FS2 = S.FS_BASE, S.FS_SEC
CONFIGS = ["I", "II", "I,II", "I,II,V1", "I,II,V2", "I,II,V1,V4"]
CLAB = {c: c.replace(",", "+") for c in CONFIGS}
LEADS = list(S.LEADS)


def baseline_key(cfg):
    return "einthoven_fir" if {"I", "II"} <= set(cfg.split(",")) else "linear_fir"


BNAME = {"linear_fir": "linear filter", "einthoven_fir": "Einthoven + linear filter"}
pl = pd.read_csv(os.path.join(CFG.RESULTS_DIR, "final", "sr_per_lead.csv"))
n_rec = {c: pd.read_csv(os.path.join(CFG.DATA_DIR, f"meta_{c}.csv"), usecols=["record_id"]) for c in ("georgia", "china")}
n_txt = {c: (m.record_id.nunique(), len(m)) for c, m in n_rec.items()}

PANELS = [("georgia", "bima_sr", "Georgia, BiMa-SR"), ("georgia", "base", "Georgia, baseline"),
          ("china", "bima_sr", "China, BiMa-SR"), ("china", "base", "China, baseline")]
src = []
mats = []
for coh, meth, _ in PANELS:
    M = np.full((len(CONFIGS), 12), np.nan)
    rl = []
    for i, cfg in enumerate(CONFIGS):
        m = meth if meth != "base" else baseline_key(cfg)
        sub = pl[(pl.config == cfg) & (pl.cohort == coh) & (pl.method == m)].set_index("lead")
        for k, ld in enumerate(LEADS):
            r = sub.loc[ld]
            if r.role == "predicted":
                M[i, k] = r.pcc_median
            src.append(dict(cohort=coh, config=cfg, method=m, lead=ld, role=r.role, pcc_median=r.pcc_median,
                            mae_mean=r.mae_mean))
        rl.append(CLAB[cfg] if meth == "bima_sr" else f"{CLAB[cfg]} ({'Einthoven + LF' if m == 'einthoven_fir' else 'LF'})")
    mats.append((M, rl))
allv = np.concatenate([m[0][~np.isnan(m[0])] for m in mats])
vmin = 0.5  # colour floor; cells below are drawn at the floor colour (values printed)
vmax = 1.0

W, H = 180.0, 196.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
X0, PW = 33.0, 128.0
PH, GY, Y0 = 36.0, 11.5, 8.0
cmap = plt.get_cmap("cividis")
for p, ((coh, meth, title), (M, rl)) in enumerate(zip(PANELS, mats)):
    ax = S.add_axes_mm(fig, X0, Y0 + p * (PH + GY), PW, PH)
    im = ax.imshow(np.where(np.isnan(M), vmin, M), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for k in range(12):
            if np.isnan(M[i, k]):
                ax.add_patch(Rectangle((k - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="#BBBBBB",
                                       hatch="////", lw=0.3))
                ax.text(k, i, "input", ha="center", va="center", fontsize=FS2, color=S.COL["input"])
            else:
                rgba = cmap(im.norm(M[i, k]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(k, i, f"{M[i, k]:.3f}", ha="center", va="center", fontsize=FS2,
                        color="black" if lum > 0.5 else "white")
    ax.axvline(5.5, color="white", lw=1.5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(LEADS if p == len(PANELS) - 1 else LEADS)
    ax.set_yticks(range(len(CONFIGS)))
    ax.set_yticklabels(rl)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=FS, loc="left", pad=2.5)
    S.letter(fig, ax, "abcd"[p], dx_mm=-32.0, dy_mm=-5.0)
cax = S.add_axes_mm(fig, X0 + PW + 3.0, Y0, 2.5, 4 * PH + 3 * GY)
cb = fig.colorbar(im, cax=cax, extend="min")
cb.set_label("Median per-window PCC (predicted leads)", fontsize=FS)
cb.outline.set_linewidth(0.4)
print("bbox issues:", S.bbox_check(fig))
fig.savefig(os.path.join(SUP_PNG, f"{NAME}.png"), dpi=600)
for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(SUP_PNG, f"{NAME}.{ext}"))
fig.savefig(os.path.join(SUP_PDF, f"{NAME}.pdf"))
S.src(pd.DataFrame(src), NAME, "per_lead_pcc")

n_below = int((allv < vmin).sum())
assert all((m[0][~np.isnan(m[0])] >= vmin).all() for m, (_, meth, _) in zip(mats, PANELS) if meth == 'bima_sr')
cap = f"""**Supplementary Figure S2 | Per-lead reconstruction accuracy on the external cohorts.**

Median per-window PCC (z-scored 5-s windows) of each reconstructed lead, per input configuration (rows) and lead (columns), on Georgia (n = {n_txt['georgia'][0]:,} records, {n_txt['georgia'][1]:,} windows) and China (Chapman-Shaoxing/Ningbo; n = {n_txt['china'][0]:,} records, {n_txt['china'][1]:,} windows). **a**, Georgia, BiMa-SR. **b**, Georgia, baseline. **c**, China, BiMa-SR. **d**, China, baseline. Baseline: linear filter (LF; per-lead 101-tap FIR fitted by least squares on 4,000 PTB-XL training windows) for I and II; Einthoven + linear filter (exact Einthoven/Goldberger limb leads, LF for the chest leads) for configurations containing I and II. Hatched cells marked "input": measured leads, not evaluated. The white vertical line separates limb (left) and chest (right) leads. For configurations containing I and II the limb leads III, aVR, aVL and aVF are exactly derivable, so the Einthoven + linear filter limb cells are ≈1 by construction and the chest-lead cells are the fair comparison. Common colour scale across panels (cividis, {vmin:.1f}–1; cells below {vmin:.1f}, {n_below} in total, all baseline cells, take the floor colour and keep their printed value). Values in each cell.

Source data: source_data/FigS2_per_lead_pcc.csv (also contains the per-lead mean MAE).
"""
open(os.path.join(CAP_DIR, f"{NAME}.md"), "w").write(cap)
print(cap[:300])
