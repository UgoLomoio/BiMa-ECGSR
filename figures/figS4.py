"""Supplementary Fig. S4 | Per-label AUROC of the real-ECG classifier on real and reconstructed ECGs, every
configuration and cohort (results/final/cls_per_label_auroc.csv).
Rows: PTB-XL superclasses, PTB-XL panel, Georgia panel, China panel; columns: input configurations.
Run:  python figures/figS4.py
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
from matplotlib.lines import Line2D  # noqa: E402

NAME = "FigS4"
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
SUPER = list(S.CLASSES)
PANEL = ["AFIB", "AFLT", "1AVB", "PAC", "PVC", "SBRAD", "STACH", "WPW", "LVH", "RVH", "CRBBB", "CLBBB"]
ROWS = [("ptbxl_test", "superclass", "PTB-XL, superclasses", SUPER), ("ptbxl_test", "panel", "PTB-XL, panel", PANEL),
        ("georgia", "panel", "Georgia, panel", PANEL), ("china", "panel", "China, panel", PANEL)]
C_B, C_D, C_LIN, C_GT, C_EIN_EDGE = S.COL["bima_sr"], S.COL["dibima_sr_1step"], S.COL["linear"], S.COL["GT"], "#4D4D4D"


def baseline_key(cfg):
    return "einthoven_fir" if {"I", "II"} <= set(cfg.split(",")) else "linear_fir"


lab = pd.read_csv(os.path.join(CFG.RESULTS_DIR, "final", "cls_per_label_auroc.csv"))
lab = lab[lab.label != "macro"]

W = 180.0
X0, PW, GX = 15.0, 25.0, 2.4
ROW_GAP, Y0 = 13.0, 9.0
slot = 4.0  # mm per label
heights = [len(r[3]) * slot for r in ROWS]
H = Y0 + sum(heights) + ROW_GAP * len(ROWS) + 2.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
src = []
y = Y0 + 5.0
for ri, (coh, grp, rtitle, labels) in enumerate(ROWS):
    sub0 = lab[(lab.cohort == coh) & (lab.group == grp)]
    ok = sub0[sub0.status == "ok"]
    lo = np.nanmin(ok.ci_low); hi = np.nanmax(ok.ci_high)
    pad = 0.03 * (hi - lo)
    axes = []
    for ci, cfg in enumerate(CONFIGS):
        ax = S.add_axes_mm(fig, X0 + ci * (PW + GX), y, PW, heights[ri])
        axes.append(ax)
        sub = sub0[sub0.config == cfg]
        bk = baseline_key(cfg)
        series = [("true_12lead", C_GT, "|", 0.0), ("bima_sr", C_B, "o", -0.26), (bk, C_LIN, "s" if bk == "linear_fir" else "D", 0.26)]
        if cfg == "II":
            series = [("true_12lead", C_GT, "|", 0.0), ("bima_sr", C_B, "o", -0.3), ("dibima_sr_onestep", C_D, "^", -0.13),
                      (bk, C_LIN, "s", 0.26)]
        for li, l in enumerate(labels):
            t = sub[(sub.input == "true_12lead") & (sub.label == l)].iloc[0]
            if t.status != "ok":
                ax.text((lo + hi) / 2, li, "not scored", ha="center", va="center", fontsize=FS2, color="#777777")
                src.append(dict(cohort=coh, group=grp, config=cfg, label=l, input="all", auroc=np.nan, ci_low=np.nan,
                                ci_high=np.nan, n_pos=t.n_pos, status=t.status))
                continue
            for key, col, mk, off in series:
                r = sub[(sub.input == key) & (sub.label == l)].iloc[0]
                ec = C_EIN_EDGE if key == "einthoven_fir" else col
                if key == "true_12lead":
                    ax.plot([r.ci_low, r.ci_high], [li, li], color=C_GT, lw=0.5, zorder=2)
                    ax.plot(r.auroc, li, marker="|", ms=5, mew=1.0, color=C_GT, zorder=4)
                else:
                    ax.plot([r.ci_low, r.ci_high], [li + off] * 2, color=col, lw=0.6, zorder=2)
                    ax.plot(r.auroc, li + off, marker=mk, ms=2.4, mfc=col, mec=ec, mew=0.4, ls="none", zorder=3)
                src.append(dict(cohort=coh, group=grp, config=cfg, label=l, input=key, auroc=r.auroc, ci_low=r.ci_low,
                                ci_high=r.ci_high, n_pos=r.n_pos, status=r.status))
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ci == 0 else [])
        x0, x1 = lo - pad, min(1.01, hi + pad)
        ax.set_xlim(x0, x1)
        step = 0.1 if (x1 - x0) < 0.45 else 0.2
        tk = [v for v in np.round(np.arange(np.ceil(x0 / step) * step, x1 + 1e-9, step), 2)
              if x0 + 0.1 * (x1 - x0) <= v <= x1 - 0.07 * (x1 - x0)]
        ax.set_xticks(tk)
        ax.set_xticklabels([f"{v:.1f}" for v in tk])
        ax.grid(axis="x", lw=0.3, color="#DDDDDD", zorder=0)
        ax.tick_params(axis="x", labelsize=FS2)
        if ri == 0:
            ax.set_title(CLAB[cfg], fontsize=FS, loc="center", pad=10.0)
    axes[0].text(0.0, 1.0, rtitle, transform=axes[0].transAxes, ha="left", va="bottom", fontsize=FS,
                 fontweight="bold")
    fig.text((X0 + 3 * (PW + GX) - GX / 2) / W, 1 - (y + heights[ri] + 6.2) / H, "AUROC", ha="center", va="top",
             fontsize=FS2)
    S.letter(fig, (1.0, y - 5.0), "abcd"[ri])
    y += heights[ri] + ROW_GAP
hand = [Line2D([], [], marker="|", ls="-", color=C_GT, ms=5, mew=1.0, lw=0.5, label="Real 12-lead ECG"),
        Line2D([], [], marker="o", ls="-", color=C_B, ms=2.8, lw=0.6, label="BiMa-SR"),
        Line2D([], [], marker="^", ls="-", color=C_D, ms=2.8, lw=0.6, label="DiBiMa-SR one-step (II)"),
        Line2D([], [], marker="s", ls="-", color=C_LIN, ms=2.8, lw=0.6, label="Linear filter (I, II)"),
        Line2D([], [], marker="D", ls="-", color=C_LIN, mfc=C_LIN, mec=C_EIN_EDGE, ms=2.6, lw=0.6,
               label="Einthoven + linear filter (configurations with I and II)")]
fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(X0 / W, 1 - 0.3 / H), ncol=5, fontsize=FS2,
           borderaxespad=0, columnspacing=1.0, handletextpad=0.3, handlelength=1.6)
print("H mm", H, "bbox issues:", S.bbox_check(fig))
fig.savefig(os.path.join(SUP_PNG, f"{NAME}.png"), dpi=600)
for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(SUP_PNG, f"{NAME}.{ext}"))
fig.savefig(os.path.join(SUP_PDF, f"{NAME}.pdf"))
SD = pd.DataFrame(src)
S.src(SD, NAME, "per_label_auroc")
n_rec = {c: pd.read_csv(os.path.join(CFG.DATA_DIR, f"meta_{c}.csv"), usecols=["record_id"]).record_id.nunique()
         for c in CFG.COHORTS}
ns = SD[SD.input == "all"].drop_duplicates(["cohort", "group", "label"])
ns_txt = "; ".join(f"{dict(ptbxl_test='PTB-XL', georgia='Georgia', china='China')[r.cohort]} {r.label} ("
                   + ("label not scored in this cohort" if r.status == "excluded_invalid_label" else f"{int(r.n_pos)} positive records")
                   + ")" for r in ns.itertuples())
cap = f"""**Supplementary Figure S4 | Per-label AUROC on real and reconstructed ECGs for every input configuration and cohort.**

The 12-lead classifier trained only on real PTB-XL ECGs (folds 1–8) is applied unchanged to real ECGs and to reconstructed ECGs. Record-level AUROC (mean score of the two 5-s windows of a record) with 95% CI (record bootstrap, B = 1,000). Columns: input configuration. **a**, PTB-XL test, diagnostic superclasses (n = {int(lab[(lab.cohort == 'ptbxl_test') & (lab.group == 'superclass') & (lab.label == 'NORM')].eval('n_pos + n_neg').iloc[0]):,} records with a superclass label). **b**, PTB-XL test, 12-label panel (n = {n_rec['ptbxl_test']:,} records). **c**, Georgia, panel (n = {n_rec['georgia']:,}). **d**, China (Chapman-Shaoxing/Ningbo), panel (n = {n_rec['china']:,}). Black tick and line: real 12-lead ECG AUROC and 95% CI (identical across columns); orange circles: BiMa-SR; blue triangles: DiBiMa-SR one-step (II only); grey squares: linear filter (I, II); grey diamonds: Einthoven + linear filter (configurations containing I and II, whose limb leads are exact). Records with an unknown label are excluded from that label; per-label positive and negative counts are in the source data. Shared x axis within each row. Not scored: {ns_txt}.

Source data: source_data/FigS4_per_label_auroc.csv.
"""
open(os.path.join(CAP_DIR, f"{NAME}.md"), "w").write(cap)
print(cap)
