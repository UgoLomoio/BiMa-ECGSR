"""Supplementary Fig. S1 | Validation curves of the seven reconstruction runs.

Validation pooled median PCC per epoch on the fixed 1,024 PTB-XL validation windows (val_select_1024.npy), chest
(predicted V1-V6) and limb (predicted limb leads) groups, from <run>/history.json; vertical line = epoch whose
all-predicted-lead validation PCC was best (the checkpoint used for testing, history 'best' flag, last occurrence).
Run:  python figures/figS1.py
"""
from __future__ import annotations

import json
import os
import re
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

NAME = "FigS1"
REPO = str(CFG.REPO)
SUP_PDF = os.path.join(REPO, "paper", "supplementary")
SUP_PNG = os.path.join(CFG.FIGURES_DIR, "supplementary")
CAP_DIR = os.path.join(CFG.FIGURES_DIR, "captions")
for d in (SUP_PDF, SUP_PNG, CAP_DIR, CFG.SOURCE_DATA_DIR):
    os.makedirs(d, exist_ok=True)
S.apply_style()
FS, FS2 = S.FS_BASE, S.FS_SEC

CONFIGS = ["I", "II", "I,II", "I,II,V1", "I,II,V2", "I,II,V1,V4"]
RUNS = [(f"bima_sr_{c.replace(',', '_')}", f"BiMa-SR, {c.replace(',', '+')}", S.COL["bima_sr"]) for c in CONFIGS]
RUNS.insert(2, ("dibima_sr_II", "DiBiMa-SR, II", S.COL["dibima_sr_1step"]))

rows, best, flag_mismatch = [], {}, {}
for run, lab, col in RUNS:
    h = json.load(open(os.path.join(CFG.RUNS_DIR, run, "history.json")))
    cfgj = json.load(open(os.path.join(CFG.RUNS_DIR, run, "config.json")))
    eps = [r["epoch"] for r in h]
    assert len(set(eps)) == len(eps), run
    for r in h:
        rows.append(dict(run=run, epoch=r["epoch"] + 1, val_pcc_all_pred=r["val_pcc_median"],
                         val_pcc_limb_pred=r["val_pcc_median_limb"], val_pcc_chest_pred=r["val_pcc_median_prec"],
                         val_mae=r["val_mae"], train_loss=r["train_loss"], best_flag=bool(r.get("best", False)),
                         val_steps=cfgj.get("val_steps", 1) if cfgj.get("mode") == "diffusion" else 1))
    # selected checkpoint = argmax of the all-predicted-lead validation PCC (train_sr.py saves best.pth on improvement);
    # cross-checked against the last '*best*' line of train.log (history.json 'best' flags can miss the final epoch)
    am = int(np.argmax([r["val_pcc_median"] for r in h])) + 1
    logb = [int(re.match(r"\[epoch (\d+)\]", ln).group(1)) + 1
            for ln in open(os.path.join(CFG.RUNS_DIR, run, "train.log")) if ln.startswith("[epoch") and "*best*" in ln]
    assert logb and logb[-1] == am, (run, am, logb[-1:])
    best[run] = am
    flag = [r["epoch"] + 1 for r in h if r.get("best")]
    flag_mismatch[run] = (flag[-1] if flag else None) != am
df = pd.DataFrame(rows)
df["selected_epoch"] = df.run.map(best) == df.epoch
print("history best-flag mismatch:", {k: v for k, v in flag_mismatch.items() if v})

W, H = 180.0, 92.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
X0, PW, GX = 13.0, 33.5, 10.3
Y0, PH, GY = 12.0, 26.0, 16.0
axes = []
for k, (run, lab, col) in enumerate(RUNS):
    i, j = divmod(k, 4)
    ax = S.add_axes_mm(fig, X0 + j * (PW + GX), Y0 + i * (PH + GY), PW, PH)
    axes.append(ax)
    d = df[df.run == run]
    ax.plot(d.epoch, d.val_pcc_chest_pred, color=col, lw=1.0, marker="o", ms=1.8, label="chest")
    ax.plot(d.epoch, d.val_pcc_limb_pred, color=col, lw=1.0, ls=(0, (3, 1.5)), marker="s", ms=1.6, mfc="white",
            mew=0.5, label="limb")
    be = best[run]
    ax.axvline(be, color="#555555", lw=0.6, ls=":", zorder=0)
    r = d[d.epoch == be].iloc[0]
    ax.text(be, 1.02, f"best {be}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS2,
            color="#333333")
    ax.set_title(lab, fontsize=FS, loc="left", pad=8.5)
    ax.set_xlim(0.3, d.epoch.max() + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.margins(y=0.1)
    ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    if j == 0:
        ax.set_ylabel("Val. pooled\nmedian PCC")
    if i == 1 or k >= 3:
        ax.set_xlabel("Epoch")
# legend in the free 8th slot
lx = S.add_axes_mm(fig, X0 + 3 * (PW + GX), Y0 + (PH + GY), PW, PH)
lx.axis("off")
lx.legend(handles=[Line2D([], [], color="#444444", marker="o", ms=2, lw=1.0, label="Chest leads (predicted V1–V6)"),
                   Line2D([], [], color="#444444", ls=(0, (3, 1.5)), marker="s", ms=2, mfc="white", lw=1.0,
                          label="Limb leads (predicted)"),
                   Line2D([], [], color="#555555", ls=":", lw=0.6, label="Selected epoch")],
          loc="center left", bbox_to_anchor=(-0.12, 0.5), fontsize=FS2, frameon=False, handlelength=2.2)
for k, ax in enumerate(axes):
    S.letter(fig, ax, "abcdefg"[k], dx_mm=-11.0 if k % 4 == 0 else -8.5, dy_mm=-8.0)
print("bbox issues:", S.bbox_check(fig))
fig.savefig(os.path.join(SUP_PNG, f"{NAME}.png"), dpi=600)
fig.savefig(os.path.join(SUP_PDF, f"{NAME}.pdf"))
fig.savefig(os.path.join(SUP_PNG, f"{NAME}.pdf"))
fig.savefig(os.path.join(SUP_PNG, f"{NAME}.svg"))
S.src(df, NAME, "validation_curves")

summ = df[df.selected_epoch][["run", "epoch", "val_pcc_all_pred", "val_pcc_chest_pred", "val_pcc_limb_pred"]]
lines = "; ".join(f"{lab}: epoch {int(r.epoch)}, chest {r.val_pcc_chest_pred:.3f}, limb {r.val_pcc_limb_pred:.3f}"
                  for (run, lab, _), r in zip(RUNS, [summ[summ.run == x[0]].iloc[0] for x in RUNS]))
panels = "; ".join(f"**{'abcdefg'[k]}**, {lab}" for k, (_, lab, _) in enumerate(RUNS))
nval = len(np.load(os.path.join(CFG.DATA_DIR, "val_select_1024.npy")))
cap = f"""**Supplementary Figure S1 | Validation curves of the seven reconstruction runs.**

Validation pooled median PCC (median over window × lead pairs, z-scored leads) after each training epoch, computed on the fixed set of {nval:,} PTB-XL validation windows (fold 9), for the predicted chest leads (solid line, circles) and the predicted limb leads (dashed line, open squares). Epochs run per model: {", ".join(sorted(set(str(int(df[df.run == r].epoch.max())) for r in df.run.unique())))}. The dotted vertical line marks the selected checkpoint: the epoch with the highest validation pooled median PCC over all predicted leads (argmax of the curve, confirmed against the training log), which is the checkpoint used for every test-set result. Epochs are numbered from 1. DiBiMa-SR (II) was validated with DDIM-6 sampling. For configurations containing I and II, the limb leads III, aVR, aVL and aVF are exactly derivable from the inputs; their validation PCC is near 1 for BiMa-SR, so the chest-lead curve carries the information. {panels}. Selected epochs and validation PCC at the selected epoch: {lines}.

Source data: source_data/FigS1_validation_curves.csv.
"""
open(os.path.join(CAP_DIR, f"{NAME}.md"), "w").write(cap)
print(cap)
