"""Graphical abstract: 12-lead ECG reconstruction from reduced lead sets with BiMa-SR (six input configurations).

(a) one example: measured leads I, II, V1 (5-electrode bedside telemetry) -> BiMa-SR -> predicted chest leads V2-V6 overlaid on
    the real ECG. Example rule: PTB-XL test window at the 90th percentile of BiMa-SR per-window PCC (mean over the 9 predicted
    leads), config I+II+V1 (np.argsort(pcc)[int(0.90 * N)]).
(b) chest-lead pooled median PCC vs configuration (ordered by number of measured leads), BiMa-SR vs the best linear baseline
    (linear filter; Einthoven + linear filter when I and II are measured), three cohorts     <- results/final/sr_overall.csv
(c) record-level macro AUROC of the 12-lead classifier on reconstructions, PTB-XL superclasses; real-ECG reference dashed
(d) the same on the 12-label panel of the two external cohorts (Georgia, China)          <- results/final/cls_macro_auroc.csv
Takeaway line computed from the same tables. Run: python figures/graphical_abstract.py   (matplotlib env)
Source data (source_data/): Graphical_Abstract_example.csv + _example_lead_pcc.csv (a), _fidelity.csv (b),
_auroc.csv (c, d), _takeaway.csv.
"""
from __future__ import annotations
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.plotting import style as S  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA, RUNS = CFG.DATA_DIR, CFG.RUNS_DIR
FINAL = os.path.join(CFG.RESULTS_DIR, "final")
FIG_DIR, SRC_DIR = CFG.FIGURES_DIR, CFG.SOURCE_DATA_DIR
PAPER_FIG = os.path.join(str(CFG.REPO), "paper", "figures")
NAME = "Graphical_Abstract"
W, H = 180.0, 100.0
FS, FS2 = S.FS_BASE, S.FS_SEC            # 7 pt / 6 pt
T0_S, DUR_S = 0.0, 2.5                    # 2.5 s of the 5-s window shown
EX_CFG, EX_RUN, EX_INPUTS = "I,II,V1", "bima_sr_I_II_V1", ["I", "II", "V1"]
SHOW = ["V2", "V3", "V4", "V5", "V6"]      # predicted chest leads shown in (a)
CONFIGS = ["I", "II", "I,II", "I,II,V1", "I,II,V2", "I,II,V1,V4"]       # by number of measured leads
XLAB = ["I", "II", "I+II", "I+II\n+V1", "I+II\n+V2", "I+II\n+V1+V4"]
NLEAD = [len(c.split(",")) for c in CONFIGS]
COHORTS = [("ptbxl_test", "PTB-XL test", "o"), ("georgia", "Georgia (external)", "s"), ("china", "China (external)", "^")]
C_B, C_L, C_IN, C_GT, C_GREY = S.COL["bima_sr"], S.COL["linear"], S.COL["input"], S.COL["GT"], "#555555"


def baseline_key(cfg):                     # best linear baseline: exact limb leads whenever I and II are measured
    return "einthoven_fir" if {"I", "II"} <= set(cfg.split(",")) else "linear_fir"


S.apply_style()

# ------------------------------------------------------------------ data: example window (rule-based)
LI = {ld: i for i, ld in enumerate(S.LEADS)}
PRED = [i for i, ld in enumerate(S.LEADS) if ld not in EX_INPUTS]
meta = pd.read_csv(f"{DATA}/meta_ptbxl_test.csv")
X = np.load(f"{DATA}/X_ptbxl_test.npy", mmap_mode="r")
P = np.load(f"{RUNS}/{EX_RUN}/pred_ptbxl_test.npy", mmap_mode="r")
mt = np.load(f"{RUNS}/{EX_RUN}/metrics_ptbxl_test.npz")
wpcc = mt["pcc"][:, PRED].mean(1)
j = int(np.argsort(wpcc)[int(0.90 * len(wpcc))])
gt, pr = np.asarray(X[j], np.float32), np.asarray(P[j], np.float32)
row = meta.iloc[j]
classes = [c for c in S.CLASSES if c in meta.columns and int(row[c]) == 1] or ["no superclass"]
i0, i1 = int(T0_S * S.FS_HZ), int((T0_S + DUR_S) * S.FS_HZ)
t = np.arange(i0, i1) / S.FS_HZ - T0_S
lead_pcc = {ld: float(mt["pcc"][j, LI[ld]]) for ld in SHOW}

# ------------------------------------------------------------------ data: summary tables
sr = pd.read_csv(f"{FINAL}/sr_overall.csv")
sr = sr[sr.group == "prec_pred"].set_index(["config", "cohort", "method"])
cl = pd.read_csv(f"{FINAL}/cls_macro_auroc.csv").set_index(["config", "cohort", "group", "input"])
eff = pd.read_csv(f"{FINAL}/efficiency.csv").set_index(["config", "method"])
n_par = int(eff.loc[(EX_CFG, "bima_sr"), "n_params"])

fid = []
for cfg in CONFIGS:
    for co, _, _ in COHORTS:
        for role, m in (("bima_sr", "bima_sr"), ("best_linear", baseline_key(cfg))):
            r = sr.loc[(cfg, co, m)]
            fid.append(dict(config=cfg, n_measured=len(cfg.split(",")), cohort=co, role=role, method=m, group="prec_pred",
                            n_records=int(r.n_records), n_windows=int(r.n_windows), n_leads=int(r.n_leads),
                            pcc_pooled_median=r.pcc_pooled_median, pcc_ci_low=r.pcc_ci_low, pcc_ci_high=r.pcc_ci_high))
fid = pd.DataFrame(fid)
au = []
for cfg in CONFIGS:
    for co, grp in (("ptbxl_test", "superclass"), ("georgia", "panel"), ("china", "panel")):
        for role, m in (("bima_sr", "bima_sr"), ("best_linear", baseline_key(cfg)), ("real_12lead", "true_12lead")):
            r = cl.loc[(cfg, co, grp, m)]
            au.append(dict(config=cfg, n_measured=len(cfg.split(",")), cohort=co, label_set=grp, role=role, input=m,
                           auroc=r.auroc, ci_low=r.ci_low, ci_high=r.ci_high))
au = pd.DataFrame(au)

# takeaway, computed from the tables
fw = fid.pivot_table(index=["config", "cohort"], columns="role", values="pcc_pooled_median")
gain = fw.bima_sr - fw.best_linear
n_win, n_tot = int((gain > 0).sum()), len(gain)
sup = au[au.label_set == "superclass"].pivot_table(index="config", columns="role", values="auroc")
gap_k3 = (sup.real_12lead - sup.bima_sr)[[c for c in CONFIGS if len(c.split(",")) >= 3]]
TAKE = (f"BiMa-SR beats linear baselines on chest leads in {n_win}/{n_tot} configuration\u2013cohort pairs; "
        f"\u2265 3 measured leads: PTB-XL AUROC {gap_k3.min():.3f}\u2013{gap_k3.max():.3f} below real ECGs")

# ------------------------------------------------------------------ source data
os.makedirs(SRC_DIR, exist_ok=True)
ex = {"time_s": t}
for ld in EX_INPUTS:
    ex[f"measured_{ld}"] = gt[LI[ld], i0:i1]
for ld in SHOW:
    ex[f"real_{ld}"] = gt[LI[ld], i0:i1]
    ex[f"bima_sr_{ld}"] = pr[LI[ld], i0:i1]
pd.DataFrame(ex).assign(config=EX_CFG, test_window=j, record_id=row.record_id, half=row.half,
                        window_pcc_mean_pred=float(wpcc[j]), rank_rule="argsort(window_pcc)[int(0.90*N)]",
                        n_windows=len(wpcc)).to_csv(f"{SRC_DIR}/{NAME}_example.csv", index=False)
pd.DataFrame([dict(lead=ld, pcc=v) for ld, v in lead_pcc.items()]).to_csv(f"{SRC_DIR}/{NAME}_example_lead_pcc.csv", index=False)
fid.to_csv(f"{SRC_DIR}/{NAME}_fidelity.csv", index=False)          # panel b
au.to_csv(f"{SRC_DIR}/{NAME}_auroc.csv", index=False)     # panels c (label_set superclass) and d (label_set panel, external)
pd.DataFrame([dict(item="chest_pcc_gain_min", value=gain.min()), dict(item="chest_pcc_gain_max", value=gain.max()),
              dict(item="n_pairs_bima_better", value=n_win), dict(item="n_pairs", value=n_tot),
              dict(item="ptbxl_superclass_auroc_gap_k3_min", value=gap_k3.min()),
              dict(item="ptbxl_superclass_auroc_gap_k3_max", value=gap_k3.max()), dict(item="takeaway", value=TAKE)]
             ).to_csv(f"{SRC_DIR}/{NAME}_takeaway.csv", index=False)

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
ov = S.add_axes_mm(fig, 0, 0, W, H); ov.set_xlim(0, W); ov.set_ylim(H, 0); ov.axis("off")


def T(x, y, s, size=FS, **kw):
    kw.setdefault("ha", "left"); kw.setdefault("va", "center")
    return ov.text(x, y, s, fontsize=size, **kw)


def arrow(x0, y0, x1, y1, col="#000000", lw=1.2, ms=8):
    ov.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw,
                                 shrinkA=0, shrinkB=0))


def strip(x, y, w, h, sigs):
    a = S.add_axes_mm(fig, x, y, w, h); a.set_xlim(0, DUR_S); a.axis("off")
    lo, hi = min(s_.min() for s_, *_ in sigs), max(s_.max() for s_, *_ in sigs)
    a.set_ylim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
    for s_, col, lw in sigs:
        a.plot(t, s_, color=col, lw=lw)
    return a


# (a) example ------------------------------------------------------
S.letter(fig, (0.0, 0.6), "a")
T(4.5, 2.6, "Measured: I, II, V1", FS, color=C_IN, fontweight="bold")
T(4.5, 6.3, "5-electrode bedside telemetry", FS2, color=C_GREY)
IX, IW, IH, IY0, IG = 6.5, 21.0, 12.0, 12.0, 2.5
for k, ld in enumerate(EX_INPUTS):
    y = IY0 + k * (IH + IG)
    strip(IX, y, IW, IH, [(gt[LI[ld], i0:i1], C_IN, 0.8)])
    T(IX - 1.0, y + IH / 2, ld, FS, ha="right", color=C_IN, fontweight="bold")
ysb = IY0 + 3 * (IH + IG) - 1.0
ov.plot([IX, IX + IW / DUR_S], [ysb, ysb], color="k", lw=1.0, solid_capstyle="butt")
T(IX + IW / DUR_S / 2, ysb + 2.6, "1 s", FS2, ha="center")
# model block
BX, BY, BW, BH = 32.0, 24.0, 24.0, 21.0
ymid = BY + BH / 2
arrow(IX + IW + 0.8, ymid, BX - 0.4, ymid, col=C_IN)
ov.add_patch(FancyBboxPatch((BX, BY), BW, BH, boxstyle="round,pad=0,rounding_size=1.8", fc="#F4F4F4", ec=C_B, lw=1.1))
T(BX + BW / 2, BY + 3.6, "BiMa-SR", FS, ha="center", color=C_B, fontweight="bold")
n_tok, tw = 6, 2.0
gap = (BW - 5 - n_tok * tw) / (n_tok - 1); xs = [BX + 2.5 + i * (tw + gap) for i in range(n_tok)]
ty = BY + 11.0
for xx in xs:
    ov.add_patch(Rectangle((xx, ty - 1.1), tw, 2.2, fc="white", ec="#555555", lw=0.5))
arrow(xs[0], ty - 3.0, xs[-1] + tw, ty - 3.0, col="#333333", lw=0.7, ms=5)
arrow(xs[-1] + tw, ty + 3.0, xs[0], ty + 3.0, col="#333333", lw=0.7, ms=5)
T(BX + BW / 2, BY + 17.8, "bidirectional Mamba", FS2, ha="center")
T(BX + BW / 2, BY + BH + 3.0, f"{n_par / 1e6:.2f} M parameters", FS2, ha="center", color=C_GREY)
# outputs
OX, OW, OY0, OH, OG = 66.0, 28.0, 8.5, 12.0, 1.4
arrow(BX + BW + 0.4, ymid, OX - 5.5, ymid)
T(OX - 4.5, 2.6, "Predicted chest leads", FS, fontweight="bold")
for k, ld in enumerate(SHOW):
    y = OY0 + k * (OH + OG)
    strip(OX, y, OW, OH, [(gt[LI[ld], i0:i1], C_GT, 0.6), (pr[LI[ld], i0:i1], C_B, 0.85)])
    T(OX - 1.0, y + OH / 2, ld, FS, ha="right", fontweight="bold")
yl = OY0 + len(SHOW) * (OH + OG) + 1.2
lx = OX - 4.5
for lab, col in (("real ECG", C_GT), ("BiMa-SR", C_B)):
    ov.plot([lx, lx + 3.5], [yl, yl], color=col, lw=1.4, solid_capstyle="butt"); T(lx + 4.5, yl, lab, FS2)
    lx += 4.5 + (13.0 if lab == "real ECG" else 12.0)
for k, s_ in enumerate(("Example (rule-based): PTB-XL test window at the", "90th percentile of BiMa-SR per-window PCC",
                        f"(mean over the 9 predicted leads; here {wpcc[j]:.2f}).",
                        "Predicted limb leads not shown.")):
    T(4.5, 64.5 + k * 3.4, s_, FS2, color=C_GREY)

# (b)-(d) summary vs configuration ---------------------------------------
RX, RW = 116.0, 61.0
xc = np.arange(len(CONFIGS), dtype=float)
XLO, XHI = -1.8, len(CONFIGS) - 0.5         # left margin holds the real-ECG reference labels
OFF = {"ptbxl_test": -0.14, "georgia": 0.0, "china": 0.14}
# legend header
LX = RX - 14.0
ov.plot([LX, LX + 3.5], [2.6, 2.6], color=C_B, lw=1.2, solid_capstyle="butt")
T(LX + 4.5, 2.6, "BiMa-SR", FS2, color=C_B, fontweight="bold")
ov.plot([LX + 18.0, LX + 21.5], [2.6, 2.6], color=C_L, lw=1.2, solid_capstyle="butt")
T(LX + 22.5, 2.6, "linear filter / Einthoven + linear filter", FS2, color="#666666", fontweight="bold")
cx = LX
for co, lab, mk in COHORTS:
    ov.plot([cx + 1.2], [6.4], marker=mk, ms=3.4, mfc="white", mec="k", mew=0.6, ls="none")
    T(cx + 3.0, 6.4, lab.split(" (")[0], FS2); cx += 3.0 + {"ptbxl_test": 14.0, "georgia": 11.0, "china": 8.5}[co]
ov.plot([cx, cx + 4.5], [6.4, 6.4], color=C_GT, lw=0.8, ls=(0, (3, 2)))
T(cx + 5.5, 6.4, "real 12-lead ECG", FS2)


def series(ax, df_, ycol, lo, hi, mk, off, ms=3.4):
    for role, col, z in (("best_linear", C_L, 2), ("bima_sr", C_B, 3)):
        d_ = df_[df_.role == role].set_index("config").loc[CONFIGS]
        v = d_[ycol].to_numpy(); e = [v - d_[lo].to_numpy(), d_[hi].to_numpy() - v]
        ax.plot(xc + off, v, color=col, lw=0.8, zorder=z)
        ax.errorbar(xc + off, v, yerr=e, fmt=mk, ms=ms, color=col, mec="k", mew=0.4, elinewidth=0.6, capsize=0, ls="none",
                    zorder=z + 0.5)


axb = S.add_axes_mm(fig, RX, 11.0, RW, 23.0)
for co, lab, mk in COHORTS:
    series(axb, fid[fid.cohort == co], "pcc_pooled_median", "pcc_ci_low", "pcc_ci_high", mk, OFF[co])
axb.set_ylabel("Chest-lead PCC\n(pooled median)", labelpad=2)
axb.set_ylim(0.72, 1.0); axb.set_yticks([0.8, 0.9, 1.0])

axc = S.add_axes_mm(fig, RX, 38.5, RW, 17.5)
d_ = au[au.label_set == "superclass"]
series(axc, d_, "auroc", "ci_low", "ci_high", "o", 0.0)
ref = float(d_[d_.role == "real_12lead"].auroc.iloc[0])
axc.axhline(ref, color=C_GT, lw=0.8, ls=(0, (3, 2)), zorder=1)
axc.text(XLO + 0.05, ref + 0.003, "PTB-XL", fontsize=FS2, va="bottom", ha="left", color=C_GREY)
axc.set_ylabel("Macro AUROC\nPTB-XL, 5 classes", labelpad=2)
axc.set_ylim(0.75, 0.94); axc.set_yticks([0.8, 0.9])

axd = S.add_axes_mm(fig, RX, 60.5, RW, 17.5)
d_ = au[au.label_set == "panel"]
for co, lab, mk in COHORTS[1:]:
    dc = d_[d_.cohort == co]
    series(axd, dc, "auroc", "ci_low", "ci_high", mk, OFF[co] * 0.8)
    r_ = float(dc[dc.role == "real_12lead"].auroc.iloc[0])
    axd.axhline(r_, color=C_GT, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axd.text(XLO + 0.05, r_ + 0.003, lab.split(" (")[0], fontsize=FS2, va="bottom", ha="left", color=C_GREY)
axd.set_ylabel("Macro AUROC\nexternal, 12 labels", labelpad=2)
axd.set_ylim(0.76, 0.96); axd.set_yticks([0.8, 0.9])

for k, ax in enumerate((axb, axc, axd)):
    ax.set_xlim(XLO, XHI); ax.set_xticks(xc)
    ax.set_xticklabels(XLAB if ax is axd else [], fontsize=FS2, linespacing=1.0)
    S.letter(fig, ax, "bcd"[k], dx_mm=-15.0, dy_mm=-1.5)
# brackets: number of measured leads
yb = 87.5
x_per = RW / (XHI - XLO)
for k in sorted(set(NLEAD)):
    idx = [i for i, n in enumerate(NLEAD) if n == k]
    xa, xb_ = RX + (idx[0] - 0.5 - XLO) * x_per + 1.0, RX + (idx[-1] + 0.5 - XLO) * x_per - 1.0
    ov.plot([xa, xb_], [yb, yb], color="#999999", lw=0.6)
    T((xa + xb_) / 2, yb + 2.4, f"{k} lead{'s' if k > 1 else ''}", FS2, ha="center", color=C_GREY)
T(RX + 0.5, yb + 2.4, "measured:", FS2, color=C_GREY)

# takeaway
ov.plot([2, W - 2], [93.0, 93.0], color="#BBBBBB", lw=0.5)
T(W / 2, 96.8, TAKE, FS, ha="center", fontweight="bold")

if __name__ == "__main__":
    issues = S.bbox_check(fig); print("bbox issues:", issues)
    os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(PAPER_FIG, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{FIG_DIR}/{NAME}.{ext}", dpi=600 if ext == "png" else None)
    shutil.copy(f"{FIG_DIR}/{NAME}.pdf", f"{PAPER_FIG}/{NAME}.pdf")
    print("example window", j, "record", row.record_id, "half", row.half, classes, "window PCC", round(float(wpcc[j]), 4),
          "N", len(wpcc), "lead PCC", {k: round(v, 3) for k, v in lead_pcc.items()})
    print("takeaway:", TAKE); print("gain min/max", round(gain.min(), 4), round(gain.max(), 4), "n_par", n_par)
