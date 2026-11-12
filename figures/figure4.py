"""Figure 4 | Diagnostic readability of reconstructed ECGs by a classifier trained on real ECGs only.

(a) record-level macro AUROC (95% CI) per input configuration: BiMa-SR, DiBiMa-SR one-step (II), baseline
    (linear filter for I and II; Einthoven + linear filter for configurations containing I and II); real-ECG reference.
(b) delta macro AUROC vs real ECGs (paired record bootstrap, B = 1000), open markers = 95% CI includes 0.
(c) per-superclass AUROC heatmap on PTB-XL (BiMa-SR rows per configuration + real 12-lead row).
(d) per-label AUROC difference (reconstructed - real) on Georgia and China for II and I+II+V1.

Inputs (read only): results/final/cls_macro_auroc.csv, cls_paired_deltas.csv, cls_per_label_auroc.csv,
                    meta_<cohort>.csv (record counts), judge12/summary.json (classifier parameter count).
Run:  python figures/figure4.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.plotting import style as S  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "Figure4"
FIN = os.path.join(CFG.RESULTS_DIR, "final")
PAPER_FIG = os.path.join(str(CFG.REPO), "paper", "figures")
CAP_DIR = os.path.join(CFG.FIGURES_DIR, "captions")
for d in (PAPER_FIG, CAP_DIR, CFG.SOURCE_DATA_DIR):
    os.makedirs(d, exist_ok=True)

S.apply_style()
FS, FS2 = S.FS_BASE, S.FS_SEC
C_B, C_D, C_LIN, C_GT = S.COL["bima_sr"], S.COL["dibima_sr_1step"], S.COL["linear"], S.COL["GT"]
C_EIN_EDGE = "#4D4D4D"

# configuration order used everywhere in the paper (by number of measured leads)
CONFIGS = ["I", "II", "I,II", "I,II,V1", "I,II,V2", "I,II,V1,V4"]
CLAB = {c: c.replace(",", "+") for c in CONFIGS}


def baseline_key(cfg):
    return "einthoven_fir" if ("I" in cfg.split(",") and "II" in cfg.split(",")) else "linear_fir"


BNAME = {"linear_fir": "Linear filter", "einthoven_fir": "Einthoven + linear filter"}
BMARK = {"linear_fir": "s", "einthoven_fir": "D"}
GROUPS = [("ptbxl_test", "superclass", "PTB-XL, superclasses"), ("ptbxl_test", "panel", "PTB-XL, 12-label panel"),
          ("georgia", "panel", "Georgia, panel"), ("china", "panel", "China, panel")]
OFF = {"bima_sr": -0.2, "dibima_sr_onestep": 0.0, "base": 0.2}

mac = pd.read_csv(os.path.join(FIN, "cls_macro_auroc.csv"))
dlt = pd.read_csv(os.path.join(FIN, "cls_paired_deltas.csv"))
lab = pd.read_csv(os.path.join(FIN, "cls_per_label_auroc.csv"))
judge = json.load(open(os.path.join(CFG.CLS_DIR, "judge12", "summary.json")))
n_rec = {c: pd.read_csv(os.path.join(CFG.DATA_DIR, f"meta_{c}.csv"), usecols=["record_id"]).record_id.nunique()
         for c in CFG.COHORTS}
n_super = lab[(lab.cohort == "ptbxl_test") & (lab.group == "superclass") & (lab.label == "NORM") &
              (lab.input == "true_12lead")].eval("n_pos + n_neg").iloc[0]


def get_mac(cfg, coh, grp, inp):
    r = mac[(mac.config == cfg) & (mac.cohort == coh) & (mac.group == grp) & (mac.input == inp)]
    return None if r.empty else r.iloc[0]


def get_dlt(cfg, coh, grp, a):
    r = dlt[(dlt.config == cfg) & (dlt.cohort == coh) & (dlt.group == grp) & (dlt.a == a) & (dlt.b == "true_12lead")]
    return None if r.empty else r.iloc[0]


# ------------------------------------------------------------------ figure geometry (mm from top-left)
W, H = 180.0, 174.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
X0, PW, GAP = 14.0, 34.0, 9.5
xs = [X0 + i * (PW + GAP) for i in range(4)]
YA, HA = 14.0, 32.0
YB, HB = 51.0, 29.0
YC, HC = 106.0, 62.0

# ------------------------------------------------------------------ a + b
src_a, src_b = [], []
ax_a, ax_b = [], []
for j, (coh, grp, title) in enumerate(GROUPS):
    a = S.add_axes_mm(fig, xs[j], YA, PW, HA)
    b = S.add_axes_mm(fig, xs[j], YB, PW, HB)
    ax_a.append(a); ax_b.append(b)
    real = get_mac("II", coh, grp, "true_12lead")
    # real ECG reference: identical across configurations (same records, same classifier); verify
    reals = mac[(mac.cohort == coh) & (mac.group == grp) & (mac.input == "true_12lead")].auroc
    assert reals.max() - reals.min() < 1e-12
    a.axhspan(real.ci_low, real.ci_high, color="#000000", alpha=0.10, lw=0, zorder=0)
    a.axhline(real.auroc, color=C_GT, ls=(0, (3, 2)), lw=0.8, zorder=1)
    src_a.append(dict(cohort=coh, group=grp, config="all", input="true_12lead", auroc=real.auroc,
                      ci_low=real.ci_low, ci_high=real.ci_high))
    b.axhline(0, color=C_GT, ls=(0, (3, 2)), lw=0.8, zorder=1)
    for i, cfg in enumerate(CONFIGS):
        series = [("bima_sr", C_B, "o", OFF["bima_sr"])]
        if cfg == "II":
            series.append(("dibima_sr_onestep", C_D, "^", OFF["dibima_sr_onestep"]))
        bk = baseline_key(cfg)
        series.append((bk, C_LIN, BMARK[bk], OFF["base"]))
        for key, col, mk, off in series:
            r = get_mac(cfg, coh, grp, key)
            ec = C_EIN_EDGE if key == "einthoven_fir" else col
            a.errorbar(i + off, r.auroc, yerr=[[r.auroc - r.ci_low], [r.ci_high - r.auroc]], fmt=mk, ms=3.2,
                       mfc=col, mec=ec, mew=0.6, ecolor=col, elinewidth=0.8, capsize=0, zorder=3)
            src_a.append(dict(cohort=coh, group=grp, config=cfg, input=key, auroc=r.auroc, ci_low=r.ci_low,
                              ci_high=r.ci_high))
            d = get_dlt(cfg, coh, grp, key)
            sig = not (d.ci_low <= 0 <= d.ci_high)
            b.errorbar(i + off, d.delta_macro_boot_mean,
                       yerr=[[d.delta_macro_boot_mean - d.ci_low], [d.ci_high - d.delta_macro_boot_mean]],
                       fmt=mk, ms=3.2, mfc=col if sig else "white", mec=ec, mew=0.7, ecolor=col, elinewidth=0.8,
                       capsize=0, zorder=3)
            src_b.append(dict(cohort=coh, group=grp, config=cfg, a=key, b="true_12lead",
                              delta_auroc=d.delta_macro_boot_mean, ci_low=d.ci_low, ci_high=d.ci_high,
                              p_two_sided=d.p_two_sided, ci_excludes_0=sig))
    for ax in (a, b):
        ax.set_xlim(-0.6, len(CONFIGS) - 0.4)
        ax.set_xticks(range(len(CONFIGS)))
        ax.grid(axis="y", lw=0.3, color="#DDDDDD", zorder=0)
    a.set_xticklabels([])
    b.set_xticklabels([CLAB[c] for c in CONFIGS], rotation=40, ha="right", rotation_mode="anchor")
    a.set_title(title, fontsize=FS, loc="left", pad=2.5)
    a.margins(y=0.08)
for ax in ax_b:  # shared delta axis
    ax.set_ylim(min(x.get_ylim()[0] for x in ax_b), max(0.012, max(x.get_ylim()[1] for x in ax_b)))
for ax in ax_b[1:]:
    ax.set_yticklabels([])
ax_a[0].set_ylabel("Macro AUROC")
ax_b[0].set_ylabel("ΔAUROC vs real")
ax_b[0].text(0.03, 0.96, "0 = real-ECG AUROC", transform=ax_b[0].transAxes, fontsize=FS2, va="top", ha="left",
             color="#333333")

# legend for a/b (top strip)
hand = [Line2D([], [], marker="o", ls="none", mfc=C_B, mec=C_B, ms=3.5, label="BiMa-SR"),
        Line2D([], [], marker="^", ls="none", mfc=C_D, mec=C_D, ms=3.5, label="DiBiMa-SR one-step (II)"),
        Line2D([], [], marker="s", ls="none", mfc=C_LIN, mec=C_LIN, ms=3.5, label="Linear filter"),
        Line2D([], [], marker="D", ls="none", mfc=C_LIN, mec=C_EIN_EDGE, ms=3.2, label="Einthoven + linear filter"),
        Line2D([], [], color=C_GT, ls=(0, (3, 2)), lw=0.8, label="Real 12-lead ECG (band: 95% CI)"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555", ms=3.5, label="ΔAUROC 95% CI includes 0")]
fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(X0 / W, 1 - 0.6 / H), ncol=3, fontsize=FS2,
           handletextpad=0.3, columnspacing=1.1, borderaxespad=0)

# ------------------------------------------------------------------ c: superclass heatmap (BiMa-SR, PTB-XL)
SUPER = list(S.CLASSES)
rows_c, mat = [], []
for cfg in CONFIGS:
    vals = []
    for cl in SUPER:
        r = lab[(lab.config == cfg) & (lab.cohort == "ptbxl_test") & (lab.group == "superclass") &
                (lab.input == "bima_sr") & (lab.label == cl)].iloc[0]
        vals.append(r.auroc)
        rows_c.append(dict(row=CLAB[cfg], input="bima_sr", label=cl, auroc=r.auroc, ci_low=r.ci_low,
                           ci_high=r.ci_high, n_pos=r.n_pos, n_neg=r.n_neg))
    mat.append(vals)
vals = []
for cl in SUPER:
    r = lab[(lab.config == "II") & (lab.cohort == "ptbxl_test") & (lab.group == "superclass") &
            (lab.input == "true_12lead") & (lab.label == cl)].iloc[0]
    vals.append(r.auroc)
    rows_c.append(dict(row="Real 12-lead", input="true_12lead", label=cl, auroc=r.auroc, ci_low=r.ci_low,
                       ci_high=r.ci_high, n_pos=r.n_pos, n_neg=r.n_neg))
mat.append(vals)
mat = np.array(mat)
rlabels = [CLAB[c] for c in CONFIGS] + ["Real 12-lead"]
axc = S.add_axes_mm(fig, X0 + 3.0, YC, 40.0, HC - 6.0)
vmin, vmax = np.floor(mat.min() * 50) / 50, np.ceil(mat.max() * 50) / 50
im = axc.imshow(mat, cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
for i in range(mat.shape[0]):
    for k in range(mat.shape[1]):
        v = mat[i, k]
        rgba = im.cmap(im.norm(v))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        axc.text(k, i, f"{v:.3f}", ha="center", va="center", fontsize=FS2, color="black" if lum > 0.5 else "white")
axc.axhline(len(CONFIGS) - 0.5, color="white", lw=1.5)
axc.set_xticks(range(len(SUPER)))
axc.set_xticklabels(SUPER)
axc.set_yticks(range(len(rlabels)))
axc.set_yticklabels(rlabels)
axc.tick_params(length=0)
for sp in axc.spines.values():
    sp.set_visible(False)
axc.set_title("PTB-XL, BiMa-SR by input leads", fontsize=FS, loc="left", pad=2.5)
cax = S.add_axes_mm(fig, X0 + 45.0, YC, 2.2, HC - 6.0)
cb = fig.colorbar(im, cax=cax)
cb.ax.set_title("AUROC", fontsize=FS2, loc="left", pad=2.5)
cb.outline.set_linewidth(0.4)

# ------------------------------------------------------------------ d: per-label deltas on external cohorts
PANEL = ["AFIB", "AFLT", "1AVB", "PAC", "PVC", "SBRAD", "STACH", "WPW", "LVH", "RVH", "CRBBB", "CLBBB"]
DCFG = ["II", "I,II,V1"]
DCOH = [("georgia", "Georgia"), ("china", "China")]
XD0, DW, DG = 80.0, 21.0, 3.0
src_d = []
ax_d = []
ypos = {l: i for i, l in enumerate(PANEL)}
for j, ((coh, cname), cfg) in enumerate([(c, f) for c in DCOH for f in DCFG]):
    ax = S.add_axes_mm(fig, XD0 + j * (DW + DG) + (3.0 if j >= 2 else 0.0), YC, DW, HC - 6.0)
    ax_d.append(ax)
    ax.axvline(0, color=C_GT, ls=(0, (3, 2)), lw=0.8, zorder=1)
    bk = baseline_key(cfg)
    sub = lab[(lab.config == cfg) & (lab.cohort == coh) & (lab.group == "panel")]
    for l in PANEL:
        t = sub[(sub.input == "true_12lead") & (sub.label == l)].iloc[0]
        if t.status != "ok":
            ax.text(-0.2, ypos[l], "not scored", fontsize=FS2, ha="center", va="center", color="#777777",
                    gid="allow_notscored")
            src_d.append(dict(cohort=coh, config=cfg, label=l, input="all", auroc_real=np.nan, auroc=np.nan,
                              delta_auroc=np.nan, status=t.status, n_pos=t.n_pos))
            continue
        for key, col, mk, dy in (("bima_sr", C_B, "o", -0.17), (bk, C_LIN, BMARK[bk], 0.17)):
            r = sub[(sub.input == key) & (sub.label == l)].iloc[0]
            dv = r.auroc - t.auroc
            ec = C_EIN_EDGE if key == "einthoven_fir" else col
            ax.plot([0, dv], [ypos[l] + dy] * 2, color=col, lw=0.6, zorder=2)
            ax.plot(dv, ypos[l] + dy, marker=mk, ms=3.0, mfc=col, mec=ec, mew=0.5, ls="none", zorder=3)
            src_d.append(dict(cohort=coh, config=cfg, label=l, input=key, auroc_real=t.auroc, auroc=r.auroc,
                              delta_auroc=dv, status=r.status, n_pos=r.n_pos))
    ax.set_ylim(len(PANEL) - 0.5, -0.5)
    ax.set_yticks(range(len(PANEL)))
    ax.set_yticklabels(PANEL if j == 0 else [])
    ax.grid(axis="x", lw=0.3, color="#DDDDDD", zorder=0)
    ax.set_title(f"{cname}\n{CLAB[cfg]}", fontsize=FS, loc="left", pad=2.5)
sd = pd.DataFrame(src_d)
lo, hi = np.nanmin(sd.delta_auroc), np.nanmax(sd.delta_auroc)
pad = 0.06 * (hi - lo)
for ax in ax_d:
    ax.set_xlim(lo - pad, max(hi, 0) + pad)
    ax.set_xticks([-0.4, -0.2, 0.0])
    ax.set_xticklabels(["−0.4", "", "0"])
fig.text((XD0 + 2 * DW + DG + 1.5) / W, 1 - (YC + HC + 1.0) / H, "ΔAUROC (reconstructed − real)", ha="center",
         va="top", fontsize=FS)
hand_d = [Line2D([], [], marker="o", ls="none", mfc=C_B, mec=C_B, ms=3.2, label="BiMa-SR"),
          Line2D([], [], marker="s", ls="none", mfc=C_LIN, mec=C_LIN, ms=3.2, label="Linear filter (II)"),
          Line2D([], [], marker="D", ls="none", mfc=C_LIN, mec=C_EIN_EDGE, ms=3.0,
                 label="Einthoven + linear filter (I+II+V1)")]
fig.legend(handles=hand_d, loc="upper left", bbox_to_anchor=(XD0 / W, 1 - (YC - 9.0) / H), ncol=3, fontsize=FS2,
           handletextpad=0.3, columnspacing=1.0, borderaxespad=0)

# ------------------------------------------------------------------ panel letters
S.letter(fig, (1.0, 0.5), "a")
S.letter(fig, (1.0, YB - 2.0), "b")
S.letter(fig, (1.0, YC - 9.0), "c")
S.letter(fig, (XD0 - 11.0, YC - 9.0), "d")

issues = S.bbox_check(fig)
print("bbox issues:", issues)
paths = S.save(fig, NAME)
fig.savefig(os.path.join(PAPER_FIG, f"{NAME}.pdf"))
print("saved", paths)

# ------------------------------------------------------------------ source data
S.src(pd.DataFrame(src_a), NAME, "a_macro_auroc")
S.src(pd.DataFrame(src_b), NAME, "b_delta_auroc")
S.src(pd.DataFrame(rows_c), NAME, "c_superclass_heatmap")
S.src(sd, NAME, "d_label_deltas")

# ------------------------------------------------------------------ caption (numbers formatted from the CSVs)
def f3(x):
    return f"{x:.3f}"


def mtxt(cfg, coh, grp, key):
    r = get_mac(cfg, coh, grp, key)
    return f"{f3(r.auroc)} ({f3(r.ci_low)}–{f3(r.ci_high)})"


def dtxt(cfg, coh, grp, key):
    d = get_dlt(cfg, coh, grp, key)
    return f"{d.delta_macro_boot_mean:+.3f} ({d.ci_low:+.3f} to {d.ci_high:+.3f})".replace("-", "−")


nsig = pd.DataFrame(src_b)
ns_b = nsig[(nsig.a == "bima_sr") & (~nsig.ci_excludes_0)]
ns_list = "; ".join(f"{CLAB[r.config]} on {dict((g[0] + g[1], g[2]) for g in GROUPS)[r.cohort + r.group]}"
                    for r in ns_b.itertuples())
skipped = lab[lab.status.isin(["skipped_lt_10_pos", "excluded_invalid_label"])][["cohort", "label", "status", "n_pos"]] \
    .drop_duplicates(["cohort", "label"])
skip_txt = "; ".join(f"{dict(ptbxl_test='PTB-XL', georgia='Georgia', china='China')[r.cohort]} {r.label} "
                     + ("(not scored)" if r.status == "excluded_invalid_label" else f"({int(r.n_pos)} positive records)")
                     for r in skipped.itertuples())
cap = f"""**Figure 4 | Diagnostic readability of reconstructed ECGs by a classifier trained on real ECGs only.**

One 12-lead 1D-ResNet classifier ({judge['n_params'] / 1e6:.2f} M parameters), trained only on real PTB-XL ECGs (folds 1–8), is applied unchanged to real 12-lead ECGs and to 12-lead ECGs reconstructed from each input configuration (measured input leads re-inserted). Scores are averaged over the two 5-s windows of a record; all AUROCs are record level. Cohorts: PTB-XL test fold 10 (n = {n_rec['ptbxl_test']:,} records; {int(n_super):,} with a diagnostic superclass label), Georgia (external, n = {n_rec['georgia']:,}) and China (Chapman-Shaoxing/Ningbo, external, n = {n_rec['china']:,}); records with an unknown label are excluded from that label. 95% CIs: record bootstrap, B = 1,000, identical resamples for every input type. Baseline: linear filter (per-lead 101-tap FIR, least squares) for I and II; Einthoven + linear filter (exact Einthoven/Goldberger limb leads, linear filter for chest leads) for configurations containing I and II.

**a**, Macro AUROC (95% CI) versus input configuration for BiMa-SR (orange circles), the baseline (grey squares: linear filter; grey diamonds: Einthoven + linear filter) and DiBiMa-SR one-step (blue triangle, II only), on the PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP) and on the 12-label rhythm/conduction/hypertrophy panel (AFIB, AFLT, 1AVB, PAC, PVC, SBRAD, STACH, WPW, LVH, RVH, CRBBB, CLBBB) of PTB-XL, Georgia and China. Dashed line and band: AUROC on the real 12-lead ECGs and its 95% CI. Macro AUROC averages the scored labels; labels with fewer than 10 positive records are skipped ({skip_txt}). Example: PTB-XL superclasses, real {mtxt('II', 'ptbxl_test', 'superclass', 'true_12lead')}; from lead II alone BiMa-SR {mtxt('II', 'ptbxl_test', 'superclass', 'bima_sr')}, linear filter {mtxt('II', 'ptbxl_test', 'superclass', 'linear_fir')}; from I+II+V1+V4 BiMa-SR {mtxt('I,II,V1,V4', 'ptbxl_test', 'superclass', 'bima_sr')}.

**b**, Difference in macro AUROC between reconstructed and real ECGs (mean over the paired bootstrap resamples, 95% percentile CI; B = 1,000). Filled markers: CI excludes 0; open markers: CI includes 0 (difference from the real-ECG AUROC not significant at the 5% level). Shared y axis. BiMa-SR differences whose CI includes 0: {ns_list if ns_list else 'none'}. Example, China panel from lead II: BiMa-SR {dtxt('II', 'china', 'panel', 'bima_sr')}, linear filter {dtxt('II', 'china', 'panel', 'linear_fir')}.

**c**, Per-superclass AUROC of the classifier on PTB-XL ECGs reconstructed by BiMa-SR (one row per input configuration) and on the real 12-lead ECGs (bottom row), n = {int(n_super):,} records; positives per class are listed in the source data. Values printed in each cell; colour scale cividis.

**d**, Per-label AUROC difference (reconstructed − real; point estimates, no CI) on the external cohorts for the single-lead configuration II and the 5-electrode configuration I+II+V1. Orange circles: BiMa-SR; grey square: linear filter (II); grey diamond: Einthoven + linear filter (I+II+V1). "not scored": Georgia WPW (fewer than 10 positive records) and China AFLT (label not scored in China). Per-label AUROCs with 95% CIs for every configuration and cohort are in Supplementary Fig. S4.

Source data: source_data/Figure4_a_macro_auroc.csv, Figure4_b_delta_auroc.csv, Figure4_c_superclass_heatmap.csv, Figure4_d_label_deltas.csv.
"""
open(os.path.join(CAP_DIR, f"{NAME}.md"), "w").write(cap)
print(cap)
