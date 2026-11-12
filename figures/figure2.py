"""Figure 2 | Reconstruction accuracy across input configurations and cohorts.

a  chest-lead (predicted V1-V6) pooled median PCC [95% record-bootstrap CI], BiMa-SR vs comparator, 3 cohorts
b  per-lead pooled median PCC heatmap, BiMa-SR on PTB-XL test (configs x 12 leads) + BiMa-SR minus comparator
c  chest-lead MAE [95% CI], BiMa-SR vs comparator, 3 cohorts
d  lead II: BiMa-SR, DiBiMa-SR one-step / DDIM-6, linear filter (all predicted leads) + paired per-record difference
Inputs: results/final/sr_overall.csv, sr_per_lead.csv, sr_paired.csv.   Run: python figures/figure2.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figlib as F  # noqa: E402
from _figlib import S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "Figure2"
S.apply_style()
FS, FS2 = S.FS_BASE, S.FS_SEC
CFGS = [c for _, c, _ in F.CONFIGS]
XL = [F.CFG_LABEL[c] for c in CFGS]

ov = F.load_csv("sr_overall")
pl = F.load_csv("sr_per_lead")
pa = F.load_csv("sr_paired")

W_MM, H_MM = 180, 175
fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
A = lambda x, y, w, h: S.add_axes_mm(fig, x, y, w, h)  # noqa: E731


def comp_rows(group, metric_cols):
    rows = []
    for cfg in CFGS:
        for m in ("bima_sr", F.baseline(cfg)):
            for coh, _ in F.COHORTS:
                r = ov[(ov.config == cfg) & (ov.cohort == coh) & (ov.method == m) & (ov.group == group)]
                assert len(r) == 1, (cfg, coh, m, group)
                r = r.iloc[0]
                rows.append(dict(panel_config=F.CFG_LABEL[cfg], config=cfg, cohort=coh, method=m,
                                 method_name=F.NAMES[m], group=group, n_leads=int(r.n_leads),
                                 n_records=int(r.n_records), n_windows=int(r.n_windows),
                                 **{k: float(r[k]) for k in metric_cols}))
    return pd.DataFrame(rows)


def dotrow(x0, y0, w_each, h, df, val, lo, hi, ylab, ylim, letter, cue):
    axs = []
    for j, (coh, cl) in enumerate(F.COHORTS):
        ax = A(x0 + j * w_each, y0, w_each - 1.2, h)
        for k, cfg in enumerate(CFGS):
            for m, dx in (("bima_sr", -0.14), (F.baseline(cfg), 0.14)):
                r = df[(df.config == cfg) & (df.cohort == coh) & (df.method == m)].iloc[0]
                ax.errorbar(k + dx, r[val], yerr=[[r[val] - r[lo]], [r[hi] - r[val]]], fmt=F.MARK[m], ms=3.6,
                            mfc=F.COL[m], mec="black" if m == "bima_sr" else F.COL[m], mew=0.4,
                            ecolor="black", elinewidth=0.6, capsize=0, zorder=3)
        ax.set_xlim(-0.6, len(CFGS) - 0.4)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(CFGS)))
        ax.set_xticklabels(XL, rotation=40, ha="right", rotation_mode="anchor")
        ax.grid(axis="y", lw=0.3, color="#DDDDDD", zorder=0)
        ax.set_title(cl, fontsize=FS, loc="center", pad=2)
        if j == 0:
            ax.set_ylabel(ylab)
        else:
            ax.tick_params(labelleft=False)
        axs.append(ax)
    axs[0].text(-0.02, 1.0, cue, transform=axs[0].transAxes, fontsize=FS2, ha="right", va="bottom",
                color="#555555")
    S.letter(fig, axs[0], letter, dx_mm=-12, dy_mm=-3)
    return axs


# ------------------------------------------------------------------ a: chest-lead PCC
a_df = comp_rows("prec_pred", ["pcc_pooled_median", "pcc_ci_low", "pcc_ci_high"])
F.src(a_df, NAME, "a_chest_pcc")
axa = dotrow(14, 9, 54.5, 36, a_df, "pcc_pooled_median", "pcc_ci_low", "pcc_ci_high",
             "Chest-lead PCC\n(pooled median)", (0.72, 1.0), "a", "")
leg = [Line2D([], [], ls="", marker="o", ms=3.6, mfc=F.COL["bima_sr"], mec="black", mew=0.4, label="BiMa-SR"),
       Line2D([], [], ls="", marker="s", ms=3.6, mfc=F.COL["linear_fir"], mec=F.COL["linear_fir"],
              label="Linear filter (input I or II)"),
       Line2D([], [], ls="", marker="D", ms=3.2, mfc=F.COL["einthoven_fir"], mec=F.COL["einthoven_fir"],
              label="Einthoven + linear filter (configs with I+II)")]
fig.legend(handles=leg, loc="upper right", bbox_to_anchor=(1 - 2 / W_MM, 1 - 0.5 / H_MM), ncol=3,
           fontsize=FS2, handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
axa[0].text(0.03, 0.04, "higher = better", transform=axa[0].transAxes, fontsize=FS2, color="#555555")

# ------------------------------------------------------------------ b: per-lead heatmaps (PTB-XL test)
coh = "ptbxl_test"
H = np.full((len(CFGS), 12), np.nan); D = np.full_like(H, np.nan); IN = np.zeros_like(H, bool)
b_rows = []
for i, cfg in enumerate(CFGS):
    for j, lead in enumerate(F.LEADS):
        rb = pl[(pl.config == cfg) & (pl.cohort == coh) & (pl.method == "bima_sr") & (pl.lead == lead)].iloc[0]
        rc = pl[(pl.config == cfg) & (pl.cohort == coh) & (pl.method == F.baseline(cfg)) & (pl.lead == lead)].iloc[0]
        IN[i, j] = rb.role == "input"
        if not IN[i, j]:
            H[i, j] = rb.pcc_median; D[i, j] = rb.pcc_median - rc.pcc_median
        b_rows.append(dict(config=cfg, panel_config=F.CFG_LABEL[cfg], cohort=coh, lead=lead, role=rb.role,
                           bima_sr_pcc_median=float(rb.pcc_median), comparator=F.baseline(cfg),
                           comparator_pcc_median=float(rc.pcc_median),
                           diff_bima_minus_comparator=float(rb.pcc_median - rc.pcc_median) if not IN[i, j] else np.nan))
F.src(pd.DataFrame(b_rows), NAME, "b_per_lead_ptbxl_test")


def heat(ax, M, cmap, vmin, vmax, signed, cb_ax, cb_label, cb_ticks, ylabels=True):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    norm = im.norm
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if IN[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#F3DDEB", ec="white", lw=0.6))
                ax.text(j, i, "input", ha="center", va="center", fontsize=FS2, color=F.COL["input"],
                        fontfamily="Liberation Sans Narrow")
                continue
            v = M[i, j]
            rgba = cmap(norm(v)); lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, F.fmt3(v, signed), ha="center", va="center", fontsize=FS2,
                    fontfamily="Liberation Sans Narrow", color="white" if lum < 0.5 else "black")
    ax.set_xticks(range(12)); ax.set_xticklabels(F.LEADS)
    ax.set_yticks(range(len(CFGS))); ax.set_yticklabels(XL if ylabels else [])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True); ax.set_yticks(np.arange(-0.5, len(CFGS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    ax.axvline(5.5, color="black", lw=0.8)
    cb = fig.colorbar(im, cax=cb_ax, orientation="horizontal", ticks=cb_ticks)
    cb_ax.xaxis.set_ticks_position("top"); cb_ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=FS2, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    cb_ax.text(-0.03, 0.5, cb_label, transform=cb_ax.transAxes, ha="right", va="center", fontsize=FS2)
    return cb


YB = 70
axb1 = A(18, YB, 79, 34); cb1 = A(62, YB - 3.2, 34, 1.6)
heat(axb1, H, plt.get_cmap("viridis"), 0.75, 1.0, False, cb1, "BiMa-SR PCC", [0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
axb1.set_xlabel("Lead (BiMa-SR, PTB-XL test)", labelpad=1.5)
vmax = 0.30
axb2 = A(100.5, YB, 79, 34); cb2 = A(140, YB - 3.2, 34, 1.6)
heat(axb2, D, plt.get_cmap("PuOr_r"), -vmax, vmax, True, cb2, "\u0394PCC, BiMa-SR \u2212 comparator",
     [-0.3, -0.15, 0, 0.15, 0.3], ylabels=False)
axb2.set_xlabel("Lead (BiMa-SR minus comparator, PTB-XL test)", labelpad=1.5)
S.letter(fig, axb1, "b", dx_mm=-16, dy_mm=-5)

# ------------------------------------------------------------------ c: chest-lead MAE
c_df = comp_rows("prec_pred", ["mae_mean", "mae_ci_low", "mae_ci_high"])
F.src(c_df, NAME, "c_chest_mae")
axc = dotrow(14, 126, 33, 32, c_df, "mae_mean", "mae_ci_low", "mae_ci_high",
             "Chest-lead MAE\n(z units)", (0.15, 0.46), "c", "")
axc[0].text(0.04, 0.04, "lower = better", transform=axc[0].transAxes, fontsize=FS2, color="#555555")

# ------------------------------------------------------------------ d: lead II, regression vs diffusion
MD = ["bima_sr", "dibima_sr_onestep", "dibima_sr_ddim6", "linear_fir"]
d_rows = []
axd1 = A(128, 126, 24, 32)
for j, (coh, cl) in enumerate(F.COHORTS):
    for k, m in enumerate(MD):
        r = ov[(ov.config == "II") & (ov.cohort == coh) & (ov.method == m) & (ov.group == "all_pred")].iloc[0]
        x = j + (k - 1.5) * 0.2
        axd1.errorbar(x, r.pcc_pooled_median, yerr=[[r.pcc_pooled_median - r.pcc_ci_low],
                                                    [r.pcc_ci_high - r.pcc_pooled_median]],
                      fmt=F.MARK[m], ms=3.2, mfc=F.COL[m], mec="black" if m == "bima_sr" else F.COL[m], mew=0.4,
                      ecolor="black", elinewidth=0.6, zorder=3)
        d_rows.append(dict(part="pcc_all_pred", config="II", cohort=coh, method=m, method_name=F.NAMES[m],
                           n_records=int(r.n_records), n_windows=int(r.n_windows), n_leads=int(r.n_leads),
                           value=float(r.pcc_pooled_median), ci_low=float(r.pcc_ci_low), ci_high=float(r.pcc_ci_high),
                           frac_a_better=np.nan, wilcoxon_p=np.nan))
axd1.set_xticks(range(3)); axd1.set_xticklabels([c for _, c in F.COHORTS], rotation=40, ha="right",
                                                 rotation_mode="anchor")
axd1.set_xlim(-0.55, 2.55); axd1.set_ylim(0.75, 0.94)
axd1.set_ylabel("PCC, all predicted leads\n(pooled median; input II)")
axd1.grid(axis="y", lw=0.3, color="#DDDDDD", zorder=0)
legd = [Line2D([], [], ls="", marker=F.MARK[m], ms=3.2, mfc=F.COL[m], mec="black" if m == "bima_sr" else F.COL[m],
               mew=0.4, label=F.NAMES[m].replace("DiBiMa-SR ", "DiBiMa-SR\n")) for m in MD]
axd1.legend(handles=legd, loc="lower left", bbox_to_anchor=(-0.25, 1.02), ncol=2, fontsize=FS2,
            handletextpad=0.2, columnspacing=0.6, borderaxespad=0, labelspacing=0.3)
axd2 = A(164.5, 126, 12.5, 32)
for j, (coh, cl) in enumerate(F.COHORTS):
    r = pa[(pa.config == "II") & (pa.cohort == coh) & (pa.group == "all_pred") & (pa.metric == "pcc")
           & (pa.a == "bima_sr") & (pa.b == "dibima_sr_onestep")].iloc[0]
    axd2.errorbar(j, r.mean_diff, yerr=[[r.mean_diff - r.ci_low], [r.ci_high - r.mean_diff]], fmt="o", ms=3.2,
                  mfc=F.COL["bima_sr"], mec=F.COL["dibima_sr_onestep"], mew=1.0, ecolor="black", elinewidth=0.6,
                  zorder=3)
    axd2.text(j, r.ci_high + 0.0012, f"{100 * r.frac_a_better:.0f}%", ha="center", va="bottom", fontsize=FS2)
    d_rows.append(dict(part="paired_per_record_pcc_diff", config="II", cohort=coh, method="bima_sr - dibima_sr_onestep",
                       method_name="BiMa-SR minus DiBiMa-SR (one-step)", n_records=int(r.n_records), n_windows=np.nan,
                       n_leads=11, value=float(r.mean_diff), ci_low=float(r.ci_low), ci_high=float(r.ci_high),
                       frac_a_better=float(r.frac_a_better), wilcoxon_p=float(r.wilcoxon_p)))
axd2.axhline(0, color="black", lw=0.5)
axd2.set_xlim(-0.6, 2.6); axd2.set_ylim(-0.002, 0.029)
axd2.set_xticks(range(3)); axd2.set_xticklabels([c for _, c in F.COHORTS], rotation=55, ha="right",
                                                 rotation_mode="anchor")
axd2.set_title("% records\nBiMa-SR better", fontsize=FS2, loc="center", pad=2)
axd2.set_ylabel("\u0394 per-record PCC\nBiMa-SR \u2212 one-step", labelpad=1)
F.src(pd.DataFrame(d_rows), NAME, "d_leadII_models")
S.letter(fig, axd1, "d", dx_mm=-13, dy_mm=-3)


def caption():
    def o(cfg, coh, m, g, k="pcc_pooled_median"):
        return float(ov[(ov.config == cfg) & (ov.cohort == coh) & (ov.method == m) & (ov.group == g)].iloc[0][k])
    n = {coh: ov[ov.cohort == coh].iloc[0] for coh, _ in F.COHORTS}
    ncoh = "; ".join(f"{cl} {int(n[c].n_records):,} records / {int(n[c].n_windows):,} windows" for c, cl in F.COHORTS)
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    pr = pa[(pa.config == "II") & (pa.group == "all_pred") & (pa.metric == "pcc") & (pa.a == "bima_sr")
            & (pa.b == "dibima_sr_onestep")].set_index("cohort")
    def pfmt(p):
        return "< 10⁻³⁰⁰" if p == 0 else f"= {p:.1e}"
    d_txt = "; ".join(f"{cl} {pr.loc[c].mean_diff:+.4f} [{pr.loc[c].ci_low:+.4f}, {pr.loc[c].ci_high:+.4f}], "
                      f"{100 * pr.loc[c].frac_a_better:.0f}% of records, Wilcoxon P {pfmt(pr.loc[c].wilcoxon_p)}"
                      for c, cl in F.COHORTS)
    t = f"""**Figure 2 | Reconstruction accuracy across input configurations and cohorts.**
Configurations are ordered by the number of measured leads. Metrics are computed on 5-s windows (500 Hz, per-lead z-scored) and on predicted leads only; measured input leads are excluded. Because III, aVR, aVL and aVF are exactly derivable from I and II, the fair comparison for configurations containing I+II is on the predicted chest leads, which is the primary metric in **a** and **c**. Comparator: the linear filter (per-lead FIR, 101 taps per input lead, least squares on 4,000 PTB-XL training windows) for inputs I or II alone, and Einthoven + linear filter (exact Einthoven/Goldberger limb leads, linear filter for chest leads) for configurations containing I and II; on chest leads the two comparators are identical. Cohorts: {ncoh}.
**a**, Chest-lead pooled median Pearson correlation coefficient (PCC; median over all window × predicted-chest-lead pairs) with 95% record-bootstrap confidence intervals (B = 1,000; both windows of a record resampled together); most intervals are narrower than the markers. On PTB-XL test, BiMa-SR reaches {f3(o('I','ptbxl_test','bima_sr','prec_pred'))} (input I) to {f3(o('I,II,V1,V4','ptbxl_test','bima_sr','prec_pred'))} (I+II+V1+V4), versus {f3(o('I','ptbxl_test','linear_fir','prec_pred'))} to {f3(o('I,II,V1,V4','ptbxl_test','einthoven_fir','prec_pred'))} for the comparator.
**b**, Per-lead pooled median PCC of BiMa-SR on PTB-XL test (left; n = {int(n['ptbxl_test'].n_windows):,} windows per cell) and BiMa-SR minus comparator (right; diverging scale centred at 0). Pink cells, measured input leads (not evaluated). The vertical line separates limb from chest leads. For configurations containing I+II, the Einthoven + linear filter comparator is exact on limb leads (per-lead PCC > 0.9999 on PTB-XL test; the residual is numerical), so the small negative limb differences are expected; for the same reason the all-predicted-lead PCC for I+II+V1+V4 on PTB-XL test is lower for BiMa-SR ({f3(o('I,II,V1,V4','ptbxl_test','bima_sr','all_pred'))}) than for Einthoven + linear filter ({f3(o('I,II,V1,V4','ptbxl_test','einthoven_fir','all_pred'))}), although BiMa-SR is higher on chest leads.
**c**, Chest-lead mean absolute error (MAE, z units; mean over window × lead pairs) with 95% record-bootstrap CIs (B = 1,000).
**d**, Lead II input only. Left, pooled median PCC over all 11 predicted leads with 95% record-bootstrap CIs for BiMa-SR (regression), DiBiMa-SR (diffusion, DDPM T = 20) sampled in one step or with DDIM-6, and the linear filter. Right, paired per-record difference in PCC (per-record PCC = mean over a record's predicted leads and windows), BiMa-SR minus DiBiMa-SR one-step: mean difference [95% record-bootstrap CI]; the label above each point is the percentage of records in which BiMa-SR has the higher per-record PCC. {d_txt}.
Source data: source_data/Figure2_a_chest_pcc.csv, Figure2_b_per_lead_ptbxl_test.csv, Figure2_c_chest_mae.csv, Figure2_d_leadII_models.csv."""
    return F.write_caption(NAME, t)


if __name__ == "__main__":
    print("bbox issues:", S.bbox_check(fig))
    print("saved:", F.save(fig, NAME))
    print("caption:", caption())
