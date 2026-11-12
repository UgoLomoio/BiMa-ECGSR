"""Figure 5 | Generalisation, diagnostic subgroups and efficiency.

a  BiMa-SR chest-lead (predicted V1-V6) pooled median PCC per PTB-XL superclass x configuration (PTB-XL test), and
   BiMa-SR minus comparator (results/final/sr_per_class.csv, group prec_pred)
b  external drop: chest-lead pooled median PCC on Georgia / China minus PTB-XL test, BiMa-SR vs comparator.
   Point estimates are differences of sr_overall.csv values; 95% CIs come from independent record bootstraps of the two
   cohorts (B = 1000, the seeds of final_stats.py), recomputed here from metrics_<cohort>*.npz.  The recomputed point
   estimates and single-cohort CIs are checked against sr_overall.csv.
c  efficiency (results/final/efficiency.csv): parameters, GPU ms per 10-s record (batch 2), GPU windows/s (batch 128),
   CPU ms per record (2 threads; the linear filter was timed with 1 thread only)
Run: python figures/figure5.py   (about 2 min; bootstrap cache in source_data/Figure5_b_external_drop.csv, --recompute)
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

NAME = "Figure5"
S.apply_style()
FS, FS2 = S.FS_BASE, S.FS_SEC
CFGS = [c for _, c, _ in F.CONFIGS]
XL = [F.CFG_LABEL[c] for c in CFGS]
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]

ov = F.load_csv("sr_overall")
pc = F.load_csv("sr_per_class")
ef = F.load_csv("efficiency")

W_MM, H_MM = 180, 112
fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
A = lambda x, y, w, h: S.add_axes_mm(fig, x, y, w, h)  # noqa: E731

# ------------------------------------------------------------------ a: superclass heatmaps
M = np.zeros((len(CFGS), 5)); Dm = np.zeros_like(M); a_rows = []
for i, cfg in enumerate(CFGS):
    for j, lab in enumerate(SUPER):
        rb = pc[(pc.config == cfg) & (pc.cohort == "ptbxl_test") & (pc.method == "bima_sr") & (pc.label == lab)
                & (pc.group == "prec_pred")].iloc[0]
        rc = pc[(pc.config == cfg) & (pc.cohort == "ptbxl_test") & (pc.method == F.baseline(cfg)) & (pc.label == lab)
                & (pc.group == "prec_pred")].iloc[0]
        M[i, j] = rb.pcc_pooled_median; Dm[i, j] = rb.pcc_pooled_median - rc.pcc_pooled_median
        a_rows.append(dict(config=cfg, panel_config=F.CFG_LABEL[cfg], cohort="ptbxl_test", label=lab,
                           group="prec_pred", n_records=int(rb.n_records), bima_sr_pcc=float(rb.pcc_pooled_median),
                           comparator=F.baseline(cfg), comparator_pcc=float(rc.pcc_pooled_median),
                           diff=float(M[i, j] - rc.pcc_pooled_median)))
a_df = pd.DataFrame(a_rows); F.src(a_df, NAME, "a_superclass")
nrec = a_df.drop_duplicates("label").set_index("label").n_records


def heat(ax, Mx, cmap, vmin, vmax, signed, cb_ax, cb_label, ticks, ylabels):
    im = ax.imshow(Mx, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            rgba = cmap(im.norm(Mx[i, j])); lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, F.fmt3(Mx[i, j], signed), ha="center", va="center", fontsize=FS2,
                    fontfamily="Liberation Sans Narrow", color="white" if lum < 0.5 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{l}\n{nrec[l]}" for l in SUPER])
    ax.set_yticks(range(len(CFGS))); ax.set_yticklabels(XL if ylabels else [])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True); ax.set_yticks(np.arange(-0.5, len(CFGS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, cax=cb_ax, orientation="horizontal", ticks=ticks)
    cb_ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=FS2, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
    cb_ax.set_title(cb_label, fontsize=FS2, loc="center", pad=7)
    return cb


axa1 = A(17, 15, 34, 33); cba1 = A(19, 7.0, 30, 1.5)
heat(axa1, M, plt.get_cmap("viridis"), 0.75, 1.0, False, cba1, "BiMa-SR chest-lead PCC", [0.75, 0.85, 0.95], True)
axa2 = A(53, 15, 34, 33); cba2 = A(55, 7.0, 30, 1.5)
heat(axa2, Dm, plt.get_cmap("PuOr_r"), -0.3, 0.3, True, cba2, "\u0394PCC, BiMa-SR \u2212 comparator",
     [-0.3, 0, 0.3], False)
fig.text(52 / W_MM, 1 - 53.8 / H_MM, "PTB-XL superclass (records, PTB-XL test)", ha="center", va="top", fontsize=FS)
S.letter(fig, (2, 2), "a")

# ------------------------------------------------------------------ b: external drop with bootstrap CIs
cache = os.path.join(F.SRC_DIR, f"{NAME}_b_external_drop.csv")
if os.path.exists(cache) and "--recompute" not in sys.argv:
    b_df = pd.read_csv(cache)
else:
    boots, rows, checks = {}, [], []
    for coh, _ in F.COHORTS:
        codes, counts = F.boot_counts(coh)
        for cfg in CFGS:
            ci = F.chest_idx(cfg)
            for m in ("bima_sr", F.baseline(cfg)):
                v = np.load(F.metrics_path(m, cfg, coh))["pcc"][:, ci].astype(np.float64)
                est, dist = F.wmedian_boot_dist(v.ravel(), np.repeat(codes, len(ci)), counts)
                boots[(coh, cfg, m)] = (est, dist)
                r = ov[(ov.config == cfg) & (ov.cohort == coh) & (ov.method == m) & (ov.group == "prec_pred")].iloc[0]
                lo, hi = np.percentile(dist, [2.5, 97.5])
                checks.append(max(abs(est - r.pcc_pooled_median), abs(lo - r.pcc_ci_low), abs(hi - r.pcc_ci_high)))
        print("bootstrapped", coh, flush=True)
    print("max |recomputed - sr_overall| (estimate, CI bounds):", max(checks))
    assert max(checks) < 1e-6
    for coh in ("georgia", "china"):
        for cfg in CFGS:
            for m in ("bima_sr", F.baseline(cfg)):
                e1, d1 = boots[(coh, cfg, m)]; e0, d0 = boots[("ptbxl_test", cfg, m)]
                lo, hi = np.percentile(d1 - d0, [2.5, 97.5])
                rows.append(dict(config=cfg, panel_config=F.CFG_LABEL[cfg], external_cohort=coh, method=m,
                                 method_name=F.NAMES[m], group="prec_pred", pcc_external=e1, pcc_ptbxl_test=e0,
                                 pcc_drop=e1 - e0, drop_ci_low=lo, drop_ci_high=hi, B=len(d1),
                                 ci_method="independent record bootstraps of the two cohorts, percentile"))
    b_df = pd.DataFrame(rows); F.src(b_df, NAME, "b_external_drop")

axb = []
for k, coh in enumerate(("georgia", "china")):
    ax = A(112 + k * 34, 15, 32.5, 33)
    for i, cfg in enumerate(CFGS):
        for m, dx in (("bima_sr", -0.14), (F.baseline(cfg), 0.14)):
            r = b_df[(b_df.config == cfg) & (b_df.external_cohort == coh) & (b_df.method == m)].iloc[0]
            ax.errorbar(i + dx, r.pcc_drop, yerr=[[r.pcc_drop - r.drop_ci_low], [r.drop_ci_high - r.pcc_drop]], fmt=F.MARK[m],
                        ms=3.4, mfc=F.COL[m], mec="black" if m == "bima_sr" else F.COL[m], mew=0.4, ecolor="black",
                        elinewidth=0.6, zorder=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(-0.6, len(CFGS) - 0.4); ax.set_ylim(-0.075, 0.005)
    ax.set_xticks(range(len(CFGS))); ax.set_xticklabels(XL, rotation=40, ha="right", rotation_mode="anchor")
    ax.grid(axis="y", lw=0.3, color="#DDDDDD", zorder=0)
    ax.set_title(f"{F.COH_LABEL[coh]} \u2212 PTB-XL test", fontsize=FS, loc="center", pad=2)
    if k == 0:
        ax.set_ylabel("\u0394 chest-lead PCC\n(external \u2212 internal)")
    else:
        ax.tick_params(labelleft=False)
    axb.append(ax)
axb[1].text(0.97, 0.04, "closer to 0 = smaller drop", transform=axb[1].transAxes, ha="right", fontsize=FS2,
            color="#555555")
legb = [Line2D([], [], ls="", marker="o", ms=3.4, mfc=F.COL["bima_sr"], mec="black", mew=0.4, label="BiMa-SR"),
        Line2D([], [], ls="", marker="s", ms=3.4, mfc=F.COL["linear_fir"], mec=F.COL["linear_fir"],
               label="Linear filter (input I or II)"),
        Line2D([], [], ls="", marker="D", ms=3.0, mfc=F.COL["einthoven_fir"], mec=F.COL["einthoven_fir"],
               label="Einthoven + linear filter")]
fig.legend(handles=legb, loc="upper right", bbox_to_anchor=(1 - 1 / W_MM, 1 - 1.0 / H_MM), ncol=3, fontsize=FS2,
           handletextpad=0.2, columnspacing=0.9, borderaxespad=0)
S.letter(fig, (99, 2), "b")

# ------------------------------------------------------------------ c: efficiency
MC = ["bima_sr", "dibima_sr_onestep", "dibima_sr_ddim6", "linear_fir"]
ef["cpu_ms"] = ef.cpu_ms_per_record_threads2.where(ef.method != "linear_fir", ef.cpu_ms_per_record_threads1)
ef["cpu_threads"] = np.where(ef.method == "linear_fir", 1, 2)
F.src(ef.assign(panel_config=ef.config.map(F.CFG_LABEL))[
    ["config", "panel_config", "method", "run", "n_params", "nfe", "gpu_ms_per_record", "gpu_windows_per_s",
     "cpu_ms", "cpu_threads"]], NAME, "c_efficiency")
cols = [("n_params", "Parameters", (5e2, 2e7)),
        ("gpu_ms_per_record", "GPU ms per 10-s record\n(batch 2)", (10, 1000)),
        ("gpu_windows_per_s", "GPU windows per s\n(batch 128)", (10, 1000)),
        ("cpu_ms", "CPU ms per 10-s record\n(2 threads; linear filter 1 thread)", (0.5, 5e4))]


def valtxt(col, v):
    if col == "n_params":
        return f"{v / 1e6:.3g}M" if v >= 1e6 else f"{v:,.0f}"
    return f"{v:,.0f}" if v >= 1000 else f"{v:.3g}"


def human(v, _):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


yc = 70
for k, (col, lab, lim) in enumerate(cols):
    ax = A(30 + k * 37.5, yc, 33, 27)
    for r_i, m in enumerate(MC):
        sub = ef[ef.method == m]
        y = len(MC) - 1 - r_i
        if sub[col].isna().all():
            ax.text(np.sqrt(lim[0] * lim[1]), y, "not run on GPU", ha="center", va="center", fontsize=FS2,
                    color="#6E6E6E")
            continue
        main = sub[sub.config == "II"].iloc[0]
        others = sub[sub.config != "II"]
        ax.plot([lim[0], main[col]], [y, y], color="#CCCCCC", lw=0.6, zorder=1)
        if len(others):
            ax.plot(others[col], [y] * len(others), ls="", marker="|", ms=4, color=F.COL[m], mew=0.7, zorder=2)
        ax.plot(main[col], y, ls="", marker=F.MARK[m], ms=3.6, mfc=F.COL[m],
                mec="black" if m == "bima_sr" else F.COL[m], mew=0.4, zorder=3)
        ax.text(main[col], y + 0.2, valtxt(col, main[col]), ha="center", va="bottom", fontsize=FS2,
                fontfamily="Liberation Sans Narrow")
    ax.set_xscale("log"); ax.set_xlim(*lim); ax.set_ylim(-0.6, len(MC) - 0.2)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_major_formatter(FuncFormatter(human))
    ax.set_yticks(range(len(MC)))
    ax.set_yticklabels([F.NAMES[m] for m in MC[::-1]] if k == 0 else [])
    ax.set_xlabel(lab, labelpad=1.5)
    ax.grid(axis="x", which="major", lw=0.3, color="#DDDDDD", zorder=0)
    ax.text(0.98, 1.02, "higher = better" if col == "gpu_windows_per_s" else
            ("lower = better" if col != "n_params" else ""), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS2, color="#555555")
S.letter(fig, (2, yc - 5), "c")
fig.text(31 / W_MM, 1 - (yc - 4.6) / H_MM,
         "Marker: lead-II model; ticks: the other five input configurations (BiMa-SR, linear filter)",
         ha="left", va="top", fontsize=FS2, color="#333333")


def caption():
    n = {c: ov[ov.cohort == c].iloc[0] for c, _ in F.COHORTS}
    bd = b_df.set_index(["config", "external_cohort", "method"])
    def drop(cfg, coh, m):
        r = bd.loc[(cfg, coh, m)]
        return f"{r.pcc_drop:+.3f} [{r.drop_ci_low:+.3f}, {r.drop_ci_high:+.3f}]"
    smaller = [(F.CFG_LABEL[c], F.COH_LABEL[coh]) for coh in ("georgia", "china") for c in CFGS
               if bd.loc[(c, coh, "bima_sr")].drop_ci_low > bd.loc[(c, coh, F.baseline(c))].drop_ci_high]
    larger = [(F.CFG_LABEL[c], F.COH_LABEL[coh]) for coh in ("georgia", "china") for c in CFGS
              if bd.loc[(c, coh, "bima_sr")].drop_ci_high < bd.loc[(c, coh, F.baseline(c))].drop_ci_low]
    lst = lambda L: ", ".join(f"{a} ({b})" for a, b in L) or "none"  # noqa: E731
    ext_ok = all(float(ov[(ov.config == c) & (ov.cohort == coh) & (ov.method == "bima_sr") & (ov.group == "prec_pred")]
                       .pcc_pooled_median.iloc[0]) >
                 float(ov[(ov.config == c) & (ov.cohort == coh) & (ov.method == F.baseline(c)) & (ov.group == "prec_pred")]
                       .pcc_pooled_median.iloc[0]) for c in CFGS for coh in ("georgia", "china"))
    e = ef.set_index(["config", "method"])
    bs = ef[ef.method == "bima_sr"]; lf = ef[ef.method == "linear_fir"]
    t = f"""**Figure 5 | Generalisation, diagnostic subgroups and efficiency.**
Accuracy is the chest-lead pooled median PCC (median over window × predicted-chest-lead pairs, z-scored 5-s windows); configurations are ordered by the number of measured leads. Chest leads are used for every configuration, including I and II, for consistency with Figure 2a; limb leads are excluded because the Einthoven + linear filter comparator is exact on them for configurations containing I+II. Comparator: linear filter for inputs I or II alone; Einthoven + linear filter otherwise (identical chest leads).
**a**, PTB-XL test, per diagnostic superclass (NORM, normal; MI, myocardial infarction; STTC, ST/T change; CD, conduction disturbance; HYP, hypertrophy); a record contributes to every superclass it carries, and the number under each label is the number of records. Left, BiMa-SR; right, BiMa-SR minus comparator (diverging scale centred at 0). BiMa-SR is higher than the comparator in all {M.size} cells (smallest difference {Dm.min():+.3f}, largest {Dm.max():+.3f}).
**b**, External-cohort change: pooled median PCC on Georgia ({int(n['georgia'].n_records):,} records) or China ({int(n['china'].n_records):,} records) minus PTB-XL test ({int(n['ptbxl_test'].n_records):,} records), for BiMa-SR and the comparator. 95% CIs: percentile intervals of the difference between independent record bootstraps of the two cohorts (B = 1,000 each, same resamples as the single-cohort CIs). Both methods lose accuracy on the external cohorts. The BiMa-SR drop is larger than the comparator's (non-overlapping CIs) for {lst(larger)}, and smaller for {lst(smaller)}. BiMa-SR remains above the comparator in absolute chest-lead PCC in every configuration and external cohort{'' if ext_ok else ' (NOT in every case — see source data)'} (Figure 2a). Examples: input I, Georgia {drop('I','georgia','bima_sr')} for BiMa-SR versus {drop('I','georgia','linear_fir')} for the linear filter; I+II+V1+V4, China {drop('I,II,V1,V4','china','bima_sr')} versus {drop('I,II,V1,V4','china','einthoven_fir')}.
**c**, Efficiency (from results/final/efficiency.csv). Markers: lead-II models; vertical ticks: the other five input configurations (BiMa-SR {bs.n_params.min():,}–{bs.n_params.max():,} parameters, linear filter {lf.n_params.min():,}–{lf.n_params.max():,} coefficients = 101 taps × input leads × predicted leads). GPU latency is the time per 10-s record (batch of its two 5-s windows); throughput is windows per second at batch 128; CPU latency is per 10-s record with 2 threads for the neural models and 1 thread for the linear filter (the only CPU setting timed for it). The linear filter was not timed on GPU. Lead II: BiMa-SR {e.loc[('II','bima_sr')].gpu_ms_per_record:.1f} ms (GPU) and {e.loc[('II','bima_sr')].cpu_ms_per_record_threads2:,.0f} ms (CPU); DiBiMa-SR one-step {e.loc[('II','dibima_sr_onestep')].gpu_ms_per_record:.1f} ms and {e.loc[('II','dibima_sr_onestep')].cpu_ms_per_record_threads2:,.0f} ms; DDIM-6 ({int(e.loc[('II','dibima_sr_ddim6')].nfe)} network evaluations) {e.loc[('II','dibima_sr_ddim6')].gpu_ms_per_record:.1f} ms and {e.loc[('II','dibima_sr_ddim6')].cpu_ms_per_record_threads2:,.0f} ms; linear filter {e.loc[('II','linear_fir')].cpu_ms_per_record_threads1:.2f} ms (CPU, 1 thread). Hardware and timing protocol as in Methods.
Source data: source_data/Figure5_a_superclass.csv, Figure5_b_external_drop.csv, Figure5_c_efficiency.csv."""
    return F.write_caption(NAME, t)


if __name__ == "__main__":
    print("bbox issues:", S.bbox_check(fig))
    print("saved:", F.save(fig, NAME))
    print("caption:", caption())
    print(b_df[["panel_config", "external_cohort", "method", "pcc_drop", "drop_ci_low", "drop_ci_high"]].round(4).to_string())
