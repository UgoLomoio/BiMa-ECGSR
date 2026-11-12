"""Supplementary Fig. S3 | DiBiMa-SR (lead II): one-step versus DDIM-6 sampling.

(a) per-lead median per-window PCC, (b) per-lead mean MAE (results/final/sr_per_lead.csv), three cohorts;
(c, d) per-record paired differences one-step minus DDIM-6 (PCC, MAE; all predicted leads), computed here from
<runs>/dibima_sr_II/metrics_<cohort>_{onestep,ddim6}.npz with the exact definitions and bootstrap resamples of
bima_ecgsr/evaluation/final_stats.py (per-record value = mean over the predicted leads, averaged over the record's
windows; record bootstrap B = 1000, seeds ptbxl_test 0, georgia 1, china 2; Wilcoxon signed-rank test).
The computation is validated by reproducing the BiMa-SR minus one-step row of results/final/sr_paired.csv.
Run:  python figures/figS3.py
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
from scipy.stats import wilcoxon  # noqa: E402

NAME = "FigS3"
REPO = str(CFG.REPO)
FIN = os.path.join(CFG.RESULTS_DIR, "final")
SUP_PDF = os.path.join(REPO, "paper", "supplementary")
SUP_PNG = os.path.join(CFG.FIGURES_DIR, "supplementary")
CAP_DIR = os.path.join(CFG.FIGURES_DIR, "captions")
for d in (SUP_PDF, SUP_PNG, CAP_DIR, CFG.SOURCE_DATA_DIR):
    os.makedirs(d, exist_ok=True)
S.apply_style()
FS, FS2 = S.FS_BASE, S.FS_SEC
LEADS = list(S.LEADS)
PRED = [i for i, l in enumerate(LEADS) if l != "II"]
GROUPS = {"all_pred": PRED, "limb_pred": [i for i in PRED if i < 6], "prec_pred": [i for i in PRED if i >= 6]}
COHORTS = [("ptbxl_test", "PTB-XL"), ("georgia", "Georgia"), ("china", "China")]
SEED = {"ptbxl_test": 0, "georgia": 1, "china": 2}
C1, C6, CB = S.COL["dibima_sr_1step"], S.COL["dibima_sr_ddim6"], S.COL["bima_sr"]
RUN = CFG.RUNS_DIR

pl = pd.read_csv(os.path.join(FIN, "sr_per_lead.csv"))
srp = pd.read_csv(os.path.join(FIN, "sr_paired.csv"))


def per_record(npz, codes, R, idx):
    d = np.load(npz)
    n_w = np.bincount(codes, minlength=R).astype(np.float64)
    pcc = np.bincount(codes, np.nanmean(d["pcc"][:, idx].astype(np.float64), 1), minlength=R) / n_w
    mae = np.bincount(codes, d["mae"][:, idx].astype(np.float64).mean(1), minlength=R) / n_w
    return pcc, mae


paired, dist = [], {}
for coh, cname in COHORTS:
    meta = pd.read_csv(os.path.join(CFG.DATA_DIR, f"meta_{coh}.csv"), usecols=["record_id"])
    codes, uniq = pd.factorize(meta.record_id.astype(str), sort=False)
    R = len(uniq)
    rng = np.random.default_rng(SEED[coh])
    counts = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(1000)])
    for g, idx in GROUPS.items():
        v = {m: per_record(os.path.join(RUN, f), codes, R, idx) for m, f in
             [("onestep", f"dibima_sr_II/metrics_{coh}_onestep.npz"), ("ddim6", f"dibima_sr_II/metrics_{coh}_ddim6.npz"),
              ("bima", f"bima_sr_II/metrics_{coh}.npz")]}
        # validation against sr_paired.csv (BiMa-SR minus one-step, PCC)
        ref = srp[(srp.config == "II") & (srp.cohort == coh) & (srp.group == g) & (srp.metric == "pcc") &
                  (srp.a == "bima_sr") & (srp.b == "dibima_sr_onestep")].iloc[0]
        chk = v["bima"][0] - v["onestep"][0]
        assert abs(chk.mean() - ref.mean_diff) < 1e-9, (coh, g, chk.mean(), ref.mean_diff)
        assert abs(np.percentile((counts @ chk) / R, 2.5) - ref.ci_low) < 1e-9
        for k, (metric, better) in enumerate([("pcc", 1), ("mae", -1)]):
            diff = v["onestep"][k] - v["ddim6"][k]
            bs = (counts @ diff) / R
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            paired.append(dict(config="II", cohort=coh, group=g, metric=metric, a="dibima_sr_onestep",
                               b="dibima_sr_ddim6", n_records=R, mean_diff=float(diff.mean()),
                               median_diff=float(np.median(diff)), ci_low=float(np.percentile(bs, 2.5)),
                               ci_high=float(np.percentile(bs, 97.5)), frac_a_better=float(np.mean(better * diff > 0)),
                               wilcoxon_p=float(p)))
            if g == "all_pred":
                dist[(coh, metric)] = diff
PD = pd.DataFrame(paired)
print(PD[PD.group == "all_pred"].round(4).to_string())

# ------------------------------------------------------------------ figure
W, H = 180.0, 140.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
X0, PW, GX = 14.0, 50.0, 7.5
YA, HA, YB, HB = 9.0, 30.0, 46.0, 30.0
YC, HC = 94.0, 36.0
src_ab = []
axa, axb = [], []
xl = np.arange(len(PRED))
for j, (coh, cname) in enumerate(COHORTS):
    a = S.add_axes_mm(fig, X0 + j * (PW + GX), YA, PW, HA)
    b = S.add_axes_mm(fig, X0 + j * (PW + GX), YB, PW, HB)
    axa.append(a); axb.append(b)
    for m, col, mk, off in (("dibima_sr_onestep", C1, "o", -0.15), ("dibima_sr_ddim6", C6, "D", 0.15)):
        sub = pl[(pl.config == "II") & (pl.cohort == coh) & (pl.method == m)].set_index("lead").loc[[LEADS[i] for i in PRED]]
        a.plot(xl + off, sub.pcc_median, ls="none", marker=mk, ms=3.0, mfc=col, mec=col, zorder=3)
        b.plot(xl + off, sub.mae_mean, ls="none", marker=mk, ms=3.0, mfc=col, mec=col, zorder=3)
        for ld, r in sub.iterrows():
            src_ab.append(dict(cohort=coh, method=m, lead=ld, pcc_median=r.pcc_median, mae_mean=r.mae_mean))
    for ax in (a, b):
        ax.set_xticks(xl)
        ax.set_xlim(-0.6, len(PRED) - 0.4)
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
        ax.axvline(4.5, color="#BBBBBB", lw=0.5)
    a.set_xticklabels([])
    b.set_xticklabels([LEADS[i] for i in PRED], rotation=90)
    a.set_title(cname, fontsize=FS, loc="left", pad=2.5)
for axs in (axa, axb):
    lo = min(x.get_ylim()[0] for x in axs); hi = max(x.get_ylim()[1] for x in axs)
    for k, x in enumerate(axs):
        x.set_ylim(lo, hi)
        if k:
            x.set_yticklabels([])
axa[0].set_ylabel("Median PCC")
axb[0].set_ylabel("Mean MAE (z)")
axa[0].text(0.02, 0.04, "higher = better", transform=axa[0].transAxes, fontsize=FS2, color="#555555")
axb[0].text(0.02, 0.92, "lower = better", transform=axb[0].transAxes, fontsize=FS2, color="#555555", va="top")
fig.legend(handles=[Line2D([], [], marker="o", ls="none", mfc=C1, mec=C1, ms=3.5, label="DiBiMa-SR one-step (1 network evaluation)"),
                    Line2D([], [], marker="D", ls="none", mfc=C6, mec=C6, ms=3.2, label="DiBiMa-SR DDIM-6 (6 network evaluations)")],
           loc="upper left", bbox_to_anchor=(X0 / W, 1 - 0.3 / H), ncol=2, fontsize=FS2, borderaxespad=0,
           columnspacing=1.5, handletextpad=0.3)

# c, d: paired per-record differences (violin, clipped display range, mean + CI)
CW = (W - X0 - 4.0 - 18.0) / 2
src_cd = []
for k, (metric, ylab, cue) in enumerate([("pcc", "ΔPCC per record\n(one-step − DDIM-6)", "> 0: one-step better"),
                                         ("mae", "ΔMAE per record\n(one-step − DDIM-6)", "< 0: one-step better")]):
    ax = S.add_axes_mm(fig, X0 + 2.0 + k * (CW + 18.0), YC, CW, HC)
    data = [dist[(c, metric)] for c, _ in COHORTS]
    q = np.concatenate([np.percentile(d, [1, 99]) for d in data])
    lim = (q.min(), q.max())
    clipped = [d[(d >= lim[0]) & (d <= lim[1])] for d in data]
    vp = ax.violinplot(clipped, positions=range(3), widths=0.75, showextrema=False, showmedians=False)
    for body in vp["bodies"]:
        body.set_facecolor("#CCCCCC"); body.set_edgecolor("#888888"); body.set_alpha(1.0); body.set_linewidth(0.4)
    ax.axhline(0, color="black", ls=(0, (3, 2)), lw=0.8)
    for i, (coh, cname) in enumerate(COHORTS):
        r = PD[(PD.cohort == coh) & (PD.group == "all_pred") & (PD.metric == metric)].iloc[0]
        ax.errorbar(i, r.mean_diff, yerr=[[r.mean_diff - r.ci_low], [r.ci_high - r.mean_diff]], fmt="o", ms=3.0,
                    color=C1, mec="black", mew=0.4, elinewidth=1.0, capsize=1.5, zorder=4)
        src_cd.append(dict(cohort=coh, metric=metric, n_records=len(dist[(coh, metric)]),
                           display_low=lim[0], display_high=lim[1],
                           n_outside_display=int(((dist[(coh, metric)] < lim[0]) | (dist[(coh, metric)] > lim[1])).sum()),
                           mean_diff=r.mean_diff, ci_low=r.ci_low, ci_high=r.ci_high, median_diff=r.median_diff,
                           frac_onestep_better=r.frac_a_better, wilcoxon_p=r.wilcoxon_p))
    ax.set_xticks(range(3))
    ax.set_xticklabels([c for _, c in COHORTS])
    ax.set_ylabel(ylab)
    ax.text(0.98, 0.97 if metric == "pcc" else 0.03, cue, transform=ax.transAxes, ha="right",
            va="top" if metric == "pcc" else "bottom", fontsize=FS2, color="#555555")
    ax.set_xlim(-0.6, 2.6)
    S.letter(fig, ax, "cd"[k], dx_mm=-13.0, dy_mm=-4.0)
S.letter(fig, axa[0], "a", dx_mm=-13.0, dy_mm=-4.0)
S.letter(fig, axb[0], "b", dx_mm=-13.0, dy_mm=-1.0)
print("bbox issues:", S.bbox_check(fig))
fig.savefig(os.path.join(SUP_PNG, f"{NAME}.png"), dpi=600)
for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(SUP_PNG, f"{NAME}.{ext}"))
fig.savefig(os.path.join(SUP_PDF, f"{NAME}.pdf"))
S.src(pd.DataFrame(src_ab), NAME, "ab_per_lead")
S.src(pd.DataFrame(src_cd), NAME, "cd_paired_display")
S.src(PD, NAME, "paired_onestep_vs_ddim6_all_groups")

# ------------------------------------------------------------------ caption
nrec = {c: int(PD[(PD.cohort == c)].n_records.iloc[0]) for c, _ in COHORTS}


def ptxt(p):
    return "< 10⁻³⁰⁰ (floating-point underflow)" if p == 0 else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


def row(c, g, m):
    r = PD[(PD.cohort == c) & (PD.group == g) & (PD.metric == m)].iloc[0]
    num = f"{r.mean_diff:+.4f} ({r.ci_low:+.4f} to {r.ci_high:+.4f})".replace("-", "−")
    return f"{num}; one-step better in {100 * r.frac_a_better:.1f}% of records; Wilcoxon p {ptxt(r.wilcoxon_p)}"


ctx = " ".join(f"{cn}: ΔPCC {row(c, 'all_pred', 'pcc')}; ΔMAE {row(c, 'all_pred', 'mae')}." for c, cn in COHORTS)
chest = " ".join(f"{cn} {row(c, 'prec_pred', 'pcc').split(';')[0]}." for c, cn in COHORTS)
nout = pd.DataFrame(src_cd)
cap = f"""**Supplementary Figure S3 | DiBiMa-SR from lead II: one-step versus DDIM-6 sampling.**

DiBiMa-SR (one network, DDPM with T = 20 steps) reconstructs the 11 missing leads from lead II either in one step (one network evaluation) or with a 6-step DDIM sampler (six evaluations). Cohorts: PTB-XL test (n = {nrec['ptbxl_test']:,} records), Georgia (n = {nrec['georgia']:,}) and China (n = {nrec['china']:,}). **a**, Median per-window PCC of each predicted lead (the vertical grey line separates limb from chest leads). **b**, Mean per-window MAE (z units) of each predicted lead. Shared y axis within each row. **c**, **d**, Distribution of per-record paired differences, one-step minus DDIM-6, of PCC (**c**) and MAE (**d**) over all 11 predicted leads (per-record value = mean over the predicted leads, averaged over the record's two 5-s windows). Violins are drawn over a common display range spanning the 1st–99th percentiles of every cohort (records outside it: {int(nout[nout.metric == 'pcc'].n_outside_display.sum()):,} for PCC and {int(nout[nout.metric == 'mae'].n_outside_display.sum()):,} for MAE, all cohorts, listed per cohort in the source data); dots and bars: mean difference with 95% CI (record bootstrap, B = 1,000; the CIs are narrower than the markers). Values (mean, 95% CI): {ctx} Chest leads only (V1–V6), ΔPCC: {chest}

Source data: source_data/FigS3_ab_per_lead.csv, FigS3_cd_paired_display.csv, FigS3_paired_onestep_vs_ddim6_all_groups.csv (all, limb and chest groups).
"""
open(os.path.join(CAP_DIR, f"{NAME}.md"), "w").write(cap)
print(cap)
