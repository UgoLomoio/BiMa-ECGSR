"""Figure 3 | Qualitative reconstructions (and Supplementary FigS_failures, same groups at the 10th percentile).

Selection rule (no hand-picking): within a group of windows, rank windows by a per-window score and take the window at
quantile q (nearest rank on the ascending sort; q = 0.90 for Figure 3, q = 0.10 for FigS_failures).
  a  score = BiMa-SR per-window PCC averaged over the 11 predicted leads, config II, PTB-XL test windows of NORM
     records; the selected window is then shown for all six configurations (chest leads V1, V3, V5).
  b  score = BiMa-SR per-window PCC averaged over the predicted chest leads (V2-V6), config I+II+V1, PTB-XL test
     windows of records labelled MI, CLBBB, LVH (one window per label).
  c  same score and config, all windows of Georgia and of China.
Per-lead PCCs printed on the traces are read from the per-window metrics files (metrics_<cohort>*.npz).
Run: python figures/figure3.py            (renders Figure3 and supplementary/FigS_failures)
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

S.apply_style()
FS, FS2 = S.FS_BASE, S.FS_SEC
FSAMP = 500
T = np.arange(2500) / FSAMP
CFGS = [c for _, c, _ in F.CONFIGS]
LI = {l: i for i, l in enumerate(F.LEADS)}
A_LEADS = ["V1", "V3", "V5"]
BB = dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85)
B_CFG = "I,II,V1"
B_LABELS = ["MI", "CLBBB", "LVH"]
CHEST = ["V1", "V2", "V3", "V4", "V5", "V6"]

_meta, _pcc, _X, _P = {}, {}, {}, {}


def meta(coh):
    if coh not in _meta:
        _meta[coh] = pd.read_csv(os.path.join(F.DATA, f"meta_{coh}.csv"), low_memory=False)
    return _meta[coh]


def pcc(method, cfg, coh):
    k = (method, cfg, coh)
    if k not in _pcc:
        _pcc[k] = np.load(F.metrics_path(method, cfg, coh))["pcc"].astype(np.float64)
    return _pcc[k]


def X(coh):
    if coh not in _X:
        _X[coh] = np.load(os.path.join(F.DATA, f"X_{coh}.npy"), mmap_mode="r")
    return _X[coh]


def P(method, cfg, coh):
    k = (method, cfg, coh)
    if k not in _P:
        _P[k] = np.load(F.pred_path(method, cfg, coh), mmap_mode="r")
    return _P[k]


def select(coh, cfg, leads_idx, mask, q):
    s = pcc("bima_sr", cfg, coh)[:, leads_idx].mean(1)
    cand = np.flatnonzero(mask)
    w = cand[F.pick_rank(s[cand], q)]
    return int(w), float(s[w]), int(len(cand)), int(meta(coh).loc[mask, "record_id"].nunique())


def trace_ax(ax, t, real, pred, base, measured, ylim, show_x=False):
    if measured:
        ax.plot(t, real, color=F.COL["input"], lw=0.6)
    else:
        ax.plot(t, real, color="black", lw=0.6, zorder=2)
        if base is not None:
            ax.plot(t, base, color=F.COL["linear_fir"], lw=0.6, ls=(0, (2.2, 1.2)), zorder=3)
        ax.plot(t, pred, color=F.COL["bima_sr"], lw=0.6, zorder=4, alpha=0.95)
    ax.set_xlim(0, 5); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def ylim_of(*arrs, pad=0.08):
    lo = min(float(np.min(a)) for a in arrs if a is not None); hi = max(float(np.max(a)) for a in arrs if a is not None)
    r = hi - lo
    return lo - pad * r, hi + 0.30 * r       # headroom for the PCC label


def label(ax, items):
    """items: list of (text, colour); right-aligned at the top of the trace axes."""
    x = 0.995
    for txt, col in reversed(items):
        t = ax.text(x, 0.99, txt, transform=ax.transAxes, ha="right", va="top", fontsize=FS2, color=col,
                    fontfamily="Liberation Sans Narrow")
        fig_ = ax.figure; fig_.canvas.draw() if False else None
        x -= 0.115 * len(items[0][0]) / 5 * (1 if len(items) > 1 else 0) + (0.0 if len(items) == 1 else 0.02)


def render(q, name, supplementary):
    rows = []
    W_MM, H_MM = 180, 151
    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
    A = lambda x, y, w, h: S.add_axes_mm(fig, x, y, w, h)  # noqa: E731

    # ------------------------------------------------------------- a
    coh = "ptbxl_test"; m = meta(coh)
    w, sc, nw, nr = select(coh, "II", F.pred_idx("II"), (m.NORM == 1).values, q)
    rid = m.record_id.iloc[w]; half = int(m.half.iloc[w])
    x0, y0, cw, ch, gap = 20, 14, 51.5, 8.2, 1.6
    real = X(coh)[w].astype(np.float32)
    ylims = {l: ylim_of(real[LI[l]], *[P("bima_sr", c, coh)[w][LI[l]].astype(np.float32) for c in CFGS])
             for l in A_LEADS}
    for j, l in enumerate(A_LEADS):
        fig.text((x0 + j * (cw + 2) + cw / 2) / W_MM, 1 - (y0 - 1.0) / H_MM, l,
                 ha="center", va="bottom", fontsize=FS)
    for i, cfg in enumerate(CFGS):
        ay = y0 + i * (ch + gap)
        fig.text((x0 - 2) / W_MM, 1 - (ay + ch / 2) / H_MM, F.CFG_LABEL[cfg], ha="right", va="center", fontsize=FS)
        pr = P("bima_sr", cfg, coh)[w].astype(np.float32)
        pc = pcc("bima_sr", cfg, coh)[w]
        for j, l in enumerate(A_LEADS):
            ax = A(x0 + j * (cw + 2), ay, cw, ch)
            meas = l in cfg.split(",")
            trace_ax(ax, T, real[LI[l]], pr[LI[l]], None, meas, ylims[l])
            if meas:
                ax.text(0.995, 0.99, "measured input", transform=ax.transAxes, ha="right", va="top", fontsize=FS2,
                        color=F.COL["input"])
            else:
                ax.text(0.995, 0.99, f"PCC {F.fmt3(pc[LI[l]])}", transform=ax.transAxes, ha="right", va="top",
                        fontsize=FS2, color="#B36B00", bbox=BB, zorder=6)
            rows.append(dict(panel="a", quantile=q, cohort=coh, group="NORM", selection_config="II",
                             selection_score="mean BiMa-SR PCC over 11 predicted leads", n_windows_in_group=nw,
                             n_records_in_group=nr, window_index=w, record_id=rid, half=half, selection_value=sc,
                             shown_config=cfg, lead=l, role="input" if meas else "predicted",
                             pcc_bima_sr=np.nan if meas else float(pc[LI[l]]), pcc_linear_filter=np.nan))
            if i == 0 and j == 0:
                first_ax = ax
    S.letter(fig, (2, 3.2), "a")
    fig.text(7 / W_MM, 1 - 4.0 / H_MM,
             f"PTB-XL test, NORM record {rid} (window {half + 1} of 2); traces shown for every input configuration",
             ha="left", va="top", fontsize=FS2, color="#333333")
    a_bottom = y0 + 6 * (ch + gap)
    # scale bars (a): 1 s, 1 z unit, drawn in the last V5 axes coordinates
    ax_sb = A(x0 + 2 * (cw + 2), a_bottom - gap + 0.8, cw, 3.2); ax_sb.axis("off")
    ax_sb.set_xlim(0, 5); ax_sb.set_ylim(0, 1)
    ax_sb.plot([4.0, 5.0], [0.55, 0.55], color="black", lw=0.8)
    ax_sb.text(4.5, 0.4, "1 s", ha="center", va="top", fontsize=FS2)

    # ------------------------------------------------------------- b, c
    groups = []
    m = meta("ptbxl_test")
    for lab in B_LABELS:
        groups.append(("b", "ptbxl_test", lab, (m[lab] == 1).values, lab))
    for coh in ("georgia", "china"):
        groups.append(("c", coh, "all", np.ones(len(meta(coh)), bool), F.COH_LABEL[coh]))
    by0 = a_bottom + 16
    bx0, bcw, bch, bgap, colgap = 11, 31.9, 7.6, 1.3, 1.5
    ci = F.chest_idx(B_CFG)
    for j, (pan, coh, lab, mask, title) in enumerate(groups):
        w, sc, nw, nr = select(coh, B_CFG, ci, mask, q)
        mm = meta(coh); rid = mm.record_id.iloc[w]; half = int(mm.half.iloc[w])
        real = X(coh)[w].astype(np.float32)
        pr = P("bima_sr", B_CFG, coh)[w].astype(np.float32)
        bs = P("linear_fir", B_CFG, coh)[w].astype(np.float32)
        pb = pcc("bima_sr", B_CFG, coh)[w]; pl = pcc("linear_fir", B_CFG, coh)[w]
        xx = bx0 + j * (bcw + colgap) + (2.5 if j >= 3 else 0)
        fig.text((xx + bcw / 2) / W_MM, 1 - (by0 - 4.4) / H_MM, title, ha="center", va="bottom", fontsize=FS)
        fig.text((xx + bcw / 2) / W_MM, 1 - (by0 - 1.4) / H_MM, f"record {rid} \u00b7 window score {F.fmt3(sc)}",
                 ha="center", va="bottom", fontsize=FS2, color="#333333", fontfamily="Liberation Sans Narrow")
        for i, l in enumerate(CHEST):
            ay = by0 + i * (bch + bgap)
            ax = A(xx, ay, bcw, bch)
            meas = l in B_CFG.split(",")
            trace_ax(ax, T, real[LI[l]], pr[LI[l]], bs[LI[l]], meas,
                     ylim_of(real[LI[l]], None if meas else pr[LI[l]], None if meas else bs[LI[l]]))
            if meas:
                ax.text(0.995, 0.99, "measured", transform=ax.transAxes, ha="right", va="top", fontsize=FS2,
                        color=F.COL["input"])
            else:
                ax.text(0.995, 0.99, F.fmt3(pl[LI[l]]), transform=ax.transAxes, ha="right", va="top",
                        fontsize=FS2, color="#6E6E6E", fontfamily="Liberation Sans Narrow", bbox=BB, zorder=6)
                ax.text(0.80, 0.99, F.fmt3(pb[LI[l]]), transform=ax.transAxes, ha="right", va="top",
                        fontsize=FS2, color="#B36B00", fontfamily="Liberation Sans Narrow", bbox=BB, zorder=6)
            if j == 0:
                fig.text((xx - 1.5) / W_MM, 1 - (ay + bch / 2) / H_MM, l, ha="right", va="center", fontsize=FS)
            rows.append(dict(panel=pan, quantile=q, cohort=coh, group=lab, selection_config=B_CFG,
                             selection_score="mean BiMa-SR PCC over predicted chest leads V2-V6",
                             n_windows_in_group=nw, n_records_in_group=nr, window_index=w, record_id=rid, half=half,
                             selection_value=sc, shown_config=B_CFG, lead=l, role="input" if meas else "predicted",
                             pcc_bima_sr=np.nan if meas else float(pb[LI[l]]),
                             pcc_linear_filter=np.nan if meas else float(pl[LI[l]])))
    S.letter(fig, (2, by0 - 12.3), "b")
    S.letter(fig, (bx0 + 3 * (bcw + colgap) + 0.6, by0 - 12.3), "c")
    fig.text(14 / W_MM + 0.0, 1 - (by0 - 9.5) / H_MM, "", fontsize=FS2)
    fig.text(7 / W_MM, 1 - (by0 - 11.2) / H_MM, "Input I+II+V1; pathological examples (PTB-XL test)",
             ha="left", va="top", fontsize=FS2, color="#333333")
    fig.text((bx0 + 3 * (bcw + colgap) + 5.6) / W_MM, 1 - (by0 - 11.2) / H_MM, "Input I+II+V1; external cohorts",
             ha="left", va="top", fontsize=FS2, color="#333333")
    b_bottom = by0 + 6 * (bch + bgap)
    ax_sb = A(bx0 + 4 * (bcw + colgap) + 2.5, b_bottom - bgap + 0.8, bcw, 3.2); ax_sb.axis("off")
    ax_sb.set_xlim(0, 5); ax_sb.set_ylim(0, 1)
    ax_sb.plot([4.0, 5.0], [0.55, 0.55], color="black", lw=0.8)
    ax_sb.text(4.5, 0.4, "1 s", ha="center", va="top", fontsize=FS2)

    leg = [Line2D([], [], color="black", lw=0.9, label="Real ECG"),
           Line2D([], [], color=F.COL["bima_sr"], lw=0.9, label="BiMa-SR"),
           Line2D([], [], color=F.COL["linear_fir"], lw=0.9, ls=(0, (2.2, 1.2)), label="Linear filter (b, c)"),
           Line2D([], [], color=F.COL["input"], lw=0.9, label="Measured input lead")]
    fig.legend(handles=leg[:4], loc="upper right", bbox_to_anchor=(1 - 1 / W_MM, 1 - 0.3 / H_MM), ncol=4,
               fontsize=FS2, handlelength=2.0, columnspacing=1.2, borderaxespad=0)
    fig.text(1 - 1 / W_MM, 1 - (by0 + 6 * (bch + bgap) + 3.6) / H_MM,
             "Numbers: per-lead PCC of this window, BiMa-SR (orange) and linear filter (grey)",
             ha="right", va="top", fontsize=FS2, color="#333333")
    df = pd.DataFrame(rows)
    F.src(df, name, "selected_windows")
    issues = S.bbox_check(fig)
    paths = F.save(fig, name, supplementary=supplementary)
    plt.close(fig)
    return df, issues, paths



def caption(df, q, name):
    pct = int(round(q * 100))
    sup = name != "Figure3"
    a = df[df.panel == "a"].iloc[0]
    g = df.drop_duplicates(["panel", "group", "cohort"]).set_index(["panel", "group", "cohort"])
    def grp(p, lab, coh):
        r = g.loc[(p, lab, coh)]
        return (f"{lab if p == 'b' else F.COH_LABEL[coh]}: record {r.record_id}, window {int(r.half) + 1} of 2, score "
                f"{r.selection_value:.3f} (group of {int(r.n_windows_in_group):,} windows / "
                f"{int(r.n_records_in_group):,} records)")
    b_txt = "; ".join(grp("b", l, "ptbxl_test") for l in B_LABELS)
    c_txt = "; ".join(grp("c", "all", c) for c in ("georgia", "china"))
    head = (f"**Supplementary Figure FigS_failures | Low-percentile reconstructions (10th percentile).**" if sup else
            "**Figure 3 | Qualitative reconstructions.**")
    rule = (f"Every example is selected by a fixed rule, not by eye: within a group, windows are ranked by a per-window "
            f"score (BiMa-SR PCC averaged over the scored leads of that window) and the window at the {pct}th percentile "
            f"(nearest rank) is shown.")
    neg = df[(df.pcc_bima_sr < 0) & (df.pcc_linear_filter < 0)]
    honest = ("The 90th percentile shows a good but typical case, not the best one; the same groups at the 10th "
              "percentile are shown in Supplementary FigS_failures." if not sup else
              "This figure complements Figure 3 (same groups, same rule, 90th percentile there). Low-percentile windows "
              "include real ECGs with noisy or atypical leads, where neither method follows the recording" +
              ("" if not len(neg) else " (PCC below 0 for both BiMa-SR and the linear filter: " +
               "; ".join(f"{F.COH_LABEL[r.cohort] if r.panel == 'c' else r.group} {r.lead}, "
                         f"{r.pcc_bima_sr:.3f} and {r.pcc_linear_filter:.3f}" for r in neg.itertuples()) + ")") + ".")
    t = f"""{head}
{rule} {honest} Traces are 5-s windows at 500 Hz in z units (each lead scaled separately; horizontal bar, 1 s). Black, real ECG; orange, BiMa-SR; dashed grey, linear filter; pink, measured input lead (not predicted). Numbers on the traces are the per-lead PCC of the shown window (orange, BiMa-SR; grey, linear filter), read from the per-window metrics files.
**a**, PTB-XL test, NORM records: score over the 11 predicted leads of the lead II model ({int(a.n_windows_in_group):,} windows / {int(a.n_records_in_group):,} records); selected record {a.record_id}, window {int(a.half) + 1} of 2, score {a.selection_value:.3f}. The same window is reconstructed by BiMa-SR from each of the six input configurations (rows); chest leads V1, V3 and V5 are shown.
**b**, Input I+II+V1, PTB-XL test, one window per diagnostic label (records labelled MI, CLBBB or LVH; a record may carry more than one label); score over the predicted chest leads V2–V6. {b_txt}. The comparator for I+II+V1 is Einthoven + linear filter, whose chest leads are the linear-filter chest leads shown here.
**c**, Input I+II+V1, external cohorts (all windows), same score. {c_txt}.
Source data: source_data/{name}_selected_windows.csv (selection value, record identifiers and per-lead PCCs)."""
    return F.write_caption(name, t)


if __name__ == "__main__":
    for q, name, sup in ((0.90, "Figure3", False), (0.10, "FigS_failures", True)):
        df, issues, paths = render(q, name, sup)
        print(name, "bbox issues:", issues, caption(df, q, name))
        print(df.drop_duplicates(["panel", "group"])[["panel", "cohort", "group", "record_id", "half", "selection_value",
                                                      "n_windows_in_group", "n_records_in_group"]].to_string())
