"""Figure 1 | Architecture and experimental workflow (12-lead ECG reconstruction, rebuilt PTB-XL / Georgia / China data).

(a) three cohorts -> one preprocessing pipeline -> splits (counts read from the rebuilt meta_*.csv files)
(b) input-lead configurations: measured / exactly derivable (Einthoven-Goldberger) / predicted leads, models trained
(c) BiMa-SR (regression) and DiBiMa-SR (diffusion) on one bidirectional Mamba backbone; inset: T = 20 zero-SNR schedule
(d) evaluation: reconstruction accuracy vs linear FIR / exact limb leads; diagnostic readability with ONE 12-lead classifier

Facts are read from files where they exist:
  data counts     meta_{ptbxl_train,ptbxl_val,ptbxl_test,georgia,china}.csv, val_select_1024.npy, qc_report.json
  SR params       built on CPU with $BIMA_PYTHON (bima_ecgsr.training.train_sr.build_net) -> cached CSV
  classifier      judge12/summary.json (n_params)
  schedule        results/noise_schedule.csv (identical to train_sr.build_scheduler(20), max |d alphabar| ~1e-16)
  FIR             baselines_*/config.json (fir_L = 50 -> 101 taps, fir_nfit = 4000)
  training recipe bima_sr_II / dibima_sr_II config.json
Run:  python figures/figure1.py          (matplotlib env; add --recount-params to rebuild the SR models with $BIMA_PYTHON)
Panel b lists all six configurations, ordered by the number of measured leads (BiMa-SR trained for each).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.plotting import style as S  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

DATA = CFG.DATA_DIR
RUNS = CFG.RUNS_DIR
CLS_RUNS = CFG.CLS_DIR
REPO = str(CFG.REPO)
SCHED_CSV = os.path.join(CFG.RESULTS_DIR, "noise_schedule.csv")
FIG_DIR = CFG.FIGURES_DIR
SRC_DIR = CFG.SOURCE_DATA_DIR
PY_MODEL = CFG.PYTHON
NAME = "Figure1"

S.apply_style()
FS, FS2 = S.FS_BASE, S.FS_SEC                     # 7 pt (titles, box headers) / 6 pt (content)
C_IN, C_GT, C_B, C_D, C_D6, C_LIN = (S.COL[k] for k in ("input", "GT", "bima_sr", "dibima_sr_1step",
                                                        "dibima_sr_ddim6", "linear"))
C_BOX, C_EDGE, C_FAINT, C_GREY = "#F2F2F2", "#4D4D4D", "#B3B3B3", "#555555"
LEADS = S.LEADS
# Configurations shown in panel b, ordered by the number of measured leads (BiMa-SR trained for all six).
CONFIGS = [("I", ["I"]), ("II", ["II"]), ("I+II", ["I", "II"]), ("I+II+V1", ["I", "II", "V1"]),
           ("I+II+V2", ["I", "II", "V2"]), ("I+II+V1+V4", ["I", "II", "V1", "V4"])]
DERIVABLE = ["III", "aVR", "aVL", "aVF"]          # exact from I and II (Einthoven / Goldberger)
os.makedirs(SRC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def src(df, panel):
    p = os.path.join(SRC_DIR, f"{NAME}_{panel}.csv")
    df.to_csv(p, index=False)
    return p


# ============================================================================================ facts
def split_counts():
    rows = []
    spec = [("ptbxl_train", "PTB-XL", "train", "folds 1-8"), ("ptbxl_val", "PTB-XL", "validation", "fold 9"),
            ("ptbxl_test", "PTB-XL", "test", "fold 10"), ("georgia", "Georgia", "external test", "all"),
            ("china", "China", "external test", "all")]
    for key, cohort, role, folds in spec:
        m = pd.read_csv(os.path.join(DATA, f"meta_{key}.csv"), usecols=["record_id"] +
                        (["strat_fold"] if key.startswith("ptbxl") else []))
        fl = sorted(m.strat_fold.unique().tolist()) if "strat_fold" in m else None
        rows.append(dict(split=key, cohort=cohort, role=role, folds=folds,
                         strat_folds_found=",".join(map(str, fl)) if fl else "", n_windows=len(m),
                         n_records=int(m.record_id.nunique())))
    df = pd.DataFrame(rows)
    df["n_model_selection_windows"] = 0
    df.loc[df.split == "ptbxl_val", "n_model_selection_windows"] = len(np.load(os.path.join(DATA, "val_select_1024.npy")))
    return df


def cohort_records():
    qc = json.load(open(os.path.join(DATA, "qc_report.json")))
    return qc


def sr_param_counts(recount=False):
    cache = os.path.join(SRC_DIR, f"{NAME}_param_counts.csv")
    if os.path.exists(cache) and not recount:
        return pd.read_csv(cache)
    code = r"""
import sys, json
sys.path.insert(0, %r)
from bima_ecgsr.training import train_sr as T
out = []
for mode, model in (("regression", "BiMa-SR"), ("diffusion", "DiBiMa-SR")):
    for cfg, k in (("I", 1), ("II", 1), ("I+II", 2), ("I+II+V1", 3), ("I+II+V2", 3), ("I+II+V1+V4", 4)):
        net = T.build_net(mode, k)
        out.append(dict(model=model, mode=mode, input_config=cfg, k_input_leads=k,
                        encoder_input_channels=k + (12 if mode == "diffusion" else 0),
                        n_params=int(sum(p.numel() for p in net.parameters()))))
print("JSON" + json.dumps(out))
""" % REPO
    r = subprocess.run([PY_MODEL, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=900,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("JSON")]
    assert line, r.stderr[-2000:]
    df = pd.DataFrame(json.loads(line[0][4:]))
    trained = {("BiMa-SR", c) for c, _ in CONFIGS} | {("DiBiMa-SR", "II")}
    df["trained_in_study"] = [(m, c) in trained for m, c in zip(df.model, df.input_config)]
    jd = json.load(open(os.path.join(CLS_RUNS, "judge12", "summary.json")))
    df = pd.concat([df, pd.DataFrame([dict(model="12-lead classifier (1D-ResNet)", mode="classification",
                                           input_config="all 12 leads", k_input_leads=12, encoder_input_channels=12,
                                           n_params=int(jd["n_params"]), trained_in_study=True)])], ignore_index=True)
    df.to_csv(cache, index=False)
    return df


splits = split_counts()
qc = cohort_records()
PC = sr_param_counts(recount="--recount-params" in sys.argv)
sched = pd.read_csv(SCHED_CSV)
cfgB = json.load(open(os.path.join(RUNS, "bima_sr_II", "config.json")))
cfgD = json.load(open(os.path.join(RUNS, "dibima_sr_II", "config.json")))
fir = json.load(open(os.path.join(RUNS, "baselines_I_II", "config.json")))

# consistency checks against the run configs and the task specification
n_par = {(m, c): int(n) for m, c, n in zip(PC.model, PC.input_config, PC.n_params)}
assert n_par[("BiMa-SR", "II")] == cfgB["n_params"] and n_par[("DiBiMa-SR", "II")] == cfgD["n_params"]
for c_, ins_ in CONFIGS:                                   # every trained BiMa-SR run matches the recounted model
    run_ = os.path.join(RUNS, "bima_sr_" + "_".join(ins_), "config.json")
    assert json.load(open(run_))["n_params"] == n_par[("BiMa-SR", c_)], c_
for k_ in ("epochs", "lr", "lr_min", "wd", "bs", "accum", "clip", "amp", "n_val", "T"):
    assert cfgB[k_] == cfgD[k_], k_
assert (cfgD["epochs"], cfgD["lr"], cfgD["lr_min"], cfgD["wd"], cfgD["bs"], cfgD["accum"], cfgD["clip"]) == \
    (20, 1e-3, 1e-5, 1e-6, 32, 4, 1.0)
assert 2 * fir["fir_L"] + 1 == 101 and fir["fir_nfit"] == 4000
grid = sched.loc[sched.in_ddim6_grid, "t"].astype(int).tolist()[::-1]
t_one = int(sched.loc[sched.is_onestep, "t"].iloc[0])
assert grid == [19, 16, 12, 9, 6, 2] and t_one == 19 and len(sched) == 20 and sched.sqrt_alphabar.iloc[-1] == 0.0
sp = splits.set_index("split")
cnt = qc["counts"]
N_READ = {"PTB-XL": sum(cnt[s]["records_read"] for s in ("ptbxl_train", "ptbxl_val", "ptbxl_test")),
          "Georgia": cnt["georgia"]["records_read"], "China": cnt["china"]["records_read"]}
assert N_READ == {"PTB-XL": 21799, "Georgia": 10344, "China": 45152}, N_READ
for s_ in sp.index:
    assert cnt[s_]["windows_kept"] == sp.loc[s_, "n_windows"] and cnt[s_]["records_kept"] == sp.loc[s_, "n_records"]
qp = qc["parameters"]
assert (qp["fs"], qp["duration_s"], qp["band_hz"], qp["order"], qp["window"], qp["flat_std_mV"]) == \
    (500, 10.0, [0.5, 40.0], 3, 2500, 0.005), qp
n_cls = n_par[("12-lead classifier (1D-ResNet)", "all 12 leads")]
mB = n_par[("BiMa-SR", "II")] / 1e6
mD = n_par[("DiBiMa-SR", "II")] / 1e6
mB_range = (min(n_par[("BiMa-SR", c)] for c, _ in CONFIGS), max(n_par[("BiMa-SR", c)] for c, _ in CONFIGS))


def fmt(n):
    return f"{int(n):,}"


# ============================================================================================ canvas
W, H = S.FIG_W_MM, 189.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
ax0 = S.add_axes_mm(fig, 0, 0, W, H)
ax0.set_xlim(0, W); ax0.set_ylim(H, 0); ax0.axis("off"); ax0.set_zorder(-1)


def txt(x, y, s, size=FS2, ha="center", va="center", color="k", **kw):
    return ax0.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, **kw)


def box(x0, y0, w, h, fc=C_BOX, ec=C_EDGE, lw=0.6, r=0.8, ls="-", z=1, hatch=None):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                       zorder=z, hatch=hatch)
    ax0.add_patch(p)
    return p


def arrow(p0, p1, color=C_EDGE, lw=0.8, ms=6, ls="-", z=3, cs="arc3"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=z,
                        shrinkA=0, shrinkB=0, connectionstyle=cs)
    ax0.add_patch(a)
    return a


def line(xs, ys, color=C_EDGE, lw=0.8, ls="-", z=2):
    ax0.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")


def plus(x, y, r=2.0, color=C_EDGE):
    ax0.add_patch(Circle((x, y), r, fc="white", ec=color, lw=0.8, zorder=4))
    line([x - r * 0.55, x + r * 0.55], [y, y], color=color, lw=0.9, z=5)
    line([x, x], [y - r * 0.55, y + r * 0.55], color=color, lw=0.9, z=5)


def title(x, y, letter, s):
    S.letter(fig, (x, y - 0.7), letter)
    txt(x + 4.2, y, s, size=FS, ha="left", va="top")


def lines_in(x, y0, rows, dy=2.75, ha="center"):
    """rows: list of (text, size, color); first baseline centre at y0."""
    for j, (s, sz, col) in enumerate(rows):
        txt(x, y0 + j * dy, s, size=sz, color=col, ha=ha)


# ============================================================================================ (a)
YA = 1.2
title(0, YA, "a", "Three cohorts, one identical preprocessing pipeline, fixed splits")
# -- cohorts
cy0, ch, cg, cx0, cw = 8.0, 10.2, 2.4, 0.5, 31.5
coh = [("PTB-XL v1.0.3", "Germany", N_READ["PTB-XL"]), ("Georgia 12-lead", "USA", N_READ["Georgia"]),
       ("China", "Chapman-Shaoxing/Ningbo", N_READ["China"])]
coh_mid = []
for i, (nm, sub, n) in enumerate(coh):
    y = cy0 + i * (ch + cg)
    box(cx0, y, cw, ch, fc="white", ec=C_EDGE, lw=0.7)
    lines_in(cx0 + cw / 2, y + 2.4, [(nm, FS, "k"), (sub, FS2, C_GREY), (f"{fmt(n)} records", FS2, "k")], dy=2.75)
    coh_mid.append(y + ch / 2)
# -- pipeline (vertical chain)
px0, pw, ph, pg = 39.0, 58.0, 7.0, 2.6
py = [cy0 + j * (ph + pg) for j in range(4)]
xb = 35.5
for ym in coh_mid:
    line([cx0 + cw, xb], [ym, ym], lw=0.7)
line([xb, xb], [coh_mid[0], coh_mid[-1]], lw=0.7)
ymid_p = (py[0] + py[-1] + ph) / 2
arrow((xb, ymid_p), (px0, ymid_p), lw=0.8)
steps = [("12 leads, 500 Hz, first 10 s", "same code for every cohort"),
         ("Band-pass 0.5\u201340 Hz", "zero-phase Butterworth, order 3"),
         ("Two non-overlapping 5-s windows", "2,500 samples each"),
         ("z-score per window and per lead", "lead mean and s.d. kept (amplitude ratios)")]
for j, (h1, h2) in enumerate(steps):
    box(px0, py[j], pw, ph, fc=C_BOX)
    lines_in(px0 + pw / 2, py[j] + 2.25, [(h1, FS2, "k"), (h2, FS2, C_GREY)], dy=2.6)
    if j:
        arrow((px0 + pw / 2, py[j - 1] + ph), (px0 + pw / 2, py[j]), lw=0.7, ms=5)
yq = py[-1] + ph + 2.6
txt(px0 + pw / 2, yq, "excluded: windows with NaN or a flat lead (s.d. < 0.005 mV)", color=C_GREY)
# -- splits (bars = windows, linear scale)
sx0, bx0, bmax = 103.0, 128.5, 22.0
arrow((px0 + pw, ymid_p), (sx0 - 1.0, ymid_p), lw=0.8)
nmax = int(sp.n_windows.max())
SPLIT_ROWS = [("hdr", "PTB-XL, stratified folds", None),
              ("ptbxl_train", "Train: folds 1\u20138", "#4D4D4D"),
              ("ptbxl_val", "Validation: fold 9", "#A6A6A6"),
              ("ptbxl_test", "Test: fold 10", "white"),
              ("hdr", "External, test only", None),
              ("georgia", "Georgia", "white"),
              ("china", "China", "white")]
ys = 8.6
for key, lab, col in SPLIT_ROWS:
    if key == "hdr":
        txt(sx0, ys, lab, size=FS2, ha="left", color="k")
        ys += 4.3
        continue
    r_ = sp.loc[key]
    txt(sx0 + 1.5, ys, lab, ha="left")
    bw = bmax * r_.n_windows / nmax
    ax0.add_patch(Rectangle((bx0, ys - 1.35), bw, 2.7, fc=col, ec="k", lw=0.5, zorder=2,
                            hatch="/////" if key in ("georgia", "china") else None))
    s_ = f"{fmt(r_.n_windows)} ({fmt(r_.n_records)})"
    txt(bx0 + bw + 1.0, ys, s_, ha="left")
    ys += 4.3
    if key == "ptbxl_val":
        txt(sx0 + 1.5, ys - 1.0, f"{fmt(r_.n_model_selection_windows)} fixed windows for model selection",
            ha="left", color=C_GREY)
        ys += 2.6
txt(W - 0.5, yq, "bars: windows; numbers: windows (records)", ha="right", color=C_GREY)
txt(sx0, ys - 1.4, "never used for training or selection", ha="left", color=C_GREY)
src(splits.assign(records_read_qc=splits.cohort.map(N_READ))[
    ["split", "cohort", "role", "folds", "strat_folds_found", "n_windows", "n_records", "n_model_selection_windows",
     "records_read_qc"]], "a_splits")
src(pd.DataFrame([
    dict(step=1, operation="channels / rate / duration", setting="12 leads, 500 Hz, first 10 s"),
    dict(step=2, operation="band-pass filter", setting="zero-phase Butterworth, order 3, 0.5-40 Hz"),
    dict(step=3, operation="windowing", setting="two non-overlapping 5-s windows, 2,500 samples each"),
    dict(step=4, operation="normalisation", setting="z-score per window and lead; per-lead mean and s.d. stored"),
    dict(step=5, operation="exclusion", setting="window excluded if it contains NaN or a flat lead (s.d. < 0.005 mV)")]),
    "a_pipeline")

# ============================================================================================ (b)
YB = 50.2
title(0, YB, "b", "Input-lead configurations: measured, exactly derivable and predicted leads")
mx0, cwid, rh = 19.0, 5.0, 3.5                 # 6 rows x 3.5 mm = previous 5 x 4.2 mm
my0 = YB + 10.6
for c_, ld in enumerate(LEADS):
    txt(mx0 + (c_ + 0.5) * cwid, my0 - 2.0, ld)
# limb / precordial group brackets
for (a_, b_, lab) in ((0, 6, "limb"), (6, 12, "precordial")):
    xa, xb_ = mx0 + a_ * cwid + 0.4, mx0 + b_ * cwid - 0.4
    line([xa, xb_], [my0 - 5.2, my0 - 5.2], color=C_FAINT, lw=0.6)
    txt((xa + xb_) / 2, my0 - 6.9, lab, color=C_GREY)
cnt_x, bima_x, dib_x = mx0 + 12 * cwid + 7.0, mx0 + 12 * cwid + 19.0, mx0 + 12 * cwid + 32.0
txt(cnt_x, my0 - 2.0, "predicted")
txt(bima_x, my0 - 2.0, "BiMa-SR", color=C_B)
txt(dib_x, my0 - 2.0, "DiBiMa-SR", color=C_D)
brows = []
for r_i, (cname, ins) in enumerate(CONFIGS):
    y = my0 + r_i * rh
    txt(mx0 - 1.2, y + rh / 2, cname, ha="right", color=C_IN)
    n_pred = 0
    for c_, ld in enumerate(LEADS):
        x = mx0 + c_ * cwid
        if ld in ins:
            state = "measured input"
            ax0.add_patch(Rectangle((x, y), cwid, rh, fc=C_IN, ec="white", lw=0.6, zorder=2))
        else:
            n_pred += 1
            der = ld in DERIVABLE and "I" in ins and "II" in ins
            state = "predicted; exactly derivable from I and II" if der else "predicted"
            ax0.add_patch(Rectangle((x, y), cwid, rh, fc="white", ec=C_FAINT, lw=0.5, zorder=2))
            if der:
                ax0.add_patch(Rectangle((x + 0.35, y + 0.35), cwid - 0.7, rh - 0.7, fc="none", ec=C_IN, lw=0.0,
                                        hatch="//////", zorder=3))
                ax0.add_patch(Rectangle((x + 0.35, y + 0.35), cwid - 0.7, rh - 0.7, fc="none", ec=C_IN, lw=0.7,
                                        zorder=3))
        brows.append(dict(input_config=cname, lead=ld, state=state))
    txt(cnt_x, y + rh / 2, str(n_pred))
    ax0.add_patch(Circle((bima_x, y + rh / 2), 1.15, fc=C_B, ec="k", lw=0.4, zorder=3))
    if cname == "II":
        ax0.add_patch(Circle((dib_x, y + rh / 2), 1.15, fc=C_D, ec="k", lw=0.4, zorder=3))
    else:
        txt(dib_x, y + rh / 2, "\u2013", color=C_FAINT)
    brows_n = dict(input_config=cname, n_input=len(ins), n_predicted=n_pred, bima_sr_trained=True,
                   dibima_sr_trained=cname == "II")
    brows.append({**brows_n, "lead": "", "state": "summary"})
assert [r["n_predicted"] for r in brows if r["state"] == "summary"] == [11, 11, 10, 9, 9, 8]
# key
kx, ky = 131.0, YB + 8.0
key_items = [("measured input lead", "in"), ("predicted by the model", "pred"),
             ("predicted; exactly derivable from I and II", "der")]
for j, (lab, kind) in enumerate(key_items):
    y = ky + j * 4.4
    if kind == "in":
        ax0.add_patch(Rectangle((kx, y - 1.5), 4.0, 3.0, fc=C_IN, ec="white", lw=0.6, zorder=2))
    else:
        ax0.add_patch(Rectangle((kx, y - 1.5), 4.0, 3.0, fc="white", ec=C_FAINT, lw=0.5, zorder=2))
        if kind == "der":
            ax0.add_patch(Rectangle((kx + 0.35, y - 1.15), 3.3, 2.3, fc="none", ec=C_IN, lw=0.7, hatch="//////",
                                    zorder=3))
    txt(kx + 5.3, y, lab, ha="left")
yk = ky + 3 * 4.4 + 0.3
txt(kx, yk, "Einthoven / Goldberger:", ha="left", color=C_GREY)
txt(kx, yk + 2.8, "III = II \u2212 I,  aVR = \u2212(I + II)/2", ha="left", color=C_GREY)
txt(kx, yk + 5.6, "aVL = I \u2212 II/2,  aVF = II \u2212 I/2", ha="left", color=C_GREY)
src(pd.DataFrame(brows), "b_input_configs")

# ============================================================================================ (c)
YC = 84.6
title(0, YC, "c", "BiMa-SR and DiBiMa-SR share one bidirectional Mamba backbone")
o = YC + 4.6
ymain = o + 24.0
bp = cfgD.get("best_params", {})
dim, dstate, nbl, nla, ver = bp.get("dim", 64), bp.get("d_state", 16), bp.get("n_mamba_blocks", 4), \
    bp.get("n_mamba_layers", 8), bp.get("version", 1)
assert (dim, dstate, nbl, nla) == (64, 16, 4, 8), bp
# input normalisation (both models)
box(0.5, o + 0.5, 33.0, 10.0, fc="white", ec=C_IN, lw=0.9)
lines_in(17.0, o + 2.6, [("k measured input leads", FS2, C_IN), ("\u00f7 common scale s, s\u00b2 = mean \u03c3\u00b2", FS2, "k"),
                         ("of input leads (k = 1: z-score)", FS2, C_GREY)], dy=2.8)
# bus to both model boxes
xbus = 2.6
line([xbus, xbus], [o + 10.5, o + 33.5], color=C_IN, lw=0.8)
arrow((xbus, o + 19.0), (5.0, o + 19.0), color=C_IN, lw=0.8, ms=5)
arrow((xbus, o + 33.5), (5.0, o + 33.5), color=C_IN, lw=0.8, ms=5)
box(5.0, o + 13.5, 28.5, 11.0, fc="white", ec=C_B, lw=1.0)
lines_in(19.25, o + 16.0, [("BiMa-SR (regression)", FS2, C_B), ("encoder input: k leads", FS2, "k"),
                           (f"{mB:.2f} M parameters", FS2, C_GREY)], dy=2.9)
box(5.0, o + 27.0, 28.5, 13.0, fc="white", ec=C_D, lw=1.0)
lines_in(19.25, o + 29.5, [("DiBiMa-SR (diffusion)", FS2, C_D), (r"encoder input: [$x_t$ (12 ch),", FS2, "k"),
                           ("k leads]", FS2, "k"), (f"{mD:.2f} M parameters (k = 1)", FS2, C_GREY)], dy=2.75)
# shared backbone region
box(36.5, o + 7.6, 110.0, 34.0, fc="#FAFAFA", ec=C_FAINT, lw=0.6, ls=(0, (3, 2)), r=1.5, z=0)
txt(145.5, o + 39.4, "shared architecture; each model trained separately", ha="right", color=C_GREY)
e0, ew = 40.0, 19.0
box(e0, ymain - 6, ew, 12)
lines_in(e0 + ew / 2, ymain - 2.3, [("Conv encoder", FS2, "k"), (f"kernel 5, {dim} ch", FS2, C_GREY)], dy=4.6)
arrow((33.5, o + 19.0), (e0, ymain - 1.5), color=C_B, lw=0.8)
arrow((33.5, o + 33.5), (e0, ymain + 1.5), color=C_D, lw=0.8)
pt = (64.5, ymain)
arrow((e0 + ew, ymain), (pt[0] - 2.0, ymain))
plus(*pt)
box(47.5, o + 32.4, 34.0, 7.6, fc="white", ec=C_D, lw=0.8)
lines_in(64.5, o + 34.6, [(r"DiBiMa-SR: timestep $t$", FS2, C_D), ("sinusoidal embedding + MLP", FS2, C_D)],
         dy=3.0)
arrow((pt[0], o + 32.4), (pt[0], ymain + 2.0), color=C_D, lw=0.8)
m0, mw_ = 70.0, 36.0
arrow((pt[0] + 2.0, ymain), (m0, ymain))
box(m0, ymain - 8.0, mw_, 16.0, fc="#E8E8E8")
lines_in(m0 + mw_ / 2, ymain - 5.2, [("Bidirectional Mamba (v1)", FS2, "k"),
                                     (f"{nbl} blocks \u00d7 {nla} layers", FS2, "k"),
                                     (f"d_model {dim}, d_state {dstate}", FS2, C_GREY),
                                     ("forward + reversed scan", FS2, C_GREY)], dy=3.45)
d0, dw = 112.0, 19.0
arrow((m0 + mw_, ymain), (d0, ymain))
box(d0, ymain - 6, dw, 12)
lines_in(d0 + dw / 2, ymain - 2.3, [("Conv decoder", FS2, "k"), ("12 ch", FS2, C_GREY)], dy=4.6)
line([e0 + ew / 2, e0 + ew / 2], [ymain - 6, o + 11.0], color=C_FAINT, lw=0.7)
line([e0 + ew / 2, d0 + dw / 2], [o + 11.0, o + 11.0], color=C_FAINT, lw=0.7)
arrow((d0 + dw / 2, o + 11.0), (d0 + dw / 2, ymain - 6), color=C_FAINT, lw=0.7)
txt(88.0, o + 9.5, "encoder\u2013decoder skip", color=C_GREY)
pr = (139.5, ymain)
arrow((d0 + dw, ymain), (pr[0] - 2.0, ymain))
plus(*pr, color=C_IN)
yres = o + 4.6
line([33.5, pr[0]], [yres, yres], color=C_IN, lw=0.9)
arrow((pr[0], yres), (pr[0], ymain - 2.0), color=C_IN, lw=0.9)
txt(88.0, yres - 1.5, "lead residual: each input lead added to its own output channel", color=C_IN)
q0, qw = 149.0, 30.0
arrow((pr[0] + 2.0, ymain), (q0, ymain))
box(q0, ymain - 6.0, qw, 12.0, fc="white", ec=C_GT, lw=0.8)
lines_in(q0 + qw / 2, ymain - 3.0, [("12 leads", FS, "k"), ("measured leads re-inserted", FS2, C_GREY),
                                    ("at evaluation", FS2, C_GREY)], dy=2.9)
# diffusion loop (DiBiMa-SR inference)
yl = o + 45.0
xl = 151.5
line([xl, xl], [ymain + 6.0, yl], color=C_D6, lw=0.9)
line([xl, 19.25], [yl, yl], color=C_D6, lw=0.9)
arrow((19.25, yl), (19.25, o + 40.0), color=C_D6, lw=0.9)
txt(84.0, yl - 1.6, r"DiBiMa-SR inference from $x_{19}\sim\mathcal{N}(0,I)$: one-step = a single $\hat{x}_0$ at "
    r"t = 19;  DDIM-6 ($\eta$ = 0) = $\hat{x}_0$ re-noised to $x_t$ at t = 19, 16, 12, 9, 6, 2", color=C_D)
txt(153.5, ymain + 9.5, "BiMa-SR: one", color=C_B, ha="left")
txt(153.5, ymain + 12.3, "forward pass", color=C_B, ha="left")
# noise-schedule inset (top right)
axs = S.add_axes_mm(fig, 157.5, o + 1.2, 20.5, 10.6)
t = sched.t.values
axs.plot(t, sched.sqrt_alphabar, color=C_GT, lw=0.9, zorder=3)
axs.plot(t, sched.sqrt_one_minus_alphabar, color=C_LIN, lw=0.9, ls=(0, (2.5, 1.2)), zorder=3)
g = sched[sched.in_ddim6_grid]
axs.plot(g.t, g.sqrt_alphabar, ls="none", marker="o", ms=2.2, mfc=C_D6, mec=C_D, mew=0.4, zorder=4)
axs.set_xlim(-0.8, 19.8); axs.set_ylim(-0.08, 1.12)
axs.set_xticks([0, 19]); axs.set_yticks([0, 1])
axs.tick_params(labelsize=FS2, pad=1, length=1.8)
axs.set_xlabel("t", fontsize=FS2, labelpad=-4)
axs.text(0.4, 0.62, "signal", fontsize=FS2, color=C_GT, ha="left", va="top")
axs.text(19.3, 0.62, "noise", fontsize=FS2, color=C_GREY, ha="right", va="top")
txt(178.0, o - 1.6, "T = 20, zero terminal SNR", ha="right")
# training strip
yt = o + 49.0
box(0.5, yt, 179.0, 7.2, fc="white", ec=C_FAINT, lw=0.6)
txt(2.0, yt + 2.1, "Training (every model): 20 epochs; AdamW, lr $10^{-3}$ \u2192 $10^{-5}$ (cosine), weight decay "
    "$10^{-6}$; effective batch 32 (4 \u00d7 8); bf16; gradient clip 1; MSE to the 12-lead target", ha="left")
txt(2.0, yt + 5.0, "Model selection: pooled median PCC over predicted leads on 1,024 fixed validation windows. "
    "DiBiMa-SR: DDPM, T = 20, cosine schedule, x\u2080-prediction", ha="left")
src(sched[["t", "alphabar", "sqrt_alphabar", "sqrt_one_minus_alphabar", "snr", "in_ddim6_grid", "is_onestep"]],
    "c_noise_schedule")
arch = PC.copy()
arch_common = dict(encoder="Conv1d, kernel 5, 64 ch", backbone=f"bidirectional Mamba v{ver}, {nbl} blocks x {nla} "
                   f"layers, d_model {dim}, d_state {dstate}, forward + reversed scan",
                   decoder="Conv decoder, 12 ch, encoder-decoder skip", lead_residual="each input lead added to its "
                   "own output channel", input_norm="common scale s = sqrt(mean over input leads of std^2)",
                   epochs=cfgD["epochs"], optimizer="AdamW", lr=cfgD["lr"], lr_min=cfgD["lr_min"], lr_schedule="cosine",
                   weight_decay=cfgD["wd"], effective_batch=cfgD["bs"], accum=cfgD["accum"], precision="bf16",
                   grad_clip=cfgD["clip"], loss="MSE to 12-lead target",
                   selection="pooled median PCC over predicted leads, 1,024 fixed validation windows",
                   diffusion="DDPM T=20, cosine + zero terminal SNR, x0-prediction; one-step t=19; DDIM-6 eta=0 "
                   "t=19,16,12,9,6,2")
src(arch.assign(**{k: v for k, v in arch_common.items()}), "c_architecture")

# ============================================================================================ (d)
YD = 147.6
title(0, YD, "d", "Evaluation: reconstruction accuracy and diagnostic readability")
r1y, r1h = YD + 5.4, 12.6
r2y, r2h = r1y + r1h + 3.0, 13.6
# reconstructions source box
box(0.5, r1y, 28.0, r2y + r2h - r1y, fc="white", ec=C_EDGE, lw=0.7)
lines_in(14.5, r1y + 3.0, [("Reconstructed", FS, "k"), ("12-lead ECGs", FS, "k")], dy=3.1)
txt(9.2, r1y + 10.6, "BiMa-SR", color=C_B, ha="left")
txt(9.2, r1y + 13.4, "DiBiMa-SR", color=C_D, ha="left")
ax0.add_patch(Circle((6.8, r1y + 10.6), 1.0, fc=C_B, ec="k", lw=0.4, zorder=3))
ax0.add_patch(Circle((6.8, r1y + 13.4), 1.0, fc=C_D, ec="k", lw=0.4, zorder=3))
lines_in(14.5, r1y + 18.4, [("PTB-XL test", FS2, "k"), ("Georgia", FS2, "k"), ("China", FS2, "k")], dy=2.7)
# (1) accuracy
a1x, a1w = 33.0, 66.0
box(a1x, r1y, a1w, r1h, fc=C_BOX)
lines_in(a1x + 2.0, r1y + 2.5, [("(1) Reconstruction accuracy", FS, "k"),
                                ("pooled median PCC and MAE over predicted leads", FS2, "k"),
                                ("limb and precordial leads separately;", FS2, C_GREY),
                                ("per lead and per diagnostic class", FS2, C_GREY)], dy=2.65, ha="left")
arrow((28.5, r1y + r1h / 2), (a1x, r1y + r1h / 2))
c1x = a1x + a1w + 3.5
box(c1x, r1y, W - 0.5 - c1x, r1h, fc="white", ec=C_LIN, lw=0.9)
lines_in(c1x + 2.0, r1y + 2.5, [("Comparators", FS, C_GREY),
                                (f"per-lead linear FIR filter, {2 * fir['fir_L'] + 1} taps per input lead,", FS2, "k"),
                                (f"float64 least squares on {fmt(fir['fir_nfit'])} PTB-XL training windows", FS2, C_GREY),
                                ("exact Einthoven/Goldberger limb leads (I and II inputs)", FS2, C_GREY)], dy=2.65, ha="left")
line([a1x + a1w, c1x], [r1y + r1h / 2, r1y + r1h / 2], color=C_LIN, lw=0.8, ls=(0, (2, 1.5)))
# (2) readability
real_x, real_w = 33.0, 22.0
box(real_x, r2y + 1.0, real_w, 5.0, fc="white", ec=C_GT, lw=0.8)
txt(real_x + real_w / 2, r2y + 3.5, "real 12-lead ECG")
box(real_x, r2y + r2h - 6.0, real_w, 5.0, fc="white", ec=C_D, lw=0.8)
txt(real_x + real_w / 2, r2y + r2h - 3.5, "reconstruction", color=C_D)
arrow((28.5, r2y + r2h - 3.5), (real_x, r2y + r2h - 3.5))
k2x, k2w = 59.5, 57.0
box(k2x, r2y, k2w, r2h, fc=C_BOX)
arrow((real_x + real_w, r2y + 3.5), (k2x, r2y + 5.0), color=C_GT, lw=0.8)
arrow((real_x + real_w, r2y + r2h - 3.5), (k2x, r2y + r2h - 5.0), color=C_D, lw=0.8)
lines_in(k2x + 2.0, r2y + 2.5, [("(2) One 12-lead classifier, run unchanged", FS, "k"),
                                (f"1D-ResNet, {n_cls / 1e6:.2f} M parameters", FS2, "k"),
                                ("trained only on real PTB-XL 12-lead ECGs", FS2, C_GREY),
                                ("(folds 1\u20138; early stopping on fold 9)", FS2, C_GREY)], dy=2.8, ha="left")
au_x = k2x + k2w + 3.5
arrow((k2x + k2w, r2y + r2h / 2), (au_x, r2y + r2h / 2))
box(au_x, r2y, W - 0.5 - au_x, r2h, fc="white", ec=C_EDGE, lw=0.7)
lines_in(au_x + 2.0, r2y + 2.5, [("Record-level AUROC (mean of 2 windows)", FS, "k"),
                                 ("PTB-XL superclasses: NORM, MI, STTC, CD, HYP", FS2, "k"),
                                 ("12-label panel, all cohorts: AFIB, AFLT, 1AVB, PAC,", FS2, C_GREY),
                                 ("PVC, SBRAD, STACH, WPW, LVH, RVH, CRBBB, CLBBB", FS2, C_GREY)], dy=2.8, ha="left")
txt(W - 1.5, r2y + r2h + 2.2, "China: AFLT not scored", ha="right", color=C_GREY)
txt(0.5, r2y + r2h + 2.2, "Statistics: 95% record-bootstrap CIs (B = 1,000); paired comparisons on shared "
    "bootstrap resamples", ha="left")
src(pd.DataFrame([
    dict(component="cohorts evaluated", setting="PTB-XL test (fold 10), Georgia, China"),
    dict(component="accuracy metrics", setting="pooled median PCC and MAE over predicted leads; limb and precordial "
         "separately; per lead; per diagnostic class"),
    dict(component="comparator: linear FIR", setting=f"per-lead FIR from the input leads, {2 * fir['fir_L'] + 1} taps "
         f"per input lead, float64 least squares on {fir['fir_nfit']} PTB-XL training windows"),
    dict(component="comparator: exact limb leads", setting="Einthoven/Goldberger when I and II are inputs"),
    dict(component="classifier", setting=f"one 12-lead 1D-ResNet, {n_cls} parameters, trained on real PTB-XL 12-lead "
         "ECGs (folds 1-8, early stopping on fold 9); run unchanged on real and reconstructed ECGs"),
    dict(component="AUROC unit", setting="record (record probability = mean of its two windows)"),
    dict(component="labels: superclasses (PTB-XL)", setting="NORM, MI, STTC, CD, HYP"),
    dict(component="labels: shared panel", setting="AFIB, AFLT, 1AVB, PAC, PVC, SBRAD, STACH, WPW, LVH, RVH, CRBBB, "
         "CLBBB; China AFLT not scored"),
    dict(component="statistics", setting="95% record-bootstrap CIs (B = 1,000); paired comparisons on shared "
         "resamples")]), "d_evaluation")

# ============================================================================================ save
if __name__ == "__main__":
    issues = S.bbox_check(fig)
    print("bbox issues:", issues)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIG_DIR, f"{NAME}.{ext}"), dpi=600 if ext == "png" else None)
    print("params:", {f"{m}|{c}": int(n) for m, c, n in zip(PC.model, PC.input_config, PC.n_params)})
    print("splits:", splits[["split", "n_windows", "n_records"]].values.tolist())
