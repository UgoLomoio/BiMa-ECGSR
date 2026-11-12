"""Shared helpers for Figures 2, 3, 5 and FigS_failures (BiMa-ECGSR).

Reads only results/final/*.csv and the per-window metrics / reconstructions under CFG.RUNS_DIR and CFG.DATA_DIR
(read-only).  Writes figures to figures/ (PNG/PDF/SVG) and paper/figures/ (PDF), supplementary figures to
figures/supplementary/ (PNG/PDF/SVG) and paper/supplementary/ (PDF), and source data to source_data/.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.plotting import style as S  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FINAL = os.path.join(REPO, "results", "final")
RUNS, DATA = CFG.RUNS_DIR, CFG.DATA_DIR
FIG_DIR = os.path.join(REPO, "figures")
SUPP_FIG_DIR = os.path.join(FIG_DIR, "supplementary")
PAPER_FIG_DIR = os.path.join(REPO, "paper", "figures")
PAPER_SUPP_DIR = os.path.join(REPO, "paper", "supplementary")
SRC_DIR = os.path.join(REPO, "source_data")
CAP_DIR = os.path.join(FIG_DIR, "captions")
LEADS = list(CFG.LEADS)

# display label, config string in the CSVs, run tag -- ALWAYS in this order (number of measured leads)
CONFIGS = [("I", "I", "I"), ("II", "II", "II"), ("I+II", "I,II", "I_II"), ("I+II+V1", "I,II,V1", "I_II_V1"),
           ("I+II+V2", "I,II,V2", "I_II_V2"), ("I+II+V1+V4", "I,II,V1,V4", "I_II_V1_V4")]
CFG_LABEL = {c: l for l, c, _ in CONFIGS}
CFG_TAG = {c: t for _, c, t in CONFIGS}
COHORTS = [("ptbxl_test", "PTB-XL test"), ("georgia", "Georgia"), ("china", "China")]
COH_LABEL = dict(COHORTS)
SEED = {"ptbxl_test": 0, "georgia": 1, "china": 2}       # identical to bima_ecgsr/evaluation/final_stats.py

NAMES = {"bima_sr": "BiMa-SR", "dibima_sr_onestep": "DiBiMa-SR (one-step)", "dibima_sr_ddim6": "DiBiMa-SR (DDIM-6)",
         "linear_fir": "Linear filter", "einthoven_fir": "Einthoven + linear filter"}
COL = {"bima_sr": S.COL["bima_sr"], "dibima_sr_onestep": S.COL["dibima_sr_1step"],
       "dibima_sr_ddim6": S.COL["dibima_sr_ddim6"], "linear_fir": S.COL["linear"], "einthoven_fir": S.COL["linear"],
       "GT": "#000000", "input": S.COL["input"]}
MARK = {"bima_sr": "o", "dibima_sr_onestep": "^", "dibima_sr_ddim6": "v", "linear_fir": "s", "einthoven_fir": "D"}


def baseline(cfg: str) -> str:
    """Comparator: linear filter for single-lead configs, Einthoven + linear filter when I and II are inputs."""
    return "einthoven_fir" if {"I", "II"} <= set(cfg.split(",")) else "linear_fir"


def pred_idx(cfg: str):
    inp = cfg.split(",")
    return [i for i, l in enumerate(LEADS) if l not in inp]


def chest_idx(cfg: str):
    return [i for i in pred_idx(cfg) if i >= 6]


def load_csv(name):
    return pd.read_csv(os.path.join(FINAL, f"{name}.csv"))


def metrics_path(method: str, cfg: str, cohort: str) -> str:
    tag = CFG_TAG[cfg]
    return {"bima_sr": f"bima_sr_{tag}/metrics_{cohort}.npz",
            "dibima_sr_onestep": f"dibima_sr_{tag}/metrics_{cohort}_onestep.npz",
            "dibima_sr_ddim6": f"dibima_sr_{tag}/metrics_{cohort}_ddim6.npz",
            "linear_fir": f"baselines_{tag}/metrics_{cohort}_fir.npz",
            "einthoven_fir": f"baselines_{tag}/metrics_{cohort}_einthoven_fir.npz"}[method].join([RUNS + "/", ""])


def pred_path(method: str, cfg: str, cohort: str) -> str:
    tag = CFG_TAG[cfg]
    return os.path.join(RUNS, {"bima_sr": f"bima_sr_{tag}/pred_{cohort}.npy",
                               "dibima_sr_onestep": f"dibima_sr_{tag}/pred_{cohort}_onestep.npy",
                               "dibima_sr_ddim6": f"dibima_sr_{tag}/pred_{cohort}_ddim6.npy",
                               "linear_fir": f"baselines_{tag}/pred_{cohort}_fir.npy",
                               "einthoven_fir": f"baselines_{tag}/pred_{cohort}_einthoven_fir.npy"}[method])


def boot_counts(cohort: str, B: int = 1000):
    """Record codes and [B, R] bootstrap multiplicities, generated exactly as in final_stats.py."""
    meta = pd.read_csv(os.path.join(DATA, f"meta_{cohort}.csv"), usecols=["record_id"])
    codes, uniq = pd.factorize(meta.record_id.astype(str), sort=False)
    R = len(uniq)
    rng = np.random.default_rng(SEED[cohort])
    counts = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(B)])
    return codes, counts


def wmedian_boot_dist(v, rec_of, counts):
    """Point estimate and bootstrap distribution of the pooled median (same algorithm as final_stats.wmedian_boot)."""
    o = np.argsort(v, kind="stable"); vs = v[o]; ro = rec_of[o]
    res = np.empty(len(counts))
    for b, c in enumerate(counts):
        cw = np.cumsum(c[ro].astype(np.float64)); res[b] = vs[np.searchsorted(cw, 0.5 * cw[-1])]
    return float(np.median(v)), res


def pick_rank(values: np.ndarray, q: float) -> int:
    """Index (into values) of the element at quantile q by nearest rank on the ascending sort (ties: stable)."""
    o = np.argsort(values, kind="stable")
    return int(o[int(round(q * (len(values) - 1)))])


def src(df: pd.DataFrame, fig: str, panel: str) -> str:
    os.makedirs(SRC_DIR, exist_ok=True)
    p = os.path.join(SRC_DIR, f"{fig}_{panel}.csv")
    df.to_csv(p, index=False)
    return p


def save(fig, name: str, supplementary: bool = False):
    fdir = SUPP_FIG_DIR if supplementary else FIG_DIR
    pdir = PAPER_SUPP_DIR if supplementary else PAPER_FIG_DIR
    os.makedirs(fdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
    out = []
    for ext in ("png", "pdf", "svg"):
        p = os.path.join(fdir, f"{name}.{ext}")
        fig.savefig(p, dpi=600 if ext == "png" else None)
        out.append(p)
    p = os.path.join(pdir, f"{name}.pdf"); fig.savefig(p); out.append(p)
    return out


def write_caption(name: str, text: str):
    os.makedirs(CAP_DIR, exist_ok=True)
    p = os.path.join(CAP_DIR, f"{name}.md")
    with open(p, "w") as f:
        f.write(text.strip() + "\n")
    return p


def fmt3(x, signed=False):
    """'.913' / '+.281' / '−.003' (leading zero dropped, true minus sign)."""
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    s = s.replace("0.", ".", 1) if abs(x) < 1 else s
    return s.replace("-", "\u2212")
