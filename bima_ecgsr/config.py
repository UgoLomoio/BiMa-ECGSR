"""Central path and study configuration. Every path can be overridden with an environment variable.

Environment variables
---------------------
BIMA_DATA_DIR     preprocessed memmaps (X_*.npy, STATS_*.npy, meta_*.csv, val_select_1024.npy, qc_report.json)
BIMA_RUNS_DIR     reconstruction runs, one sub-directory per run (<run>/best.pth, config.json, metrics_*.npz, ...)
BIMA_CLS_DIR      classifier runs (<name>/best.pt, summary.json, pred_true_<cohort>.npz)
BIMA_RAW_DIR      root of the raw cohorts (PTB-XL, Georgia, China WFDB records and the PhysioNet/CinC dx maps)
BIMA_PTBXL_DIR    raw PTB-XL v1.0.3 root      (default $BIMA_RAW_DIR/ptbxl)
BIMA_GEORGIA_DIR  raw Georgia 12-lead root    (default $BIMA_RAW_DIR/georgia)
BIMA_CHINA_DIR    raw Chapman-Shaoxing/Ningbo (default $BIMA_RAW_DIR/china)
BIMA_RESULTS_DIR  derived result tables        (default <repo>/results)
BIMA_FIGURES_DIR  rendered figures             (default <repo>/figures)
BIMA_SOURCE_DATA_DIR figure source data        (default <repo>/source_data)
BIMA_LOG_DIR      queue / smoke logs           (default <repo>/scripts/logs)
BIMA_PYTHON       interpreter used by shell scripts and by figure helpers that need torch (default: python)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _p(var: str, default) -> str:
    return str(Path(os.environ.get(var, str(default))).expanduser())


DATA_DIR = _p("BIMA_DATA_DIR", "/media/gtimpano/Data/UGO/dibima_sr12_v2_data")
RUNS_DIR = _p("BIMA_RUNS_DIR", "/media/gtimpano/Data/UGO/dibima_sr12_v2_runs")
CLS_DIR = _p("BIMA_CLS_DIR", "/media/gtimpano/Data/UGO/dibima_cls_v2_runs")
RAW_DIR = _p("BIMA_RAW_DIR", "/home/gtimpano/Scrivania/UGO/LoopedMambaECG/data")
PTBXL_DIR = _p("BIMA_PTBXL_DIR", Path(RAW_DIR) / "ptbxl")
GEORGIA_DIR = _p("BIMA_GEORGIA_DIR", Path(RAW_DIR) / "georgia")
CHINA_DIR = _p("BIMA_CHINA_DIR", Path(RAW_DIR) / "china")
DX_REFERENCE_DIR = _p("BIMA_DX_REFERENCE_DIR", Path(RAW_DIR) / "reference")
RESULTS_DIR = _p("BIMA_RESULTS_DIR", REPO / "results")
FIGURES_DIR = _p("BIMA_FIGURES_DIR", REPO / "figures")
SOURCE_DATA_DIR = _p("BIMA_SOURCE_DATA_DIR", REPO / "source_data")
LOG_DIR = _p("BIMA_LOG_DIR", REPO / "scripts" / "logs")
PYTHON = os.environ.get("BIMA_PYTHON", "python")

# ------------------------------------------------------------------ study design
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
COHORTS = ("ptbxl_test", "georgia", "china")
# input-lead configurations (display label, comma-separated CLI spec, run-name tag)
CONFIGS = (
    ("II", "II", "II"),
    ("I", "I", "I"),
    ("I+II", "I,II", "I_II"),
    ("I+II+V1", "I,II,V1", "I_II_V1"),          # 5-electrode bedside telemetry
    ("I+II+V2", "I,II,V2", "I_II_V2"),
    ("I+II+V1+V4", "I,II,V1,V4", "I_II_V1_V4"),
)
# run directories under RUNS_DIR: bima_sr_<tag> (BiMa-SR), dibima_sr_<tag> (DiBiMa-SR), baselines_<tag> (FIR / Einthoven)
DIFFUSION_CONFIGS = ("II",)


def run_dir(kind: str, tag: str) -> str:
    """kind in {'bima_sr', 'dibima_sr', 'baselines'}; tag e.g. 'I_II_V2'."""
    return os.path.join(RUNS_DIR, f"{kind}_{tag}")


def judge_ckpt() -> str:
    return os.path.join(CLS_DIR, "judge12", "best.pt")
