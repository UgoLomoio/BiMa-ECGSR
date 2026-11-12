"""Figure style for BiMa-ECGSR (single module; no text-rewriting hooks).

House style: bold text, 600 dpi PNG + PDF/SVG, 180 mm width (double column), bold lower-case panel letters,
Okabe-Ito colours: BiMa-SR #E69F00, DiBiMa-SR (one-step) #0072B2, DiBiMa-SR (DDIM-6) #56B4E9,
linear filter #999999, measured input lead #CC79A7, ground truth black.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import pandas as pd

from bima_ecgsr import config as CFG

MM = 1 / 25.4
FIG_W_MM = 180.0
FS_BASE, FS_SEC, FS_TICK, FS_LETTER = 7.0, 6.0, 6.0, 9.0

LEADS = list(CFG.LEADS)
FS_HZ, N_SAMPLES = 500, 2500
LIMB = [0, 1, 2, 3, 4, 5]
PRECORDIAL = list(range(6, 12))
# PTB-XL diagnostic superclasses: label-index order (alphabetical) and conventional display order
LABEL_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# method key -> (display name, colour)
METHODS = {
    "bima_sr": ("BiMa-SR", "#E69F00"),
    "dibima_sr_1step": ("DiBiMa-SR (one-step)", "#0072B2"),
    "dibima_sr_ddim6": ("DiBiMa-SR (DDIM-6)", "#56B4E9"),
    "linear": ("Linear filter", "#999999"),
}
NAME = {k: v[0] for k, v in METHODS.items()}
COL = {k: v[1] for k, v in METHODS.items()}
COL.update({"GT": "#000000", "input": "#CC79A7"})
# aliases used by result tables
COL["dibima_sr_onestep"] = COL["dibima_sr_1step"]
COL["linear_fir"] = COL["linear"]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Nimbus Sans", "DejaVu Sans"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.titleweight": "bold",
    "font.size": FS_BASE,
    "axes.titlesize": FS_BASE,
    "axes.labelsize": FS_BASE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_SEC,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "axes.titlelocation": "left",
    "axes.titlepad": 3.0,
    "hatch.linewidth": 0.5,
    "mathtext.default": "bf",
    "mathtext.fontset": "custom",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.rm": "Liberation Sans:bold",
    "mathtext.it": "Liberation Sans:bold:italic",
}


def apply_style():
    mpl.rcParams.update(RC)


def letter(fig, ax_or_xy, ch, dx_mm=0.0, dy_mm=0.0):
    """Bold lower-case panel letter. ax_or_xy: axes (top-left of its position) or (x, y) in mm from top-left."""
    W, H = fig.get_size_inches() / MM
    if isinstance(ax_or_xy, tuple):
        x_mm, y_mm = ax_or_xy
    else:
        bb = ax_or_xy.get_position()
        x_mm, y_mm = bb.x0 * W, (1 - bb.y1) * H
    fig.text((x_mm + dx_mm) / W, 1 - (y_mm + dy_mm) / H, ch, fontsize=FS_LETTER,
             fontweight="bold", ha="left", va="top", gid=f"panel_letter_{ch}")


def add_axes_mm(fig, x, y, w, h, **kw):
    """Axes from mm coordinates measured from the TOP-LEFT corner of the figure."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def bbox_check(fig, ignore_gids=("allow_overlap",)):
    """Text/text overlaps and text outside the figure (returns a list of issues)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t.get_gid() not in ignore_gids
             and not (t.get_gid() or "").startswith("allow")]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    fb = fig.bbox
    issues = []
    for t, b in boxes:
        if b.x0 < fb.x0 - 1 or b.y0 < fb.y0 - 1 or b.x1 > fb.x1 + 1 or b.y1 > fb.y1 + 1:
            issues.append(("outside", t.get_text()[:40]))
    for i, (a, ba) in enumerate(boxes):
        for b_, bb in boxes[i + 1:]:
            if ba.overlaps(bb):
                inter = (min(ba.x1, bb.x1) - max(ba.x0, bb.x0)) * (min(ba.y1, bb.y1) - max(ba.y0, bb.y0))
                if inter > 2.0:
                    issues.append(("overlap", a.get_text()[:30], b_.get_text()[:30]))
    return issues


def save(fig, name, fig_dir=None):
    fig_dir = fig_dir or CFG.FIGURES_DIR
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg"):
        p = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=600 if ext == "png" else None)
        paths.append(p)
    return paths


def src(df: pd.DataFrame, fig_name: str, panel: str, src_dir=None):
    src_dir = src_dir or CFG.SOURCE_DATA_DIR
    os.makedirs(src_dir, exist_ok=True)
    p = os.path.join(src_dir, f"{fig_name}_{panel}.csv")
    df.to_csv(p, index=False)
    return p
