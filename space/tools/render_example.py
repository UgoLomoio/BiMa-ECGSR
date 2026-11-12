"""Render the app output for one bundled example (plot + per-lead table + classifier table) to a PNG, without launching
the web server.  BIMA_WEIGHTS_DIR=<staging> python space/tools/render_example.py [--example 5] [--out app_render.png]"""
import argparse, os, sys
SPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPACE)
ap = argparse.ArgumentParser(); ap.add_argument("--example", type=int, default=5); ap.add_argument("--config", default="II")
ap.add_argument("--method", default="BiMa-SR"); ap.add_argument("--out", default=os.path.join(SPACE, "tools", "app_render.png"))
a = ap.parse_args()
import matplotlib.pyplot as plt  # noqa: E402
import app as A  # noqa: E402

fig, df, jt, info = A.run("Bundled example", A.example_choices()[a.example], None, "mV", a.config, a.method, True, True, "mV")
W, H = fig.get_size_inches(); extra = 4.2
fig.set_size_inches(W, H + extra)
for ax in fig.axes:   # shift the ECG panels up to make room for the tables
    b = ax.get_position(); ax.set_position([b.x0, extra / (H + extra) + b.y0 * H / (H + extra), b.width, b.height * H / (H + extra)])
for t in fig.legends + fig.texts:
    pass
import textwrap  # noqa: E402
txt = info.replace("**", "")
fig.text(0.02, (extra - 0.1) / (H + extra), "Research demonstration — not for clinical use", fontsize=9, va="top", color="#B00000", fontweight="bold")
fig.text(0.02, (extra - 0.4) / (H + extra), "\n".join(textwrap.wrap(txt, 150)), fontsize=7.5, va="top", family="monospace")
d = df.copy(); d = d[d.role == "predicted"][["lead", "pcc", "mae", "pcc_fir", "mae_fir"]]
t1 = fig.add_axes([0.03, 0.01, 0.45, (extra - 1.5) / (H + extra)]); t1.axis("off")
tb = t1.table(cellText=[[r.lead] + [f"{v:.3f}" for v in r.iloc[1:]] for _, r in d.iterrows()], colLabels=["lead", f"PCC {a.method}", "MAE", "PCC FIR", "MAE FIR"], loc="center", fontsize=7)
tb.auto_set_font_size(False); tb.set_fontsize(7); t1.set_title("Per-lead metrics (z units)", fontsize=8)
if jt is not None:
    t2 = fig.add_axes([0.52, 0.01, 0.46, (extra - 1.5) / (H + extra)]); t2.axis("off")
    cells = [[r.iloc[0].split(" ")[0]] + [f"{v:.3f}" for v in r.iloc[1:]] for _, r in jt.iterrows()]
    labs = ["class"] + [c.replace("reconstructed ", "recon. ") for c in jt.columns[1:]]
    tb2 = t2.table(cellText=cells, colLabels=labs, loc="center"); tb2.auto_set_font_size(False); tb2.set_fontsize(7)
    t2.set_title("12-lead classifier probabilities (record level)", fontsize=8)
fig.savefig(a.out, dpi=110)
print("[saved]", a.out); print(info)
