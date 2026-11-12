"""Compose static 'screenshots' of each tab from tools/render_tabs.py output (Plotly figures rendered with kaleido,
tables and texts drawn with PIL) in the app's layout. Used when no headless browser can run; these are renders of
the app's outputs, not browser captures.

  python space/tools/compose_screens.py --dump <dir> --out <dir>
"""
import argparse, json, os, re, textwrap

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = json.load(open(os.path.join(a.dump, "dump.json")))
W = 1500
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except OSError:
        return ImageFont.load_default(size=sz)


def text_block(md, width=W - 40, size=14):
    md = re.sub(r"<[^>]+>", "", md)
    lines = []
    for para in md.split("\n"):
        bold = para.startswith("#")
        p = re.sub(r"^#+\s*", "", para).replace("**", "").replace("`", "").replace("*", "")
        if not p.strip():
            lines.append(("", False)); continue
        for ln in textwrap.wrap(p, width=int(width / (size * 0.56))):
            lines.append((ln, bold))
    h = 10 + len(lines) * int(size * 1.45)
    im = Image.new("RGB", (W, h), "white"); d = ImageDraw.Draw(im)
    y = 5
    for ln, b in lines:
        d.text((20, y), ln, fill="#222222", font=font(size + (2 if b else 0), b)); y += int(size * 1.45)
    return im


def fig_img(name):
    f = pio.from_json(open(os.path.join(a.dump, name + ".json")).read())
    h = int(f.layout.height or 450)
    png = pio.to_image(f, format="png", width=W - 40, height=h, scale=1, engine="kaleido")
    p = os.path.join(a.out, "_tmp.png"); open(p, "wb").write(png)
    im = Image.open(p).convert("RGB")
    out = Image.new("RGB", (W, im.height + 10), "white"); out.paste(im, (20, 5)); return out


def table_img(name, max_rows=40):
    df = pd.read_csv(os.path.join(a.dump, name + ".csv")).head(max_rows)
    df = df.apply(lambda c: c.map(lambda v: "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))))
    f = go.Figure(go.Table(header=dict(values=[f"<b>{c}</b>" for c in df.columns], fill_color="#F3F4F6", align="left",
                                       font=dict(size=11)),
                           cells=dict(values=[df[c].tolist() for c in df.columns], align="left", font=dict(size=11),
                                      height=22)))
    h = 60 + 23 * (len(df) + 1) + 20
    f.update_layout(width=W - 40, height=h, margin=dict(l=5, r=5, t=5, b=5))
    p = os.path.join(a.out, "_tmp.png"); open(p, "wb").write(pio.to_image(f, format="png", engine="kaleido"))
    im = Image.open(p).convert("RGB")
    out = Image.new("RGB", (W, im.height + 10), "white"); out.paste(im, (20, 5)); return out


def img(path, width=W - 200):
    im = Image.open(path).convert("RGB")
    im = im.resize((width, int(im.height * width / im.width)))
    out = Image.new("RGB", (W, im.height + 10), "white"); out.paste(im, (100, 5)); return out


def header(active, config, example, controls):
    im = Image.new("RGB", (W, 330), "white"); d = ImageDraw.Draw(im)
    d.rectangle([20, 12, W - 20, 52], fill="#FFF4E5", outline="#E69F00", width=2)
    d.text((32, 22), "Research demonstration — not for clinical use.", fill="#000000", font=font(16, True))
    d.text((480, 24), "Reconstructed leads are model estimates, not measurements; do not use them for diagnosis or "
                      "patient management.", fill="#333333", font=font(13))
    d.text((20, 62), "BiMa-ECGSR: 12-lead ECG reconstruction from reduced lead sets", fill="#111111", font=font(22, True))
    for i, ln in enumerate(textwrap.wrap(re.sub(r"[`*]", "", D["runtime"]), 190)):
        d.text((20, 96 + 17 * i), ln, fill="#555555", font=font(12))
    d.text((20, 140), "ECG source: (•) Bundled example  ( ) Upload", fill="#222222", font=font(13))
    d.rectangle([20, 162, 840, 192], outline="#BBBBBB"); d.text((28, 169), "Example record: " + example[:110],
                                                                   fill="#222222", font=font(12))
    d.text((20, 200), "Examples: 75th percentile of BiMa-SR per-record PCC for I+II+V1 within each label group; "
                      "PTB-XL, Georgia, Chapman-Shaoxing/Ningbo (CC BY 4.0).", fill="#777777", font=font(11))
    d.text((880, 140), "Measured input leads (use case)", fill="#222222", font=font(13, True))
    for i, (n, c) in enumerate(D["presets"]):
        x = 880 + (i % 3) * 200; y = 165 + (i // 3) * 46
        sel = c == config
        d.rounded_rectangle([x, y, x + 190, y + 40], 6, fill="#E69F00" if sel else "#F3F4F6",
                            outline="#C27C00" if sel else "#CCCCCC")
        d.text((x + 10, y + 4), n, fill="white" if sel else "#222222", font=font(12, True))
        d.text((x + 10, y + 21), c, fill="white" if sel else "#444444", font=font(11))
    names = ["1 · Reconstruct", "2 · Diagnose & explain", "3 · Compare input sets", "4 · Diffusion uncertainty (lead II)",
             "5 · Results & About"]
    x = 20
    for i, n in enumerate(names):
        w = int(len(n) * 8.2) + 24
        on = str(i + 1) == active
        d.text((x + 12, 272), n, fill="#E69F00" if on else "#555555", font=font(14, on))
        if on:
            d.line([x, 298, x + w, 298], fill="#E69F00", width=3)
        x += w + 10
    d.line([20, 300, W - 20, 300], fill="#DDDDDD", width=1)
    if controls:
        d.text((20, 308), controls, fill="#333333", font=font(12))
    return im


def stack(parts):
    h = sum(p.height for p in parts)
    out = Image.new("RGB", (W, h), "white"); y = 0
    for p in parts:
        out.paste(p, (0, y)); y += p.height
    return out


titles = {"1": "tab1_reconstruct", "1b": "tab1_monitor_ddim6", "2": "tab2_diagnose_explain", "3": "tab3_compare",
          "4": "tab4_uncertainty", "5": "tab5_results_about"}
for k, t in D["tabs"].items():
    parts = [header(k[0], t.get("config", D["config"]), D["example"], t["controls"])]
    for kind, v in t["items"]:
        if v is None:
            continue
        parts.append({"md": text_block, "fig": fig_img, "table": table_img, "img": img}[kind](v))
    foot = text_block("[static render of the app's outputs (Plotly figures via kaleido, tables and text via PIL); "
                      "not a browser capture]", size=11)
    out = stack(parts + [foot])
    p = os.path.join(a.out, titles[k] + ".png"); out.save(p, optimize=True)
    print(p, out.size)
os.remove(os.path.join(a.out, "_tmp.png"))
