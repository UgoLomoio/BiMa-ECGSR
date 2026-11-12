"""Plotly figures of the BiMa-ECGSR Space. All trace data are passed as plain Python lists (no typed arrays)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
COL = {"real": "#000000", "input": "#CC79A7", "BiMa-SR": "#E69F00", "DiBiMa-SR (one-step)": "#0072B2",
       "DiBiMa-SR (DDIM-6)": "#56B4E9", "filter": "#999999", "band": "rgba(0,114,178,0.22)"}
GRID = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
FONT = dict(family="Arial, Helvetica, sans-serif", size=11)


def L(a, nd=4):
    """numpy -> plain list (rounded; NaN -> None)."""
    a = np.round(np.asarray(a, dtype=np.float64), nd)
    return [None if not np.isfinite(v) else float(v) for v in a.ravel()]


def _cat(a):
    """[W, 12, Lw] -> [12, W*Lw]."""
    return None if a is None else np.concatenate(list(a), axis=-1)


def _style(fig, height, title=None):
    fig.update_layout(height=height, margin=dict(l=40, r=15, t=95 if title else 40, b=35), font=FONT,
                      title=dict(text=title, x=0.01, y=0.995, yref="container", yanchor="top", font=dict(size=13))
                      if title else None,
                      legend=dict(orientation="h", y=1.0, x=0.0, yanchor="bottom", font=dict(size=11)),
                      plot_bgcolor="white", paper_bgcolor="white", hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False, ticks="outside", tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False, tickfont=dict(size=9))
    return fig


def clinical(series, measured, title, unit="mV", band=None):
    """Clinical 4 x 3 layout (I, II, III | aVR, aVL, aVF | V1-V3 | V4-V6; column j shows the j-th quarter of the record)
    plus a full-length lead II rhythm strip.
    series: list of dict(name, data [W,12,Lw] or [12,n], color, dash=None, width=1, role='pred'|'real'|'input').
    measured: indices of measured leads (drawn once, pink, from the 'real'/'input' series).
    band: optional dict(lo, hi [12,n] or [W,12,Lw], name) drawn as a filled band (predicted leads)."""
    specs = [[{}, {}, {}, {}] for _ in range(3)] + [[{"colspan": 4}, None, None, None]]
    fig = make_subplots(rows=4, cols=4, specs=specs, vertical_spacing=0.06, horizontal_spacing=0.035,
                        row_heights=[1, 1, 1, 1.1])
    data = [dict(s, data=_cat(s["data"]) if np.ndim(s["data"]) == 3 else s["data"]) for s in series if s["data"] is not None]
    n = data[0]["data"].shape[-1]
    t = np.arange(n) / FS
    q = n // 4
    if band is not None:
        band = {k: (_cat(v) if np.ndim(v) == 3 else v) if k in ("lo", "hi") else v for k, v in band.items()}
    shown = set()

    def add(lead, sl, row, col):
        c = LEADS.index(lead)
        for s in data:
            if c in measured and s.get("role") != "real":
                continue
            if c not in measured and s.get("role") == "input":
                continue
            y = s["data"][c, sl]
            if not np.isfinite(y).any():
                continue
            nm = "measured input" if c in measured else s["name"]
            color = COL["input"] if c in measured else s["color"]
            fig.add_trace(go.Scatter(x=L(t[sl], 4), y=L(y), mode="lines", name=nm, legendgroup=nm,
                                     showlegend=nm not in shown,
                                     line=dict(color=color, width=s.get("width", 1.0), dash=s.get("dash")),
                                     hovertemplate=f"{lead} {nm}<br>%{{x:.3f}} s<br>%{{y:.3f}} {unit}<extra></extra>"),
                          row=row, col=col)
            shown.add(nm)

    def add_band(lead, sl, row, col):
        c = LEADS.index(lead)
        if band is None or c in measured:
            return
        nm = band.get("name", "range")
        fig.add_trace(go.Scatter(x=L(t[sl], 4), y=L(band["lo"][c, sl]), mode="lines", line=dict(width=0),
                                 showlegend=False, legendgroup=nm, hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(x=L(t[sl], 4), y=L(band["hi"][c, sl]), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=COL["band"], name=nm, legendgroup=nm,
                                 showlegend=nm not in shown, hoverinfo="skip"), row=row, col=col)
        shown.add(nm)

    for r in range(3):
        for j in range(4):
            lead = GRID[r][j]
            sl = slice(j * q, (j + 1) * q if j < 3 else n)
            add_band(lead, sl, r + 1, j + 1)
            add(lead, sl, r + 1, j + 1)
            ax = fig.get_subplot(r + 1, j + 1)
            fig.add_annotation(x=0.01, y=0.98, xref=f"{ax.xaxis.plotly_name.replace('axis', '')} domain",
                               yref=f"{ax.yaxis.plotly_name.replace('axis', '')} domain", text=f"<b>{lead}</b>"
                               + (" (measured)" if LEADS.index(lead) in measured else ""),
                               showarrow=False, xanchor="left", yanchor="top", font=dict(size=11))
    add_band("II", slice(0, n), 4, 1)
    add("II", slice(0, n), 4, 1)
    ax = fig.get_subplot(4, 1)
    fig.add_annotation(x=0.003, y=0.98, xref=f"{ax.xaxis.plotly_name.replace('axis', '')} domain",
                       yref=f"{ax.yaxis.plotly_name.replace('axis', '')} domain",
                       text="<b>II</b> rhythm strip" + (" (measured)" if 1 in measured else ""), showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=11))
    fig.update_xaxes(title_text="time (s)", row=4, col=1)
    return _style(fig, 860, title)


def class_bars(probs, gt, highlight_thr=0.10, labels=None, names=None):
    """probs: dict name -> [17]; gt: list of positive labels (or None). Horizontal grouped bars."""
    labels = labels or []
    order = list(range(len(labels)))[::-1]
    real = probs.get("real 12-lead ECG")
    recon_key = next((k for k in probs if k.startswith("reconstruction")), None)
    ticks = []
    for i in order:
        lab = labels[i]
        txt = lab + (" ●" if gt and lab in gt else "")
        if real is not None and recon_key and abs(probs[recon_key][i] - real[i]) > highlight_thr:
            txt = f"<b><span style='color:#D55E00'>{txt} Δ</span></b>"
        ticks.append(txt)
    fig = go.Figure()
    colors = {"real 12-lead ECG": COL["real"], "linear filter": COL["filter"], "Einthoven + linear filter": COL["filter"]}
    for k, p in probs.items():
        col = colors.get(k) or next((v for m, v in COL.items() if m in k), COL["BiMa-SR"])
        fig.add_trace(go.Bar(y=[labels[i] for i in order], x=L([p[i] for i in order], 3), orientation="h", name=k,
                             marker=dict(color=col, line=dict(width=0)),
                             hovertemplate="%{y}: %{x:.3f}<extra>" + k + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(tickmode="array", tickvals=[labels[i] for i in order], ticktext=ticks)
    fig.update_xaxes(range=[0, 1], title_text="classifier probability (record = mean of the two 5-s windows)")
    _style(fig, 640, "Classifier probabilities - ● ground-truth label, <span style='color:#D55E00'>Δ</span> |real - "
                     f"reconstruction| > {highlight_thr:.2f}")
    return fig


def _bin(a, b=20):
    n = a.shape[-1] // b * b
    return np.abs(a[..., :n]).reshape(*a.shape[:-1], -1, b).sum(-1)


def ig_heat(sig_real, att_real, sig_rec, att_rec, measured, method, cls_name, binsz=20):
    """12 rows x 2 columns (real | reconstruction): ECG trace (z units) over |IG| shading binned to 25 Hz,
    one colour scale shared by all panels, normalised to the record maximum."""
    cols = [("real ECG", sig_real, att_real), (f"reconstruction ({method})", sig_rec, att_rec)]
    cols = [c for c in cols if c[1] is not None]
    nc = len(cols)
    fig = make_subplots(rows=12, cols=nc, shared_xaxes=True, vertical_spacing=0.008, horizontal_spacing=0.04,
                        column_titles=[c[0] for c in cols])
    bins = [_bin(_cat(a), binsz) for _, _, a in cols]
    vmax = max(float(b.max()) for b in bins) or 1.0
    for j, ((name, sig, _), B) in enumerate(zip(cols, bins)):
        s = _cat(sig); n = s.shape[-1]; t = np.arange(n) / FS
        tb = (np.arange(B.shape[-1]) + 0.5) * binsz / FS
        for c in range(12):
            y = s[c]
            lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
            fig.add_trace(go.Heatmap(x=L(tb, 3), y=[lo, hi], z=[L(B[c] / vmax, 3)] * 2, coloraxis="coloraxis",
                                     hovertemplate=f"{LEADS[c]} %{{x:.2f}} s |IG| %{{z:.2f}}<extra></extra>"),
                          row=c + 1, col=j + 1)
            is_real = name == "real ECG"
            color = "#000000" if is_real else (COL["input"] if c in measured else COL.get(method, COL["BiMa-SR"]))
            fig.add_trace(go.Scatter(x=L(t[::2], 4), y=L(y[::2]), mode="lines", line=dict(color=color, width=0.8),
                                     showlegend=False, hoverinfo="skip"), row=c + 1, col=j + 1)
            if j == 0:
                fig.update_yaxes(title_text=LEADS[c] + ("*" if c in measured else ""), row=c + 1, col=1,
                                 title_font=dict(size=10))
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(coloraxis=dict(colorscale="YlOrRd", cmin=0, cmax=1,
                                     colorbar=dict(title="|IG|<br>(norm.)", thickness=10, len=0.4)))
    fig.update_xaxes(title_text="time (s)", row=12)
    return _style(fig, 1150, f"Integrated Gradients for <b>{cls_name}</b> (shading = |attribution|, 25-Hz bins, one "
                             "colour scale normalised to the record maximum; * measured lead)")


def lead_importance(share_real, share_rec, measured, method):
    ticks = [f"<span style='color:#CC79A7'><b>{l}</b></span>" if i in measured else l for i, l in enumerate(LEADS)]
    fig = go.Figure()
    if share_real is not None:
        fig.add_trace(go.Bar(x=LEADS, y=L(share_real, 4), name="real ECG", marker_color="#000000"))
    fig.add_trace(go.Bar(x=LEADS, y=L(share_rec, 4), name=f"reconstruction ({method})",
                         marker_color=COL.get(method, COL["BiMa-SR"])))
    fig.update_xaxes(tickmode="array", tickvals=LEADS, ticktext=ticks)
    fig.update_yaxes(title_text="share of |IG| attribution", tickformat=".0%")
    fig.update_layout(barmode="group")
    msh = float(np.sum([share_rec[i] for i in measured]))
    t = f"Lead importance (pink = measured). Reconstruction: {msh:.0%} on measured, {1 - msh:.0%} on reconstructed leads"
    if share_real is not None:
        t += f"; real ECG: {float(np.sum([share_real[i] for i in measured])):.0%} on the same leads"
    return _style(fig, 380, t)


def recon_profile(inp_sig, gxi_lead, in_names, pred_name, t0, t1, binsz=20):
    """Rows = measured leads: the measured input (z units) over |grad x input| shading for one predicted lead."""
    k = len(in_names)
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    S = _cat(inp_sig); G = _cat(gxi_lead)
    n = S.shape[-1]; t = np.arange(n) / FS
    B = _bin(G, binsz); vmax = float(B.max()) or 1.0
    tb = (np.arange(B.shape[-1]) + 0.5) * binsz / FS
    for i in range(k):
        lo, hi = float(np.nanmin(S[i])), float(np.nanmax(S[i]))
        fig.add_trace(go.Heatmap(x=L(tb, 3), y=[lo, hi], z=[L(B[i] / vmax, 3)] * 2, coloraxis="coloraxis",
                                 hovertemplate=f"{in_names[i]} %{{x:.2f}} s |grad x input| %{{z:.2f}}<extra></extra>"),
                      row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=L(t, 4), y=L(S[i]), mode="lines", line=dict(color=COL["input"], width=1),
                                 name=f"{in_names[i]} (measured)", showlegend=False,
                                 hovertemplate=f"{in_names[i]} %{{x:.3f}} s %{{y:.3f}}<extra></extra>"), row=i + 1, col=1)
        fig.update_yaxes(title_text=in_names[i], row=i + 1, col=1)
        W = n // 2500
        for w in range(W):
            if t0 > 0 or t1 < 2500:
                fig.add_vrect(x0=(w * 2500 + t0) / FS, x1=(w * 2500 + t1) / FS, line=dict(color="#0072B2", width=1),
                              fillcolor="rgba(0,0,0,0)", row=i + 1, col=1)
    fig.update_layout(coloraxis=dict(colorscale="YlOrRd", cmin=0, cmax=1, colorbar=dict(title="|g·x|", thickness=10)))
    fig.update_xaxes(title_text="time (s)", row=k, col=1)
    return _style(fig, 160 + 150 * k, f"What the reconstruction of <b>{pred_name}</b> relies on: |gradient x input| "
                                      "over the measured leads (25-Hz bins)")


def lead_matrix_fig(M, pred_names, in_names):
    fig = go.Figure(go.Heatmap(z=[L(r, 3) for r in M], x=in_names, y=pred_names, colorscale="Blues", zmin=0, zmax=1,
                               text=[[f"{v:.0%}" for v in r] for r in M], texttemplate="%{text}",
                               hovertemplate="%{y} <- %{x}: %{z:.1%}<extra></extra>", colorbar=dict(title="share")))
    fig.update_yaxes(autorange="reversed", title_text="predicted lead")
    fig.update_xaxes(title_text="measured lead", side="top")
    return _style(fig, 140 + 32 * len(pred_names), "Lead-to-lead attribution: share of |gradient x input| per measured lead")


def small_multiples(real, preds, measured_by_cfg, cfgs, window=0):
    """rows = configurations, columns = V1-V6 (first 5-s window)."""
    chest = list(range(6, 12))
    fig = make_subplots(rows=len(cfgs), cols=6, shared_xaxes=True, vertical_spacing=0.02, horizontal_spacing=0.012,
                        column_titles=LEADS[6:], row_titles=cfgs)
    t = np.arange(real.shape[-1]) / FS
    shown = set()
    for r, c in enumerate(cfgs):
        for j, lead in enumerate(chest):
            meas = lead in measured_by_cfg[c]
            if np.isfinite(real[window, lead]).all():
                nm = "measured input" if meas else "real"
                fig.add_trace(go.Scatter(x=L(t[::2], 4), y=L(real[window, lead, ::2]), mode="lines",
                                         line=dict(color=COL["input"] if meas else "#000000", width=0.9 if meas else 0.7),
                                         name=nm, legendgroup=nm, showlegend=nm not in shown, hoverinfo="skip"),
                              row=r + 1, col=j + 1)
                shown.add(nm)
            if not meas:
                fig.add_trace(go.Scatter(x=L(t[::2], 4), y=L(preds[c][window, lead, ::2]), mode="lines",
                                         line=dict(color=COL["BiMa-SR"], width=0.9), name="BiMa-SR", legendgroup="b",
                                         showlegend="b" not in shown,
                                         hovertemplate=f"{c} {LEADS[lead]} %{{x:.2f}} s %{{y:.2f}}<extra></extra>"),
                              row=r + 1, col=j + 1)
                shown.add("b")
            else:
                ax = fig.get_subplot(r + 1, j + 1)
                fig.add_annotation(x=0.5, y=0.97, xref=f"{ax.xaxis.plotly_name.replace('axis', '')} domain",
                                   yref=f"{ax.yaxis.plotly_name.replace('axis', '')} domain", text="measured",
                                   showarrow=False, font=dict(size=9, color=COL["input"]), yanchor="top")
    fig.update_yaxes(showticklabels=False)
    fig.update_annotations(font_size=10)
    return _style(fig, 130 + 120 * len(cfgs), "Chest leads by input set (first 5-s window, z units; display at 250 Hz)")


def pcc_by_config(cfgs, rec_bima, rec_base, base_names, ref_bima, ref_base):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cfgs, y=L(ref_base, 4), mode="markers", name="paper, PTB-XL test: baseline",
                             marker=dict(color=COL["filter"], symbol="circle-open", size=11, opacity=0.5)))
    fig.add_trace(go.Scatter(x=cfgs, y=L(ref_bima, 4), mode="markers", name="paper, PTB-XL test: BiMa-SR",
                             marker=dict(color=COL["BiMa-SR"], symbol="circle-open", size=11, opacity=0.5)))
    fig.add_trace(go.Scatter(x=cfgs, y=L(rec_base, 4), mode="lines+markers", name="this record: baseline",
                             text=base_names, line=dict(color=COL["filter"], dash="dash"),
                             hovertemplate="%{x}: %{y:.3f} (%{text})<extra></extra>"))
    fig.add_trace(go.Scatter(x=cfgs, y=L(rec_bima, 4), mode="lines+markers", name="this record: BiMa-SR",
                             line=dict(color=COL["BiMa-SR"]), hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="chest-lead PCC (pooled median, predicted V1-V6)")
    fig.update_xaxes(title_text="measured leads")
    return _style(fig, 380, "Chest-lead accuracy vs input set (baseline: linear filter; Einthoven + linear filter "
                            "when I and II are measured)")
