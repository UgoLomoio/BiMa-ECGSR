"""BiMa-ECGSR: reconstruct the 12-lead ECG from a reduced lead set (Hugging Face Space, ZeroGPU with CPU fallback).

Research demonstration - not for clinical use. Specification: APP_SPEC.md.

Device: every network call runs inside one of four @spaces.GPU functions (reconstruct, compare-all, uncertainty,
attribution); preprocessing, the linear baselines, metrics and plotting stay on the CPU. Without ZeroGPU (no `spaces`
package or not a ZeroGPU Space) the same functions run undecorated: on CUDA if available, else on the CPU.
Override with BIMA_DEVICE=cpu|cuda.
"""
from __future__ import annotations

import os
import tempfile
import time

try:
    import spaces  # ZeroGPU
except ImportError:
    spaces = None

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import engine as E  # noqa: E402
import explain as XP  # noqa: E402
import figures as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZERO_GPU = spaces is not None and os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")
DEVICE = os.environ.get("BIMA_DEVICE") or ("cuda" if (ZERO_GPU or torch.cuda.is_available()) else "cpu")
CALL = "GPU call" if DEVICE == "cuda" else "inference call"
DUR = dict(reconstruct=60, compare=90, uncertainty=90, explain=120)   # seconds requested per ZeroGPU call

BANNER = ("<div style='background:#FFF4E5;border:1px solid #E69F00;border-radius:6px;padding:8px 12px;font-size:15px'>"
          "<b>Research demonstration — not for clinical use.</b> Reconstructed leads are model estimates, not "
          "measurements; do not use them for diagnosis or patient management.</div>")
PRESETS = [("Smartwatch", "I", "wrist single-lead devices"), ("Monitor", "II", "bedside / Holter rhythm strip"),
           ("6-lead device", "I+II", "limb-lead devices (6 limb leads)"),
           ("Telemetry", "I+II+V1", "5-electrode bedside telemetry"),
           ("3-lead set", "I+II+V2", "reduced-lead challenge set"), ("4-lead set", "I+II+V1+V4", "limb + two chest leads")]
DEFAULT_CFG = "I+II+V1"
CFG_ORDER = ["I", "II", "I+II", "I+II+V1", "I+II+V2", "I+II+V1+V4"]      # by number of measured leads (paper order)
DIX = {l: i for i, l in enumerate(E.LABELS17)}


def gpu(duration):
    """spaces.GPU on ZeroGPU, no-op elsewhere."""
    if not ZERO_GPU:
        return lambda f: f
    return spaces.GPU(duration=duration)


PRELOAD = E.preload(DEVICE)    # models + classifier on DEVICE (ZeroGPU defers .to('cuda') until a GPU is attached)


# ----------------------------------------------------------------------------- GPU entry points
@gpu(DUR["reconstruct"])
def gpu_reconstruct(z, st, config, method, base, want_judge, real_ok):
    return E.infer(z, st, config, method, DEVICE, fir=base, want_judge=want_judge, real_ok=real_ok)


@gpu(DUR["compare"])
def gpu_compare(z, st):
    t0 = time.perf_counter()
    res = E.reconstruct_all(z, st, DEVICE)
    return {c: p for c, (p, _) in res.items()}, dict(next(iter(res.values()))[1], call_s=time.perf_counter() - t0,
                                                      model_ms={c: i["model_ms"] for c, (_, i) in res.items()},
                                                      backends={c: i["backend"] for c, (_, i) in res.items()})


@gpu(DUR["uncertainty"])
def gpu_uncertainty(z, st, n):
    t0 = time.perf_counter()
    draws, info = E.diffusion_draws(z, st, n, DEVICE)
    bima, ib = E.reconstruct(z, st, "II", E.METHOD_BIMA, DEVICE)
    info.update(bima_ms=ib["model_ms"], call_s=time.perf_counter() - t0)
    return draws, bima, info


@gpu(DUR["explain"])
def gpu_explain(z, pred, st, config, method, cls, t0, t1, real_ok, leads=None):
    tt = time.perf_counter()
    ig, gxi, info = E.explain(z, pred, st, config, method, cls, t0, t1, DEVICE, real_ok, leads)
    info["call_s"] = time.perf_counter() - tt
    return ig, gxi, info


# ----------------------------------------------------------------------------- helpers
def available_cfgs():
    return [c for c in CFG_ORDER if c in E.available()]


def example_choices():
    return [E.example_label(e) for e in E.EXAMPLES]


def runtime_md():
    ok, _ = E.mamba_ssm_importable()
    where = "ZeroGPU (a GPU is attached per request)" if ZERO_GPU else ("local CUDA GPU" if DEVICE == "cuda" else "CPU")
    be = ("`mamba_ssm` CUDA kernels if they pass a one-window self-test against the pure-PyTorch Mamba "
          f"(|Δ| < {E.SELFTEST_TOL:g}), else the pure-PyTorch Mamba on CUDA") if DEVICE == "cuda" else \
        "pure-PyTorch Mamba (numba selective scan)"
    return (f"Runtime: {where}; device `{DEVICE}`; Mamba backend: {be}; mamba_ssm importable: {'yes' if ok else 'no'}; "
            f"{PRELOAD['n_models']} model instances loaded in {PRELOAD['seconds']:.1f} s.")


def run_info(tm, extra="", t_all=None):
    s = (f"**Run**: {tm['device_name']} (`{tm['device']}`), Mamba backend `{tm['backend']}` ({tm['selftest']}); "
         f"{tm.get('precision', 'float32')}.")
    if extra:
        s += " " + extra
    if t_all is not None:
        s += f" Total request {time.perf_counter() - t_all:.2f} s" + (" (incl. GPU allocation)." if ZERO_GPU else ".")
    return s


def _err(fn):
    def wrapped(*a):
        import gradio as gr
        try:
            return fn(*a)
        except E.InputError as exc:
            raise gr.Error(str(exc))
    wrapped.__name__ = fn.__name__
    return wrapped


def prepare(source, example, files, fs_csv, config):
    """Read + preprocess the record. Returns dict(desc, notes, z, st, present, windows, gt, key)."""
    if source == "Bundled example":
        idx = example_choices().index(example)
        x, desc = E.read_example(idx); notes = []
        e = E.EXAMPLES[idx]; gt = e["labels"]; key = ("ex", idx)
    else:
        paths = [getattr(f, "name", f) for f in (files or [])]
        x, desc, notes = E.read_upload(paths, config, fs_csv)
        gt = None; key = ("up", tuple(sorted(os.path.basename(p) for p in paths)), float(fs_csv or 500))
    in_leads = E.CONFIG_LEADS[config]
    missing = [l for l in in_leads if not np.isfinite(x[E.LEADS.index(l)]).all()]
    if missing:
        raise E.InputError(f"measured lead(s) {', '.join(missing)} of the {config} configuration are not in the record")
    pp = E.preprocess(x)
    return dict(desc=desc, notes=notes + pp["notes"], z=pp["z"], st=pp["stats"], present=pp["present"],
                windows=pp["windows"], gt=gt, key=key)


def to_mv(zp, st, config):
    return E.to_mv(zp, st, config, True)


def real_mv(r):
    z, st, present = r["z"], r["st"], r["present"]
    return np.where(present[None, :, None], z * st[..., 1:2] + st[..., 0:1], np.nan)


def metrics_table(pred, base, r, config, method, bname):
    """Per predicted lead PCC / MAE (z) + summary rows (pooled median PCC over windows x leads; mean MAE)."""
    in_idx = [E.LEADS.index(l) for l in E.CONFIG_LEADS[config]]
    pl = [c for c in range(12) if c not in in_idx and r["present"][c]]
    if not pl:
        return None
    pcc, mae, _ = E.T.metrics_np(pred, np.nan_to_num(r["z"]))
    rows = []
    if base is not None:
        bp, bm, _ = E.T.metrics_np(base, np.nan_to_num(r["z"]))
    for c in pl:
        row = {"lead": E.LEADS[c], f"PCC {method}": pcc[:, c].mean(), f"MAE {method}": mae[:, c].mean()}
        if base is not None:
            row[f"PCC {bname}"] = bp[:, c].mean(); row[f"MAE {bname}"] = bm[:, c].mean()
        rows.append(row)
    for name, cols in (("all predicted leads", pl), ("predicted chest leads", [c for c in pl if c in E.CHEST])):
        if not cols:
            continue
        row = {"lead": name, f"PCC {method}": float(np.median(pcc[:, cols])), f"MAE {method}": float(mae[:, cols].mean())}
        if base is not None:
            row[f"PCC {bname}"] = float(np.median(bp[:, cols])); row[f"MAE {bname}"] = float(bm[:, cols].mean())
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def top3_line(jp, method):
    sup = list(range(5))
    parts = []
    for k, lab in (("real", "real ECG"), ("pred", f"reconstruction ({method})")):
        if k in jp:
            p = jp[k]
            top = sorted(sup, key=lambda i: -p[i])[:3]
            parts.append(f"{lab}: " + ", ".join(f"{E.LABELS17[i]} {p[i]:.2f}" for i in top))
    return "**Classifier (superclasses, top 3)** — " + " | ".join(parts) + " — full view and explanations in the " \
        "*Diagnose & explain* tab."


def downloads(pred_mv, config, method, desc):
    d = tempfile.mkdtemp(prefix="bima_dl_")
    X = np.concatenate(list(pred_mv), axis=-1).astype(np.float32)        # [12, n] mV
    stem = f"bima_ecgsr_{config.replace('+', '_')}_{method.split(' ')[0]}"
    npy = os.path.join(d, stem + ".npy"); np.save(npy, X)
    csv = os.path.join(d, stem + ".csv")
    df = pd.DataFrame(X.T, columns=E.LEADS); df.insert(0, "time_s", np.arange(X.shape[1]) / E.FS)
    with open(csv, "w") as f:
        f.write(f"# BiMa-ECGSR reconstruction ({method}, measured leads {config}) of {desc}. Units mV at 500 Hz, "
                "band-passed 0.5-40 Hz. Measured leads unchanged; predicted leads = z-score prediction x training-set "
                "amplitude prior (median std_lead / input scale), zero mean. Research use only - not for clinical use.\n")
        df.to_csv(f, index=False, float_format="%.5f")
    return csv, npy


# ----------------------------------------------------------------------------- Tab 1
def do_reconstruct(source, example, files, fs_csv, config, method, show_filter, run_judge):
    t_all = time.perf_counter()
    if not config:
        raise E.InputError("choose an input-lead preset (no configuration with weights available)")
    if method != E.METHOD_BIMA and config != "II":
        method = E.METHOD_BIMA
    r = prepare(source, example, files, fs_csv, config)
    z, st, present = r["z"], r["st"], r["present"]
    tb = time.perf_counter()
    base = E.baseline(z, st, config); bname = E.baseline_name(config)
    base_ms = (time.perf_counter() - tb) * 1000
    pred, jp, tm = gpu_reconstruct(z, st, config, method, base, bool(run_judge), bool(present.all()))
    in_idx = [E.LEADS.index(l) for l in E.CONFIG_LEADS[config]]
    series = [dict(name="real", data=real_mv(r), color="#000000", width=0.9, role="real")]
    if show_filter and base is not None:
        series.append(dict(name=bname, data=to_mv(base, st, config), color=F.COL["filter"], dash="dash", width=0.9,
                           role="pred"))
    pred_mv = to_mv(pred, st, config)
    series.append(dict(name=method, data=pred_mv, color=F.COL[method], width=1.1, role="pred"))
    fig = F.clinical(series, in_idx, f"{r['desc']} — {method} from {config} (mV; predicted leads scaled by the "
                                     "training-set amplitude prior)")
    table = metrics_table(pred, base, r, config, method, bname)
    cls_md = top3_line(jp, method) if jp is not None else "Classifier not run."
    nw = z.shape[0]
    info = run_info(tm, f"Model latency {tm['model_ms']:.1f} ms per {nw * 5}-s record ({nw} window(s))"
                        + (f"; classifier {tm['judge_ms']:.1f} ms" if "judge_ms" in tm else "")
                        + f"; {bname} {base_ms:.1f} ms on CPU.", t_all)
    notes = "".join(f"\n\nNote: {n}." for n in r["notes"])
    if table is None:
        notes += "\n\nNo real reference for the predicted leads (only the measured leads were provided): no metrics."
    elif "I" in E.CONFIG_LEADS[config] and "II" in E.CONFIG_LEADS[config]:
        notes += ("\n\nWith I and II measured, the limb leads III, aVR, aVL and aVF are exactly derivable (Einthoven/"
                  "Goldberger); the chest leads are the informative comparison.")
    csv, npy = downloads(pred_mv, config, method, r["desc"])
    state = dict(r, config=config, method=method, pred=pred, base=base, bname=bname, jp=jp, tm=tm)
    return fig, table, cls_md, info + notes, csv, npy, state


# ----------------------------------------------------------------------------- Tab 2
def class_view(state, show_filter):
    import gradio as gr
    if not state or state.get("jp") is None:
        return None, "Run a reconstruction with the classifier enabled (Reconstruct tab or the button below).", \
            gr.update()
    jp, method = state["jp"], state["method"]
    probs = {}
    if "real" in jp:
        probs["real 12-lead ECG"] = jp["real"]
    probs[f"reconstruction ({method})"] = jp["pred"]
    if show_filter and "fir" in jp:
        probs[state["bname"]] = jp["fir"]
    fig = F.class_bars(probs, state["gt"], labels=E.LABELS17)
    gt = state["gt"]
    md = (f"**Record**: {state['desc']}. **Measured leads**: {state['config']}. **Ground truth** (● in the chart): "
          + (", ".join(f"{l} ({E.LABEL_NAMES[l]})" for l in gt) if gt else ("none of the 17 labels" if gt is not None
                                                                            else "not available for uploads")) + ".")
    if "real" in jp:
        d = [l for i, l in enumerate(E.LABELS17) if abs(jp["pred"][i] - jp["real"][i]) > 0.10]
        md += f" Labels with |real − reconstruction| > 0.10: {', '.join(d) if d else 'none'}."
    md += ("\n\n*A reconstruction cannot add information absent from the measured leads; agreement with the real ECG "
           "means the reconstruction is readable by a 12-lead tool, not that it is diagnostic.*")
    ref = jp["real"] if "real" in jp else jp["pred"]
    top = E.LABELS17[int(np.argmax(ref))]
    return fig, md, gr.update(value=f"{top} ({E.LABEL_NAMES[top]})")


def class_choices():
    return [f"{l} ({E.LABEL_NAMES[l]})" for l in E.LABELS17]


def pred_lead_choices(config):
    in_idx = [E.LEADS.index(l) for l in E.CONFIG_LEADS[config]]
    return [l for i, l in enumerate(E.LEADS) if i not in in_idx]


def do_explain(state, cls_label, t_start, t_end, source, example, files, fs_csv, config, method, show_filter,
               lead_sel=None):
    import gradio as gr
    t_all = time.perf_counter()
    cur = (source, example if source == "Bundled example" else None, config)
    rerun = not state or state.get("ui") != cur or state.get("jp") is None
    if rerun:                                  # no reconstruction for the current selection yet: run Tab 1's pipeline
        out = do_reconstruct(source, example, files, fs_csv, config, method, show_filter, True)
        state = out[-1]; state["ui"] = cur
    z, pred, st = state["z"], state["pred"], state["st"]
    config, method = state["config"], state["method"]
    cls = DIX[cls_label.split(" ")[0]]
    s0, s1 = int(round(min(t_start, t_end) * E.FS)), int(round(max(t_start, t_end) * E.FS))
    s0, s1 = max(0, s0), min(E.WIN, max(s1, s0 + 50))
    real_ok = bool(state["present"].all())
    all_pred = pred_lead_choices(config)
    leads = None
    if DEVICE == "cpu":     # CPU: one predicted lead per call (~20-80 s each); the full matrix needs the GPU path
        lead = lead_sel if lead_sel in all_pred else all_pred[0]
        leads = [E.LEADS.index(lead)]
    ig, gxi, info = gpu_explain(z, pred, st, config, method, cls, s0, s1, real_ok, leads)
    in_idx, pl = info["in_idx"], info["pred_leads"]
    cname = f"{E.LABELS17[cls]} ({E.LABEL_NAMES[E.LABELS17[cls]]})"
    fig_ig = F.ig_heat(z if real_ok else None, ig["real"]["attr"] if real_ok else None, pred, ig["recon"]["attr"],
                       in_idx, method, cname)

    def share(a):
        s = np.abs(a).sum((0, 2)); return s / s.sum()
    fig_imp = F.lead_importance(share(ig["real"]["attr"]) if real_ok else None, share(ig["recon"]["attr"]), in_idx, method)
    M = XP.lead_matrix(gxi)
    in_names = [E.LEADS[i] for i in in_idx]; pnames = [E.LEADS[c] for c in pl]
    fig_mat = F.lead_matrix_fig(M, pnames, in_names)
    inp = E.T.norm_inputs_np(np.nan_to_num(z), np.nan_to_num(st, nan=1.0), in_idx)
    sel = lead_sel if lead_sel in pnames else pnames[0]
    fig_prof = F.recon_profile(inp, gxi[pnames.index(sel)], in_names, sel, s0, s1)
    comp = []
    for k in ("real", "recon"):
        if k in ig:
            g = ig[k]
            comp.append(f"{'real ECG' if k == 'real' else 'reconstruction'}: Σattr − (f(x) − f(0)) relative error "
                        f"{float(g['rel_err'].max()):.1%} (max over windows; {g['steps']} Gauss-Legendre steps; "
                        f"f(x) − f(0) = {', '.join(f'{v:.2f}' for v in g['f_x'] - g['f_0'])} logit)")
    md = (f"**Attribution run**: {info['device_name']} (`{info['device']}`), float32. Integrated Gradients "
          f"{info['ig_ms']:.0f} ms; reconstruction gradients {info['recon_attr_ms']:.0f} ms with the Mamba backend "
          f"`{info['attr_backend']}`" + (f" ({info['attr_note']})" if info["attr_note"] else "")
          + f"; {CALL} {info['call_s']:.2f} s; total {time.perf_counter() - t_all:.2f} s.\n\n**IG completeness**: "
          + "; ".join(comp) + ".\n\nReconstruction attribution window: "
          f"{s0 / E.FS:.2f}-{s1 / E.FS:.2f} s of each 5-s window (sum of the predicted lead over that span)."
          + ("\n\nCPU mode: reconstruction attribution computed for the selected predicted lead only (choose another "
             "lead and press *Compute explanations* again); the full lead-to-lead matrix is computed on the GPU."
             if leads else "")
          + "\n\n*Saliency maps are heuristic; they show sensitivity of the model output, not a clinical mechanism.*")
    attr_state = dict(gxi=gxi, inp=inp, in_names=in_names, pnames=pnames, s0=s0, s1=s1)
    return (fig_ig, fig_imp, fig_prof, fig_mat, md, gr.update(choices=all_pred, value=sel), attr_state,
            state if rerun else gr.skip())


def redraw_profile(attr_state, lead):
    if not attr_state or lead not in attr_state["pnames"]:
        return None
    i = attr_state["pnames"].index(lead)
    return F.recon_profile(attr_state["inp"], attr_state["gxi"][i], attr_state["in_names"], lead, attr_state["s0"],
                           attr_state["s1"])


# ----------------------------------------------------------------------------- Tab 3
def _chest_pcc(pred, z, present, in_idx):
    cols = [c for c in E.CHEST if c not in in_idx and present[c]]
    if not cols:
        return np.nan
    pcc, _, _ = E.T.metrics_np(pred, np.nan_to_num(z))
    return float(np.median(pcc[:, cols]))


def do_compare(source, example, files, fs_csv, config):
    t_all = time.perf_counter()
    r = prepare(source, example, files, fs_csv, config)
    z, st, present = r["z"], r["st"], r["present"]
    cfgs = available_cfgs()
    usable = [c for c in cfgs if all(np.isfinite(z[:, E.LEADS.index(l)]).all() for l in E.CONFIG_LEADS[c])]
    if not usable:
        raise E.InputError("the record does not contain the measured leads of any configuration")
    preds, info = gpu_compare(z, st) if usable == cfgs else gpu_compare_subset(z, st, usable)
    meas = {c: [E.LEADS.index(l) for l in E.CONFIG_LEADS[c]] for c in usable}
    zr = np.where(present[None, :, None], z, np.nan)
    fig_sm = F.small_multiples(zr, preds, meas, usable)
    t1 = pd.read_csv(os.path.join(HERE, "assets", "results", "table1_reconstruction.csv"))
    t1 = t1[t1.cohort == "ptbxl_test"]
    spec = {lab: s for lab, s, _ in E.CFG.CONFIGS}
    ref_b = [float(t1[t1.config == spec[c]].bima_prec_pred_pcc.iloc[0]) for c in usable]
    ref_f = [float(t1[t1.config == spec[c]].base_prec_pred_pcc.iloc[0]) for c in usable]
    rb, rf, bn = [], [], []
    for c in usable:
        rb.append(_chest_pcc(preds[c], z, present, meas[c]))
        rf.append(_chest_pcc(E.baseline(z, st, c), z, present, meas[c])); bn.append(E.baseline_name(c))
    fig_p = F.pcc_by_config(usable, rb, rf, bn, ref_b, ref_f)
    tab = pd.DataFrame({"measured leads": usable, "BiMa-SR chest PCC (this record)": np.round(rb, 3),
                        "baseline chest PCC (this record)": np.round(rf, 3), "baseline": bn,
                        "paper PTB-XL test: BiMa-SR": np.round(ref_b, 3), "paper PTB-XL test: baseline": np.round(ref_f, 3)})
    ms = info["model_ms"]
    md = run_info(info, f"{len(usable)} configurations in one call: " + ", ".join(f"{c} {ms[c]:.0f} ms" for c in usable)
                  + f"; {CALL} {info['call_s']:.2f} s.", t_all)
    if not present.all():
        md += "\n\nNo real reference for some leads: their PCC is not available."
    if len(usable) < len(cfgs):
        md += f"\n\nSkipped (measured leads not in the record): {', '.join(c for c in cfgs if c not in usable)}."
    return fig_sm, fig_p, tab, md


@gpu(DUR["compare"])
def gpu_compare_subset(z, st, cfgs):
    t0 = time.perf_counter()
    res = E.reconstruct_all(z, st, DEVICE, cfgs)
    return {c: p for c, (p, _) in res.items()}, dict(next(iter(res.values()))[1], call_s=time.perf_counter() - t0,
                                                      model_ms={c: i["model_ms"] for c, (_, i) in res.items()})


# ----------------------------------------------------------------------------- Tab 4
def do_uncertainty(source, example, files, fs_csv, n):
    t_all = time.perf_counter()
    if E.METHOD_D1 not in E.available().get("II", []):
        raise E.InputError("DiBiMa-SR weights are not available")
    r = prepare(source, example, files, fs_csv, "II")
    z, st, present = r["z"], r["st"], r["present"]
    n = int(n)
    draws, bima, info = gpu_uncertainty(z, st, n)
    mean = draws.mean(0); lo = draws.min(0); hi = draws.max(0)
    in_idx = [1]
    series = [dict(name="real", data=np.where(present[None, :, None], z, np.nan), color="#000000", width=0.9, role="real"),
              dict(name="BiMa-SR", data=bima, color=F.COL["BiMa-SR"], width=1.0, role="pred"),
              dict(name=f"DiBiMa-SR mean of {n} draws", data=mean, color=F.COL["DiBiMa-SR (one-step)"], width=1.1,
                   role="pred")]
    fig = F.clinical(series, in_idx, f"{r['desc']} — DiBiMa-SR one-step, {n} noise draws (seeds 0-{n - 1}) from "
                                     "lead II (z units)", unit="z", band=dict(lo=lo, hi=hi, name="min-max of draws"))
    pl = [c for c in range(12) if c not in in_idx]
    rows = []
    pm, _, _ = E.T.metrics_np(mean, np.nan_to_num(z)); pb, _, _ = E.T.metrics_np(bima, np.nan_to_num(z))
    for c in pl:
        rows.append({"lead": E.LEADS[c],
                     "PCC of draw mean": round(float(pm[:, c].mean()), 3) if present[c] else None,
                     "spread (mean max-min, z)": round(float((hi[:, c] - lo[:, c]).mean()), 3),
                     "BiMa-SR PCC": round(float(pb[:, c].mean()), 3) if present[c] else None})
    md = run_info(info, f"DiBiMa-SR {n} draws {info['model_ms']:.0f} ms (batched); BiMa-SR {info['bima_ms']:.0f} ms; "
                        f"{CALL} {info['call_s']:.2f} s.", t_all)
    md += ("\n\n*In the paper, BiMa-SR is more accurate than DiBiMa-SR; the band shows sampling variability of the "
           "diffusion model, not calibrated uncertainty.*")
    return fig, pd.DataFrame(rows), md


# ----------------------------------------------------------------------------- Tab 5
def results_tables():
    t1 = pd.read_csv(os.path.join(HERE, "assets", "results", "table1_reconstruction.csv"))
    t2 = pd.read_csv(os.path.join(HERE, "assets", "results", "table2_classifier.csv"))
    lab = {s: l for l, s, _ in E.CFG.CONFIGS}
    coh = {"ptbxl_test": "PTB-XL test", "georgia": "Georgia", "china": "China"}
    base = {"linear_fir": "linear filter", "einthoven_fir": "Einthoven + linear filter"}
    ci = lambda r, p: f"{r[p + '_pcc']:.3f} ({r[p + '_lo']:.3f}-{r[p + '_hi']:.3f})"  # noqa: E731
    rows = []
    for _, r in t1.iterrows():
        rows.append({"measured leads": lab[r.config], "cohort": coh[r.cohort], "baseline": base[r.baseline],
                     "BiMa-SR, all predicted": ci(r, "bima_all_pred"), "baseline, all predicted": ci(r, "base_all_pred"),
                     "BiMa-SR, chest": ci(r, "bima_prec_pred"), "baseline, chest": ci(r, "base_prec_pred")})
    T1 = pd.DataFrame(rows)
    T1["_o"] = T1["measured leads"].map(CFG_ORDER.index); T1["_c"] = T1.cohort.map(list(coh.values()).index)
    T1 = T1.sort_values(["_o", "_c"]).drop(columns=["_o", "_c"])
    names = {"true_12lead": "real 12-lead", "bima_sr": "BiMa-SR", "dibima_sr_onestep": "DiBiMa-SR one-step",
             "einthoven_fir": "Einthoven + linear filter", "linear_fir": "linear filter"}
    T2 = t2.rename(columns=names)
    T2["config"] = T2.config.map(lab); T2["cohort"] = T2.cohort.map(coh)
    T2["group"] = T2.group.map({"superclass": "5 superclasses", "panel": "12-label panel"})
    T2 = T2.rename(columns={"config": "measured leads", "group": "labels"})
    T2["_o"] = T2["measured leads"].map(CFG_ORDER.index); T2["_c"] = T2.cohort.map(list(coh.values()).index)
    T2["_g"] = T2.labels.map({"5 superclasses": 0, "12-label panel": 1})
    T2 = T2.sort_values(["_c", "_g", "_o"]).drop(columns=["_o", "_c", "_g"])
    T2 = T2[["cohort", "labels", "measured leads"] + [v for v in names.values() if v in T2.columns]]
    return T1, T2.round(3)


ABOUT = """
### Method (five sentences)
BiMa-SR is a convolutional encoder - bidirectional Mamba - convolutional decoder network (2.68 M parameters) that maps
the measured leads of a 5-s, 500-Hz window to all twelve leads in one forward pass. DiBiMa-SR uses the same backbone
trained as a 20-step diffusion model with a zero-terminal-SNR cosine schedule and x0-prediction, sampled from pure noise
in one step or with DDIM-6. All ECGs pass one pipeline: first 10 s, zero-phase 0.5-40 Hz band-pass, two 5-s windows,
per-window per-lead z-score. Models were trained on PTB-XL folds 1-8, selected on fold 9, tested on fold 10 and on two
external cohorts never used for training (Georgia, USA; Chapman-Shaoxing/Ningbo, China). Fidelity is the Pearson
correlation and mean absolute error of the predicted leads; diagnostic readability is the AUROC of one 12-lead
classifier trained only on real PTB-XL ECGs and applied unchanged to reconstructed ECGs.

### Limitations
- Evaluation uses z-scored leads per window: absolute amplitudes are not assessed (the mV display here uses a
  training-set amplitude prior).
- Standard electrode positions only; modified Holter, EASI or Frank lead systems were not simulated.
- The external cohorts differ in recording devices and in label definitions.
- No prospective or clinical validation.

### Data and licences
PTB-XL v1.0.3 (Wagner et al., *Sci. Data* 7, 154, 2020), the Georgia 12-lead database (PhysioNet/Computing in
Cardiology Challenge 2020; Perez Alday et al., *Physiol. Meas.* 41, 124003, 2020) and the Chapman-Shaoxing/Ningbo
database (Zheng et al., *Sci. Data* 7, 48, 2020), all from PhysioNet (Goldberger et al., *Circulation* 101, e215, 2000)
and licensed CC BY 4.0. The bundled examples are unmodified copies of eleven of these records.

### Citation and links
Preprint: *to be added*. Paper: *link to be added*. Code: *link to be added when the repository is public*.
Code licence: MIT (c) Ugo Lomoio.
"""


# ----------------------------------------------------------------------------- UI
def build():
    import gradio as gr
    cfgs = available_cfgs()
    av = E.available()
    default = DEFAULT_CFG if DEFAULT_CFG in cfgs else (cfgs[0] if cfgs else None)
    with gr.Blocks(title="BiMa-ECGSR") as demo:
        gr.HTML(BANNER)
        gr.Markdown("# BiMa-ECGSR: 12-lead ECG reconstruction from reduced lead sets\n" + runtime_md())
        if E.manifest().get("error"):
            gr.Markdown(f"**Weights could not be loaded**: {E.manifest()['error']}")
        config = gr.State(default)
        rec = gr.State(None)
        attr = gr.State(None)
        with gr.Row():
            with gr.Column(scale=3):
                source = gr.Radio(["Bundled example", "Upload"], value="Bundled example", label="ECG source")
                example = gr.Dropdown(example_choices(), value=example_choices()[0], label="Example record")
                gr.Markdown(
                    "<small>Examples (fixed rule): within each label group, the record at the 75th percentile of "
                    "BiMa-SR per-record PCC for the Telemetry configuration (I+II+V1); PTB-XL test-fold groups NORM, MI, "
                    "STTC, CD, HYP (single superclass), CLBBB, CRBBB, AFIB, LVH (single panel label), plus one record "
                    "each from Georgia and Chapman-Shaoxing/Ningbo. Data: PTB-XL (Wagner et al. 2020), Georgia "
                    "(PhysioNet/CinC Challenge 2020), Chapman-Shaoxing/Ningbo (Zheng et al. 2020), via PhysioNet; "
                    "all CC BY 4.0.</small>")
                files = gr.File(file_count="multiple", file_types=[".hea", ".dat", ".mat", ".csv", ".npy", ".txt"],
                                label="Upload WFDB (.hea + .dat/.mat), CSV or NPY (leads x samples; 12 leads, or only "
                                      "the measured leads)", visible=False)
                fs_csv = gr.Number(value=500, label="CSV / NPY sampling rate (Hz; resampled to 500 Hz; units mV or µV "
                                                    "detected automatically)", visible=False)
            with gr.Column(scale=2):
                gr.Markdown("**Measured input leads (use case)**")
                btns = []
                with gr.Row():
                    for name, c, use in PRESETS[:3]:
                        btns.append(gr.Button(f"{name}\n{c}", variant="primary" if c == default else "secondary",
                                              interactive=c in av, size="sm"))
                with gr.Row():
                    for name, c, use in PRESETS[3:]:
                        btns.append(gr.Button(f"{name}\n{c}", variant="primary" if c == default else "secondary",
                                              interactive=c in av, size="sm"))
                cfg_md = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("1 · Reconstruct"):
                with gr.Row():
                    method = gr.Radio([E.METHOD_BIMA], value=E.METHOD_BIMA, label="Model (DiBiMa-SR: Monitor / II only)")
                    show_filter = gr.Checkbox(value=False, label="Show linear-filter overlay")
                    run_judge = gr.Checkbox(value=True, label="Run classifier")
                    b_rec = gr.Button("Reconstruct", variant="primary")
                info1 = gr.Markdown()
                plot1 = gr.Plot(label="12-lead ECG")
                cls1 = gr.Markdown()
                table1 = gr.Dataframe(label="Per-lead PCC / MAE (z units); summary rows: pooled median over windows x "
                                            "leads, as in the paper")
                with gr.Row():
                    dl_csv = gr.DownloadButton("Download reconstruction (CSV, mV)", value=None)
                    dl_npy = gr.DownloadButton("Download reconstruction (NPY, mV, 12 x N)", value=None)
            with gr.Tab("2 · Diagnose & explain"):
                gr.Markdown("The paper's 12-lead 1D-ResNet classifier (3.36 M parameters), trained only on real PTB-XL "
                            "ECGs, run unchanged on the real and the reconstructed ECG of the *Reconstruct* tab. "
                            "Explainability is part of this demo only, not of the paper.")
                show_filter2 = gr.Checkbox(value=False, label="Include the linear-filter reconstruction")
                bars = gr.Plot(label="Classifier probabilities")
                cls_md = gr.Markdown()
                with gr.Row():
                    cls_dd = gr.Dropdown(class_choices(), value=class_choices()[0], label="Class to explain (IG)")
                    t_start = gr.Slider(0, 5, value=0, step=0.05, label="Reconstruction attribution: start (s)")
                    t_end = gr.Slider(0, 5, value=5, step=0.05, label="end (s)")
                    b_exp = gr.Button("Compute explanations", variant="primary")
                exp_md = gr.Markdown()
                ig_plot = gr.Plot(label="Integrated Gradients")
                imp_plot = gr.Plot(label="Lead importance")
                lead_dd = gr.Dropdown([], label="Predicted lead (reconstruction attribution)")
                prof_plot = gr.Plot(label="Reconstruction attribution: time profile")
                mat_plot = gr.Plot(label="Lead-to-lead attribution")
            with gr.Tab("3 · Compare input sets"):
                b_cmp = gr.Button("Run all input sets (BiMa-SR)", variant="primary")
                cmp_md = gr.Markdown()
                sm_plot = gr.Plot(label="Chest leads by input set")
                pcc_plot = gr.Plot(label="Chest-lead PCC vs input set")
                cmp_tab = gr.Dataframe(label="Chest-lead PCC")
            with gr.Tab("4 · Diffusion uncertainty (lead II)"):
                gr.Markdown("DiBiMa-SR (one-step sampler) from lead II with N noise draws (fixed seeds 0..N-1); the "
                            "input-lead preset is ignored here (lead II only).")
                with gr.Row():
                    n_draws = gr.Slider(4, 16, value=8, step=1, label="Noise draws N")
                    b_unc = gr.Button("Sample", variant="primary")
                unc_md = gr.Markdown()
                unc_plot = gr.Plot(label="Draws")
                unc_tab = gr.Dataframe(label="Per lead")
            with gr.Tab("5 · Results & About"):
                T1, T2 = results_tables()
                gr.Image(os.path.join(HERE, "assets", "Graphical_Abstract.png"), label="Graphical abstract",
                         show_label=True, buttons=["download", "fullscreen"])
                gr.Markdown("### Table 1 — Reconstruction fidelity (pooled median PCC, 95% record-bootstrap CI; "
                            "generated from the bundled results CSV)")
                gr.Dataframe(T1, wrap=True)
                gr.Markdown("### Table 2 — Diagnostic readability (record-level macro AUROC of the 12-lead classifier)")
                gr.Dataframe(T2, wrap=True)
                gr.Image(os.path.join(HERE, "assets", "Figure1.png"), label="Figure 1 — method", show_label=True,
                         buttons=["download", "fullscreen"])
                gr.Markdown(ABOUT)

        # ---- events
        def _cfg_md(c):
            use = next(u for n, cc, u in PRESETS if cc == c)
            return f"Selected: **{c}** — {use}."

        def _pick(c):
            ms = av.get(c, [])
            upd = [gr.update(variant="primary" if cc == c else "secondary") for _, cc, _ in PRESETS]
            return [c, _cfg_md(c), gr.update(choices=ms, value=E.METHOD_BIMA)] + upd

        for b, (_, c, _) in zip(btns, PRESETS):
            b.click(lambda c=c: _pick(c), None, [config, cfg_md, method] + btns)
        demo.load(lambda: _pick(default), None, [config, cfg_md, method] + btns)

        def _src(s):
            up = s == "Upload"
            return gr.update(visible=not up), gr.update(visible=up), gr.update(visible=up)
        source.change(_src, source, [example, files, fs_csv])

        def _rec(source, example, files, fs, c, m, f, j):
            out = do_reconstruct(source, example, files, fs, c, m, f, j)
            st = out[-1]; st["ui"] = (source, example if source == "Bundled example" else None, c)
            return out[:-1] + (st,)
        b_rec.click(_err(_rec), [source, example, files, fs_csv, config, method, show_filter, run_judge],
                    [plot1, table1, cls1, info1, dl_csv, dl_npy, rec], api_name="reconstruct")
        rec.change(class_view, [rec, show_filter2], [bars, cls_md, cls_dd])
        show_filter2.change(class_view, [rec, show_filter2], [bars, cls_md, cls_dd])
        b_exp.click(_err(do_explain), [rec, cls_dd, t_start, t_end, source, example, files, fs_csv, config, method,
                                       show_filter, lead_dd],
                    [ig_plot, imp_plot, prof_plot, mat_plot, exp_md, lead_dd, attr, rec], api_name="explain")
        lead_dd.change(redraw_profile, [attr, lead_dd], prof_plot)
        b_cmp.click(_err(do_compare), [source, example, files, fs_csv, config], [sm_plot, pcc_plot, cmp_tab, cmp_md],
                    api_name="compare")
        b_unc.click(_err(do_uncertainty), [source, example, files, fs_csv, n_draws], [unc_plot, unc_tab, unc_md],
                    api_name="uncertainty")
    return demo


if __name__ == "__main__":
    build().queue(max_size=16).launch()
