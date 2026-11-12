"""BiMa-ECGSR Space engine: weights discovery, preprocessing (the study pipeline), GPU/CPU inference, baselines, judge.

Everything model-related comes from the bima_ecgsr package (bundled into the Space by deploy.py; imported from the
repository root when run locally). Checkpoints trained with mamba_ssm load unchanged into either Mamba backend:

  GPU (ZeroGPU or local CUDA): 'mamba_ssm' (CUDA selective-scan kernel + Triton RMSNorm) when it imports AND a
       one-window self-test against the pure-PyTorch backend passes (max |diff| < SELFTEST_TOL); otherwise the
       pure-PyTorch backend (bima_ecgsr.models.mamba_torch) on CUDA.
  CPU: the pure-PyTorch backend with the numba selective-scan kernel.
All inference runs in float32. Diffusion noise is drawn from a seeded CPU generator, so CPU and GPU see the same noise.

Weights: MANIFEST.json and the files it lists are read from the first source that has them:
  $BIMA_WEIGHTS_DIR (local staging directory) -> <space>/weights/ -> private HF model repo $BIMA_WEIGHTS_REPO
  (default Ugo96/BiMa-ECGSR-weights) downloaded with the Space secret $HF_TOKEN into the HF cache (never served).
Input-lead configurations / models are enabled only if their weights are listed in the manifest (runtime discovery).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
try:
    import bima_ecgsr  # noqa: F401  (bundled copy inside the Space)
except ImportError:  # local run from the repository
    sys.path.insert(0, str(HERE.parent))

from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.data import pipeline as P  # noqa: E402
from bima_ecgsr.training import train_sr as T  # noqa: E402

LEADS = list(P.LEADS)
FS, WIN = P.FS, P.WIN
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPER_NAMES = {"NORM": "normal ECG", "MI": "myocardial infarction", "STTC": "ST/T change",
               "CD": "conduction disturbance", "HYP": "hypertrophy"}
CONFIG_LABELS = [c[0] for c in CFG.CONFIGS]                 # II, I, I+II, I+II+V1, I+II+V2, I+II+V1+V4
CONFIG_LEADS = {c[0]: c[1].split(",") for c in CFG.CONFIGS}
CONFIG_NOTES = {"I+II+V1": "5-electrode bedside telemetry"}
WEIGHTS_REPO = os.environ.get("BIMA_WEIGHTS_REPO", "Ugo96/BiMa-ECGSR-weights")
METHOD_BIMA, METHOD_D1, METHOD_D6 = "BiMa-SR", "DiBiMa-SR (one-step)", "DiBiMa-SR (DDIM-6)"
SEED = 0
SELFTEST_TOL = 1e-3
# Reconstruction precision on the GPU: bf16 autocast, the precision of the paper's test-set evaluation. For BiMa-SR
# (deterministic) it matches the stored test predictions only up to bf16 rounding, not exactly: measured on the bundled
# examples, per-sample max |delta| 0 to ~0.05 z units (one bf16 ulp at |x| ~ 4-8), mean |delta| <= ~1e-3, per-lead PCC
# within ~1e-3. DiBiMa-SR does NOT reproduce the stored predictions sample-wise: the app draws its own noise (seeded CPU
# generator), so outputs differ by up to ~1.5 (one-step) / ~3 (DDIM-6) z units and per-lead PCC by up to ~0.07 / ~0.3.
# fp32 is used on the CPU, for the self-test and for all attributions. BIMA_GPU_PRECISION=fp32 overrides.
GPU_PRECISION = os.environ.get("BIMA_GPU_PRECISION", "bf16").lower()
LABELS17 = SUPERCLASSES + ["AFIB", "AFLT", "1AVB", "PAC", "PVC", "SBRAD", "STACH", "WPW", "LVH", "RVH", "CRBBB", "CLBBB"]
LABEL_NAMES = dict(SUPER_NAMES, AFIB="atrial fibrillation", AFLT="atrial flutter", **{"1AVB": "1st-degree AV block"},
                   PAC="premature atrial complexes", PVC="premature ventricular complexes", SBRAD="sinus bradycardia",
                   STACH="sinus tachycardia", WPW="Wolff-Parkinson-White", LVH="left ventricular hypertrophy",
                   RVH="right ventricular hypertrophy", CRBBB="complete right bundle-branch block",
                   CLBBB="complete left bundle-branch block")
CHEST = [6, 7, 8, 9, 10, 11]
EXAMPLES = json.load(open(HERE / "examples" / "examples.json"))


# ----------------------------------------------------------------------------- weights
def _local_dirs():
    return [Path(p) for p in (os.environ.get("BIMA_WEIGHTS_DIR"), HERE / "weights") if p]


def weight_file(rel: str) -> Path:
    for d in _local_dirs():
        if (d / rel).is_file():
            return d / rel
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=WEIGHTS_REPO, filename=rel, repo_type="model",
                                token=os.environ.get("HF_TOKEN") or None))


@lru_cache(maxsize=1)
def manifest() -> dict:
    try:
        m = json.load(open(weight_file("MANIFEST.json")))
    except Exception as exc:  # no weights reachable: the app still starts and says so
        return dict(models=[], fir=[], judge=None, error=f"{type(exc).__name__}: {exc}")
    m["amp"] = json.load(open(weight_file(m["amplitude_prior"])))
    return m


def available():
    """{config label: [method names]} for configurations whose BiMa-SR / DiBiMa-SR weights are in the manifest."""
    out = {}
    for c in CONFIG_LABELS:
        ms = []
        for r in manifest()["models"]:
            if r["config"] != c:
                continue
            ms += [METHOD_BIMA] if r["mode"] == "regression" else [METHOD_D1, METHOD_D6]
        if ms:
            out[c] = sorted(ms, key=[METHOD_BIMA, METHOD_D1, METHOD_D6].index)
    return out


def _model_entry(config, method):
    mode = "regression" if method == METHOD_BIMA else "diffusion"
    for r in manifest()["models"]:
        if r["config"] == config and r["mode"] == mode:
            return r
    raise KeyError(f"no {method} weights for {config}")


def _strict_fp32():
    """True float32 on CUDA: no TF32 in cuDNN convolutions / matmuls (TF32 alone shifts outputs by ~1e-3)."""
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


@lru_cache(maxsize=None)
def load_sr(config, mode, backend="torch", device="cpu"):
    """Build the network with the given Mamba backend, load the checkpoint (strict) and move it to device."""
    import torch
    _strict_fp32()
    r = _model_entry(config, METHOD_BIMA if mode == "regression" else METHOD_D1)
    net = T.build_net(mode, len(r["in_idx"]), backend=backend)
    sd = torch.load(weight_file(r["weights"]), map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=True)
    model = T.make_wrapper()(net, r["in_idx"], mode).eval().to(device)
    model.n_params = sum(p.numel() for p in net.parameters())
    model.backend = net.bottleneck.backend
    return model


@lru_cache(maxsize=8)
def load_fir(config):
    for f in manifest()["fir"]:
        if f["config"] == config:
            d = np.load(weight_file(f["file"]))
            return d["W"], int(d["L"]), [int(i) for i in d["in_idx"]]
    return None


@lru_cache(maxsize=None)
def load_judge(device="cpu"):
    if not manifest().get("judge"):
        return None
    from bima_ecgsr.classification.predict import load_classifier
    model, info = load_classifier(str(weight_file(manifest()["judge"]["file"])), device=device)
    return model


# ----------------------------------------------------------------------------- devices and Mamba backends
def mamba_ssm_importable():
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:300]


def backends_for(device):
    """Candidate Mamba backends for a device, in order of preference ($BIMA_FORCE_BACKEND=torch|mamba_ssm overrides)."""
    forced = os.environ.get("BIMA_FORCE_BACKEND", "").strip()
    if forced:
        return [forced]
    if str(device).startswith("cuda") and mamba_ssm_importable()[0]:
        return ["mamba_ssm", "torch"]
    return ["torch"]


def preload(device):
    """Load every available model (all candidate backends) and the judge onto device. On ZeroGPU this is called at
    start-up outside any GPU function: .to('cuda') is deferred by the spaces package until a GPU is attached."""
    t0 = time.perf_counter(); n = 0
    for c, methods in available().items():
        for mode in sorted({"regression" if m == METHOD_BIMA else "diffusion" for m in methods}):
            for be in backends_for(device):
                try:
                    load_sr(c, mode, be, device); n += 1
                except Exception as exc:  # e.g. mamba_ssm modules that cannot be built: torch backend remains
                    _BACKEND_ERRORS[(c, mode, be)] = f"{type(exc).__name__}: {exc}"[:300]
    load_judge(device)
    return dict(n_models=n, seconds=time.perf_counter() - t0, errors=dict(_BACKEND_ERRORS))


_BACKEND_ERRORS: dict = {}
_SELFTEST: dict = {}          # (config, mode, device) -> dict(backend, diff, note)


def _selftest_input(config, mode, device):
    """One deterministic window (bundled example 0, window 1) in the model's input format, plus fixed noise."""
    import torch
    x, _ = read_example(0)
    pp = preprocess(x)
    in_idx = [LEADS.index(l) for l in CONFIG_LEADS[config]]
    inp = T.norm_inputs_t(torch.from_numpy(pp["z"][:1]).float(), torch.from_numpy(pp["stats"][:1]).float(), in_idx)
    g = torch.Generator().manual_seed(1234)
    xt = torch.randn((1, 12, inp.shape[-1]), generator=g)
    tt = torch.full((1,), 19, dtype=torch.long)
    return inp.to(device), xt.to(device), tt.to(device)


def select_backend(config, mode, device):
    """GPU: mamba_ssm if it imports, its model is built and one window agrees with the torch backend
    (max |diff| < SELFTEST_TOL); else torch. CPU: torch. Cached per process."""
    key = (config, mode, str(device))
    if key in _SELFTEST:
        return _SELFTEST[key]
    import torch
    cands = backends_for(device)
    if os.environ.get("BIMA_FORCE_BACKEND", "").strip():
        res = dict(backend=cands[0], diff=None, note="forced by BIMA_FORCE_BACKEND")
    elif cands == ["torch"]:
        why = "CPU" if not str(device).startswith("cuda") else "mamba_ssm not importable (" + mamba_ssm_importable()[1] + ")"
        res = dict(backend="torch", diff=None, note=why)
    else:
        try:
            _strict_fp32()
            inp, xt, tt = _selftest_input(config, mode, device)
            outs = {}
            with torch.no_grad():
                for be in ("mamba_ssm", "torch"):
                    m = load_sr(config, mode, be, device)
                    o = m(inp) if mode == "regression" else m(inp, xt, tt)
                    outs[be] = o.float()
                torch.cuda.synchronize()
            diff = float((outs["mamba_ssm"] - outs["torch"]).abs().max())
            ok = bool(np.isfinite(diff) and diff < SELFTEST_TOL)
            res = dict(backend="mamba_ssm" if ok else "torch", diff=diff,
                       note=f"self-test max|mamba_ssm - torch| = {diff:.2e} " + ("< " if ok else ">= ") + f"{SELFTEST_TOL:g}")
        except Exception as exc:
            res = dict(backend="torch", diff=None, note=f"mamba_ssm self-test failed ({type(exc).__name__}: {str(exc)[:200]})")
    _SELFTEST[key] = res
    return res


def device_name(device):
    import torch
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    import platform
    return f"CPU ({platform.machine()}, {torch.get_num_threads()} torch threads)"


# ----------------------------------------------------------------------------- input
class InputError(ValueError):
    pass


def _orient(a, n_expected):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise InputError(f"expected a 2-D array (leads x samples), got shape {a.shape}")
    if a.shape[0] in (12, n_expected) and a.shape[1] >= WIN:
        return a
    if a.shape[1] in (12, n_expected) and a.shape[0] >= WIN:
        return a.T
    raise InputError(f"array shape {a.shape}: need 12 x N (or N x 12) with N >= {WIN} samples at {FS} Hz")


def _to12(a, names, config):
    """[k, n] + lead names (or None) -> [12, n] mV with NaN rows for missing leads."""
    x = np.full((12, a.shape[1]), np.nan)
    if names is None:
        names = LEADS if a.shape[0] == 12 else CONFIG_LEADS[config]
        if a.shape[0] != len(names):
            raise InputError(f"{a.shape[0]} unnamed rows: give all 12 leads or exactly the {len(CONFIG_LEADS[config])} "
                             f"input leads of {config} ({', '.join(CONFIG_LEADS[config])}) in that order")
    up = [n.strip().upper() for n in names]
    for i, lead in enumerate(LEADS):
        if lead.upper() in up:
            x[i] = a[up.index(lead.upper())]
    return x


FS_RANGE = (100.0, 2000.0)


def _auto_units(a):
    """mV unless the amplitudes are implausibly large for mV (robust peak > 30) -> microvolts. Returns (a_mV, note)."""
    v = a[np.isfinite(a).all(1)] if a.ndim == 2 else a
    peak = float(np.nanmedian(np.nanpercentile(np.abs(v - np.nanmedian(v, axis=-1, keepdims=True)), 99, axis=-1)))
    if peak > 30:
        return a * 1e-3, f"amplitudes interpreted as µV (median 99th-percentile |x| = {peak:.0f}) and converted to mV"
    return a, "amplitudes interpreted as mV"


def _resample(a, fs):
    if not (FS_RANGE[0] <= fs <= FS_RANGE[1]):
        raise InputError(f"sampling rate {fs:g} Hz is outside the supported range {FS_RANGE[0]:g}-{FS_RANGE[1]:g} Hz")
    if np.isclose(fs, FS):
        return a, None
    import scipy.signal as ss
    up, down = P._rational(FS, float(fs))
    return ss.resample_poly(a, up, down, axis=-1), f"resampled from {fs:g} Hz to {FS} Hz"


def _orient_any(a, n_expected, fs):
    """2-D array -> [leads, samples]; leads = 12 or the configuration's input count."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise InputError(f"expected a 2-D array (leads x samples), got shape {a.shape}")
    need = int(np.ceil(WIN * fs / FS))
    if a.shape[0] in (12, n_expected) and a.shape[1] >= need:
        return a
    if a.shape[1] in (12, n_expected) and a.shape[0] >= need:
        return a.T
    raise InputError(f"array shape {a.shape}: need 12 x N or {n_expected} x N (or transposed) with N >= {need} samples "
                     f"(5 s at {fs:g} Hz)")


def read_upload(files, config, fs_csv=500.0):
    """files: list of paths (.hea + .dat/.mat, or one .csv / .npy). CSV/NPY: sampling rate fs_csv, units auto-detected
    (mV or µV). Returns (x [12, n] mV at 500 Hz with NaN rows for leads not provided, description, notes)."""
    files = [Path(f) for f in (files or [])]
    if not files:
        raise InputError("no file uploaded")
    ext = {f.suffix.lower() for f in files}
    notes = []
    if ".hea" in ext:
        hea = [f for f in files if f.suffix.lower() == ".hea"][0]
        tmp = Path(tempfile.mkdtemp(prefix="bima_wfdb_"))
        for f in files:  # gradio stores each upload in its own folder: put header and signal side by side
            shutil.copy2(f, tmp / f.name)
        rec = tmp / hea.stem
        import wfdb
        try:
            sig, fields = wfdb.rdsamp(str(rec))
        except Exception as exc:
            raise InputError(f"could not read the WFDB record ({type(exc).__name__}); upload the .hea together with its "
                             f"signal file (.dat or .mat)") from None
        names = [str(s) for s in fields["sig_name"]]
        units_ = [str(fields["units"][i]).strip().lower() for i in range(len(names))]
        scale = np.array([P._UNIT_SCALE.get(u, np.nan) for u in units_])[:, None]
        if np.isnan(scale).any():
            raise InputError(f"unknown signal units {sorted(set(units_))} in the WFDB header")
        a = np.asarray(sig, np.float64).T * scale
        a, n = _resample(a, float(fields["fs"]))
        notes += [n] if n else []
        x = _to12(a, names, config)
        if x.shape[1] < WIN:
            raise InputError(f"record too short: {x.shape[1] / FS:.1f} s at {FS} Hz (need >= 5 s)")
        return x, f"WFDB {hea.stem}: leads {', '.join(names)}, {float(fields['fs']):g} Hz", notes
    if len(files) > 1:
        raise InputError("upload one CSV or NPY file (or a WFDB .hea with its signal file)")
    f = files[0]
    fs = float(fs_csv or FS)
    if f.suffix.lower() == ".npy":
        try:
            a = np.load(f, allow_pickle=False)
        except Exception as exc:
            raise InputError(f"could not read the NPY file ({type(exc).__name__})") from None
        names = None
    elif f.suffix.lower() in (".csv", ".txt"):
        import pandas as pd
        df = pd.read_csv(f, header=None)
        names = None
        if df.iloc[0].astype(str).str.strip().str.upper().isin([l.upper() for l in LEADS]).any():  # header row
            names = df.iloc[0].astype(str).str.strip().tolist(); df = df.iloc[1:]
            keep = [i for i, n in enumerate(names) if n.upper() in [l.upper() for l in LEADS]]   # drop e.g. time column
            names = [names[i] for i in keep]; df = df.iloc[:, keep]
        try:
            a = df.to_numpy(dtype=np.float64)
        except ValueError:
            raise InputError("the CSV contains non-numeric values") from None
        if names is not None:
            a = a.T
    else:
        raise InputError(f"unsupported file type {f.suffix}; use WFDB (.hea + .dat/.mat), .csv or .npy")
    if names is None:
        a = _orient_any(a, len(CONFIG_LEADS[config]), fs)
    elif a.shape[1] < int(np.ceil(WIN * fs / FS)):
        raise InputError(f"{a.shape[1]} samples: need >= 5 s at {fs:g} Hz")
    a, n_units = _auto_units(a); notes.append(n_units)
    a, n_rs = _resample(a, fs)
    notes += [n_rs] if n_rs else []
    kind = f.suffix.lower().lstrip(".").upper()
    return _to12(a, names, config), f"{kind} {f.name}: {a.shape[0]} leads x {a.shape[1]} samples", notes


def example_label(e):
    where = "PTB-XL test fold" if e["cohort"] == "ptbxl_test" else e["cohort_name"]
    return f"{e['group']} - {where} {e['record_id']}: {e['diagnosis']}"


def read_example(idx):
    e = EXAMPLES[idx]
    x, info, _ = P.read_record(HERE / "examples" / e["file"])
    return x, f"{e['cohort_name']} record {e['record_id']} ({e['diagnosis']})"


def preprocess(x):
    """Study pipeline: first 10 s, zero-phase Butterworth 0.5-40 Hz (order 3), two 5-s windows, per-window per-lead
    z-score. Returns dict(z [W,12,2500] float32 with NaN rows for missing leads, stats [W,12,2] (mean_mV, std_mV),
    present [12] bool, notes)."""
    present = np.isfinite(x).all(1)
    notes = []
    n = x.shape[1]
    if present.all() and n >= P.N_SAMPLES:
        r = P.process_signal(x)                                    # identical to the dataset build
        keep = [k for k, rs in enumerate(r["reasons"]) if rs is None]
        for k, rs in enumerate(r["reasons"]):
            if rs is not None:
                notes.append(f"window {k + 1} excluded by the pipeline QC ({rs})")
        if not keep:
            raise InputError("both 5-s windows fail the pipeline QC (" + ", ".join(map(str, r["reasons"])) + ")")
        z = P.to_store(r["windows"][keep]).astype(np.float32)      # float16 storage as in the dataset
        return dict(z=z, stats=r["stats"][keep], present=present, notes=notes, windows=keep)
    if n >= P.N_SAMPLES:
        seg, n_win = x[:, :P.N_SAMPLES], 2
    else:
        seg, n_win = x[:, :WIN], 1
        notes.append("only 5 s available: band-pass applied to the 5-s segment (the study filters 10 s, then splits)")
    if n > P.N_SAMPLES:
        notes.append("first 10 s used")
    filt = np.full(seg.shape, np.nan)
    filt[present] = P.bandpass(seg[present])
    zs, sts, keep = [], [], []
    for k in range(n_win):
        w = filt[:, k * WIN:(k + 1) * WIN]
        with warnings.catch_warnings():          # all-NaN rows = leads that were not provided
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, std = np.nanmean(w, -1), np.nanstd(w, -1)
        if (std[present] < P.FLAT_STD_MV).any():
            notes.append(f"window {k + 1} excluded: flat lead (std < {P.FLAT_STD_MV} mV)"); continue
        zs.append(((w - mean[:, None]) / std[:, None]).astype(np.float16).astype(np.float32))
        sts.append(P.window_stats(mean, std)); keep.append(k)
    if not zs:
        raise InputError("no usable 5-s window")
    return dict(z=np.stack(zs), stats=np.stack(sts), present=present, notes=notes, windows=keep)


# ----------------------------------------------------------------------------- inference
def _sync(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def reconstruct(z, stats, config, method, device="cpu", backend=None):
    """z [W,12,L] (NaN rows allowed outside the inputs), stats [W,12,2] -> (pred z [W,12,L], info).
    info: device, device_name, backend, selftest note/diff, model_ms (synchronised, after the backend self-test)."""
    import torch
    _strict_fp32()                                                 # (again inside the ZeroGPU worker process)
    if str(device).startswith("cuda") and "BIMA_SCAN_CHUNK" not in os.environ:
        from bima_ecgsr.models import mamba_torch
        mamba_torch.SCAN_CHUNK = 256          # torch backend on CUDA: ~2x faster than 64 (measured), same result
    in_idx = [LEADS.index(l) for l in CONFIG_LEADS[config]]
    mode = "regression" if method == METHOD_BIMA else "diffusion"
    sel = select_backend(config, mode, device) if backend is None else dict(backend=backend, diff=None, note="forced")
    model = load_sr(config, mode, sel["backend"], device)
    zz = torch.from_numpy(np.nan_to_num(z)).float(); st = torch.from_numpy(np.nan_to_num(stats, nan=1.0)).float()
    inp = T.norm_inputs_t(zz, st, in_idx).to(device)
    gen = torch.Generator().manual_seed(SEED)                     # CPU generator: identical noise on CPU and GPU
    amp = str(device).startswith("cuda") and GPU_PRECISION == "bf16"
    _sync(device); t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        if mode == "regression":
            p = model(inp)
        else:
            from diffusers import DDIMScheduler
            ddim = DDIMScheduler.from_config(T.build_scheduler(20).config)
            ddim.set_timesteps(T.SAMPLERS["onestep" if method == METHOD_D1 else "ddim6"], device=inp.device)
            x = torch.randn((inp.shape[0], 12, inp.shape[-1]), generator=gen).to(device)
            for t in ddim.timesteps:
                tb = torch.full((inp.shape[0],), int(t), device=inp.device, dtype=torch.long)
                x0 = model(inp, x, tb).float()
                x = ddim.step(x0, t, x, eta=0.0).prev_sample
            p = x
    _sync(device); sec = time.perf_counter() - t0
    p = p.float().cpu().numpy()
    p[:, in_idx] = z[:, in_idx]                                    # measured inputs are reported as measured
    info = dict(device=str(device), device_name=device_name(device), backend=sel["backend"], selftest=sel["note"],
                selftest_diff=sel["diff"], model_ms=sec * 1000, n_windows=int(z.shape[0]),
                precision="bf16 autocast (as in the paper's evaluation)" if amp else "float32")
    return p, info


def judge_probs(X12, device="cpu"):
    m = load_judge(device)
    if m is None:
        return None
    from bima_ecgsr.classification.predict import predict
    p = predict(m, np.nan_to_num(X12), batch_size=8, device=device, amp=False)
    return p.mean(0)                 # record probability = mean over the record's windows (17 labels, LABELS17 order)


def infer(z, stats, config, method, device="cpu", fir=None, want_judge=True, real_ok=False, backend=None):
    """Everything that touches the network(s), in one call (= one ZeroGPU allocation): reconstruction and, optionally,
    the 12-lead judge on the real ECG, the reconstruction and the FIR baseline. Returns numpy arrays + timing info."""
    t0 = time.perf_counter()
    pred, info = reconstruct(z, stats, config, method, device, backend)
    judge = None
    if want_judge and load_judge(device) is not None:
        tj = time.perf_counter()
        judge = dict(pred=judge_probs(pred, device))
        if real_ok:
            judge["real"] = judge_probs(z, device)
        if fir is not None:
            judge["fir"] = judge_probs(fir, device)
        _sync(device); info["judge_ms"] = (time.perf_counter() - tj) * 1000
    info["call_s"] = time.perf_counter() - t0
    return pred, judge, info


def baseline_name(config):
    leads = CONFIG_LEADS[config]
    return "Einthoven + linear filter" if ("I" in leads and "II" in leads) else "linear filter"


def baseline(z, stats, config):
    """The paper's strongest linear baseline: linear filter (multi-input FIR); when I and II are measured, the limb
    leads III, aVR, aVL, aVF are derived exactly (Einthoven/Goldberger) and the chest leads come from the filter."""
    p = fir_baseline(z, stats, config)
    if p is None:
        return None
    leads = CONFIG_LEADS[config]
    if "I" in leads and "II" in leads:
        p = p.copy()
        p[:, 2:6] = T.einthoven_np(np.nan_to_num(z).astype(np.float64), np.nan_to_num(stats, nan=1.0).astype(np.float64))
    return p.astype(np.float32)


def fir_baseline(z, stats, config):
    f = load_fir(config)
    if f is None:
        return None
    W, L, in_idx = f
    A = T._lag(T.norm_inputs_np(np.nan_to_num(z), np.nan_to_num(stats, nan=1.0), in_idx), L)
    Pz = (A.reshape(-1, A.shape[-1]) @ W).reshape(z.shape[0], -1, 12).transpose(0, 2, 1)
    Pz[:, in_idx] = z[:, in_idx]
    return Pz.astype(np.float32)


def to_mv(zp, stats, config, true_stats_ok):
    """Display amplitudes in mV. Input leads (and any lead whose real signal is present): z*std+mean of the measured
    signal. Predicted leads: z_pred * ratio_c * s, with s the common input scale of the window and ratio_c the median
    std_c / s over the PTB-XL training windows (amplitude prior); zero mean (band-passed signals)."""
    in_idx = [LEADS.index(l) for l in CONFIG_LEADS[config]]
    ratio = np.asarray(manifest()["amp"][config]["ratio"])
    s = np.sqrt((stats[:, in_idx, 1].astype(np.float64) ** 2).mean(1))
    out = zp * ratio[None, :, None] * s[:, None, None]
    out[:, in_idx] = zp[:, in_idx] * stats[:, in_idx, 1:2] + stats[:, in_idx, 0:1]
    return out.astype(np.float32)


def reconstruct_all(z, stats, device="cpu", configs=None):
    """BiMa-SR for every available configuration (Tab 3). Returns {config: (pred, info)}."""
    out = {}
    for c in (configs or [c for c, ms in available().items() if METHOD_BIMA in ms]):
        out[c] = reconstruct(z, stats, c, METHOD_BIMA, device)
    return out


def diffusion_draws(z, stats, n, device="cpu", config="II"):
    """DiBiMa-SR one-step with noise seeds 0..n-1 (CPU generators). Returns draws [n, W, 12, L], info."""
    import torch
    _strict_fp32()
    in_idx = [LEADS.index(l) for l in CONFIG_LEADS[config]]
    sel = select_backend(config, "diffusion", device)
    model = load_sr(config, "diffusion", sel["backend"], device)
    zz = torch.from_numpy(np.nan_to_num(z)).float(); st = torch.from_numpy(np.nan_to_num(stats, nan=1.0)).float()
    inp = T.norm_inputs_t(zz, st, in_idx).to(device)
    W, L = inp.shape[0], inp.shape[-1]
    noise = torch.cat([torch.randn((W, 12, L), generator=torch.Generator().manual_seed(k)) for k in range(n)]).to(device)
    tb = torch.full((n * W,), 19, device=inp.device, dtype=torch.long)
    amp = str(device).startswith("cuda") and GPU_PRECISION == "bf16"
    _sync(device); t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        from diffusers import DDIMScheduler
        ddim = DDIMScheduler.from_config(T.build_scheduler(20).config); ddim.set_timesteps(1, device=inp.device)
        x0 = model(inp.repeat(n, 1, 1), noise, tb).float()
        d = ddim.step(x0, ddim.timesteps[0], noise, eta=0.0).prev_sample
    _sync(device); sec = time.perf_counter() - t0
    d = d.float().cpu().numpy().reshape(n, W, 12, L)
    d[:, :, in_idx] = z[None, :, in_idx]
    return d, dict(device=str(device), device_name=device_name(device), backend=sel["backend"], selftest=sel["note"],
                   model_ms=sec * 1000, n_draws=n, seeds=list(range(n)),
                   precision="bf16 autocast" if amp else "float32")


def explain(z, pred, stats, config, method, cls, t0, t1, device="cpu", real_ok=True, leads=None):
    """All attributions in one call (fp32): IG on the classifier for the real (if available) and the reconstructed ECG,
    and gradient x input of every predicted lead (or only `leads`) w.r.t. the measured inputs over [t0, t1) samples of
    each window."""
    import torch
    import explain as X
    _strict_fp32()
    info = dict(device=str(device), device_name=device_name(device))
    clf = load_judge(device)
    tt = time.perf_counter()
    ig = {}
    if clf is not None:
        if real_ok:
            ig["real"] = X.integrated_gradients_adaptive(clf, z, cls, device)
        ig["recon"] = X.integrated_gradients_adaptive(clf, pred, cls, device)
    _sync(device); info["ig_ms"] = (time.perf_counter() - tt) * 1000
    in_idx = [LEADS.index(l) for l in CONFIG_LEADS[config]]
    pl = [c for c in range(12) if c not in in_idx]
    if leads:
        pl = [c for c in pl if c in leads]
    mode = "regression" if method == METHOD_BIMA else "diffusion"
    sel = select_backend(config, mode, device)
    zz = torch.from_numpy(np.nan_to_num(z)).float(); st = torch.from_numpy(np.nan_to_num(stats, nan=1.0)).float()
    inp = T.norm_inputs_t(zz, st, in_idx).to(device)

    def run(backend):
        model = load_sr(config, mode, backend, device)
        sampler = None
        if mode == "diffusion":
            from diffusers import DDIMScheduler
            ddim = DDIMScheduler.from_config(T.build_scheduler(20).config)
            ddim.set_timesteps(T.SAMPLERS["onestep" if method == METHOD_D1 else "ddim6"], device=inp.device)
            noise0 = torch.randn((inp.shape[0], 12, inp.shape[-1]), generator=torch.Generator().manual_seed(SEED)).to(device)

            def sampler(x_in):
                reps = x_in.shape[0] // inp.shape[0]
                x = noise0.repeat(reps, 1, 1)
                for t in ddim.timesteps:
                    tb = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
                    x = ddim.step(model(x_in, x, tb).float(), t, x, eta=0.0).prev_sample
                return x
        chunk = 4 if str(device).startswith("cuda") else 2
        if backend == "torch":
            with X.torch_backend_autograd(model):
                return X.recon_grad_x_input(model, inp, pl, t0, t1, sampler=sampler, chunk=chunk)
        return X.recon_grad_x_input(model, inp, pl, t0, t1, sampler=sampler, chunk=chunk)

    tr = time.perf_counter()
    be, note = sel["backend"], ""
    try:
        gxi, ok = run(be)
        if not ok:
            raise FloatingPointError("non-finite gradients")
    except Exception as exc:
        if be == "torch":
            raise
        note = f"mamba_ssm backward unavailable ({type(exc).__name__}: {str(exc)[:120]}); PyTorch backend used"
        be = "torch"; gxi, ok = run(be)
    _sync(device); info["recon_attr_ms"] = (time.perf_counter() - tr) * 1000
    info.update(attr_backend=be, attr_note=note, pred_leads=pl, in_idx=in_idx)
    return ig, gxi, info


def lead_metrics(pred, true, present, config):
    """Per-lead PCC / MAE (z units) over the windows, for predicted leads whose real signal is present."""
    in_idx = [LEADS.index(l) for l in CONFIG_LEADS[config]]
    rows = []
    pcc, mae, _ = T.metrics_np(np.nan_to_num(pred), np.nan_to_num(true))
    for c, lead in enumerate(LEADS):
        role = "input" if c in in_idx else "predicted"
        ok = present[c] and role == "predicted"
        rows.append(dict(lead=lead, role=role, pcc=float(pcc[:, c].mean()) if ok else np.nan,
                         mae=float(mae[:, c].mean()) if ok else np.nan))
    return rows, pcc, mae
