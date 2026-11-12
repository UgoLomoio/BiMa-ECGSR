"""Local check of the Space inference paths against the stored study predictions (no gradio launch).

  BIMA_WEIGHTS_DIR=<staging> BIMA_DEVICE=cuda python space/tools/smoke_devices.py --tag gpu_auto
  BIMA_WEIGHTS_DIR=<staging> BIMA_DEVICE=cuda BIMA_FORCE_BACKEND=torch python space/tools/smoke_devices.py --tag gpu_torch
  BIMA_WEIGHTS_DIR=<staging> BIMA_DEVICE=cpu python space/tools/smoke_devices.py --tag cpu

For every bundled example (all are PTB-XL test-fold records whose two windows are rows of X_ptbxl_test) and every
available configuration / model: (1) the app preprocessing vs the stored dataset windows; (2) the reconstruction vs the
stored predictions pred_ptbxl_test[_<sampler>].npy of the run (float16, produced with bf16 autocast on the GPU; for
DiBiMa-SR the stored noise draw differs, so only the PCC comparison is meaningful); (3) per-window mean PCC over the
predicted leads vs the stored metrics_ptbxl_test[_<sampler>].npz; (4) timing (each model is run twice: the first call
includes the backend self-test / kernel compilation, the second is steady state). Also runs app.run() once per path.
Writes space/tools/smoke_devices_<tag>.csv.
"""
import argparse, os, sys, time
SPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPACE)
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
if a.threads:
    os.environ["NUMBA_NUM_THREADS"] = str(a.threads)
import numpy as np, pandas as pd, torch  # noqa: E402
if a.threads:
    torch.set_num_threads(a.threads)
t_imp = time.perf_counter()
import app as A  # noqa: E402  (preloads the models on BIMA_DEVICE)
t_imp = time.perf_counter() - t_imp
import engine as E  # noqa: E402
from bima_ecgsr import config as CFG  # noqa: E402

X = np.load(os.path.join(CFG.DATA_DIR, "X_ptbxl_test.npy"), mmap_mode="r")
suffix = {E.METHOD_BIMA: "", E.METHOD_D1: "_onestep", E.METHOD_D6: "_ddim6"}
print(f"[{a.tag}] device {A.DEVICE} | {E.device_name(A.DEVICE)} | import+preload {t_imp:.1f} s | {A.PRELOAD}", flush=True)
rows = []
for i, e in enumerate(E.EXAMPLES):
    x, _ = E.read_example(i); pp = E.preprocess(x)
    z, st = pp["z"], pp["stats"]
    dz = float(np.abs(z - np.asarray(X[e["test_windows"]], np.float32)).max())
    for cfg, methods in E.available().items():
        in_idx = [E.LEADS.index(l) for l in E.CONFIG_LEADS[cfg]]; pl = [c for c in range(12) if c not in in_idx]
        fir = E.fir_baseline(z, st, cfg)
        for m in methods:
            run = E._model_entry(cfg, m)["run"]
            ms = []
            for rep in range(2):
                pred, jp, tm = E.infer(z, st, cfg, m, A.DEVICE, fir=fir, want_judge=True, real_ok=True)
                ms.append(tm["model_ms"])
            sp = np.asarray(np.load(os.path.join(CFG.RUNS_DIR, run, f"pred_ptbxl_test{suffix[m]}.npy"), mmap_mode="r")[e["test_windows"]], np.float32)
            ref = np.load(os.path.join(CFG.RUNS_DIR, run, f"metrics_ptbxl_test{suffix[m]}.npz"))["pcc"][e["test_windows"]]
            pcc, _, _ = E.T.metrics_np(pred, z)
            app_w, ref_w = pcc[:, pl].mean(1), ref[:, pl].mean(1)
            d = pred[:, pl] - sp[:, pl]
            pcc_vs_stored = float(np.mean([np.corrcoef(pred[w, c], sp[w, c])[0, 1] for w in range(len(pred)) for c in pl]))
            rows.append(dict(path=a.tag, device_name=tm["device_name"], backend=tm["backend"], selftest=tm["selftest"],
                             ecg_id=e["ecg_id"], config=cfg, method=m, run=run,
                             preprocess_max_abs_diff=dz, max_abs_diff_vs_stored_pred=float(np.abs(d).max()),
                             mean_abs_diff_vs_stored_pred=float(np.abs(d).mean()), pcc_app_vs_stored_pred=pcc_vs_stored,
                             window_pcc_app=np.round(app_w, 4).tolist(), window_pcc_stored=np.round(ref_w, 4).tolist(),
                             max_abs_dpcc_window=float(np.abs(app_w - ref_w).max()),
                             model_ms_first=ms[0], model_ms_steady=ms[1], judge_ms=tm.get("judge_ms"),
                             judge_pred=np.round(jp["pred"], 3).tolist(), judge_real=np.round(jp["real"], 3).tolist()))
            r = rows[-1]
            print(f"  {e['ecg_id']} {cfg:11s} {m:22s} {r['backend']:9s} dPCCwin {r['max_abs_dpcc_window']:.4f} "
                  f"maxdiff {r['max_abs_diff_vs_stored_pred']:.4f} r_vs_stored {pcc_vs_stored:.5f} "
                  f"ms {ms[0]:.1f}/{ms[1]:.1f}", flush=True)
# full app path once per configuration family
app_rows = []
for cfg, m in (("II", E.METHOD_BIMA), ("I+II+V1", E.METHOD_BIMA), ("II", E.METHOD_D6)):
    if cfg in E.available() and m in E.available()[cfg]:
        t0 = time.perf_counter()
        fig, df, jt, info = A.run("Bundled example", A.example_choices()[0], None, "mV", cfg, m, True, True, "mV")
        app_rows.append(dict(config=cfg, method=m, wall_s=time.perf_counter() - t0, info=info.replace("\n\n", " | ")))
        __import__("matplotlib.pyplot", fromlist=["close"]).close(fig)
        print(f"[app.run] {cfg} {m}: {app_rows[-1]['wall_s']:.2f} s\n   {app_rows[-1]['info'][:600]}", flush=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(SPACE, "tools", f"smoke_devices_{a.tag}.csv"), index=False)
pd.DataFrame(app_rows).to_csv(os.path.join(SPACE, "tools", f"smoke_devices_{a.tag}_app.csv"), index=False)
g = out.groupby(["config", "method", "backend"]).agg(n=("ecg_id", "size"), max_dpcc=("max_abs_dpcc_window", "max"),
                                                     max_diff=("max_abs_diff_vs_stored_pred", "max"),
                                                     min_r=("pcc_app_vs_stored_pred", "min"),
                                                     ms_first=("model_ms_first", "median"), ms_steady=("model_ms_steady", "median"))
print(g.to_string()); print(f"preprocess max abs diff over examples: {out.preprocess_max_abs_diff.max():.2e}")
