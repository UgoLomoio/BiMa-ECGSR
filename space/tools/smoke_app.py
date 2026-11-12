"""Local launch-free check of the Space: import app, run the inference function on every bundled example for every
available configuration / model on the CPU, and compare with the stored study values.

  BIMA_WEIGHTS_DIR=/media/gtimpano/Data/UGO/bima_ecgsr_hf_weights_staging python space/tools/smoke_app.py [--threads N]

Checks per example: (1) the app preprocessing reproduces the stored dataset windows (X_ptbxl_test rows, float16);
(2) per-window mean PCC over the predicted leads vs the stored metrics_ptbxl_test[_<sampler>].npz of the run;
(3) latency. Writes space/tools/smoke_results.csv. Needs $BIMA_DATA_DIR / $BIMA_RUNS_DIR for (1)-(2) only.
"""
import argparse, os, sys, time
SPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPACE)
ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=0); ap.add_argument("--examples", default="all")
a = ap.parse_args()
if a.threads:
    os.environ["NUMBA_NUM_THREADS"] = str(a.threads)
import numpy as np, pandas as pd, torch  # noqa: E402
if a.threads:
    torch.set_num_threads(a.threads)
import app as A  # noqa: E402
import engine as E  # noqa: E402
from bima_ecgsr import config as CFG  # noqa: E402

X = np.load(os.path.join(CFG.DATA_DIR, "X_ptbxl_test.npy"), mmap_mode="r")
stored = {E.METHOD_BIMA: "metrics_ptbxl_test.npz", E.METHOD_D1: "metrics_ptbxl_test_onestep.npz", E.METHOD_D6: "metrics_ptbxl_test_ddim6.npz"}
rows = []
ex = range(len(E.EXAMPLES)) if a.examples == "all" else [int(i) for i in a.examples.split(",")]
print("available:", E.available(), "| torch threads", torch.get_num_threads(), flush=True)
for i in ex:
    e = E.EXAMPLES[i]
    x, _ = E.read_example(i); pp = E.preprocess(x)
    dz = float(np.abs(pp["z"] - np.asarray(X[e["test_windows"]], np.float32)).max())
    for cfg, methods in E.available().items():
        in_idx = [E.LEADS.index(l) for l in E.CONFIG_LEADS[cfg]]; pred_leads = [c for c in range(12) if c not in in_idx]
        for m in methods:
            t0 = time.perf_counter()
            fig, df, jt, info = A.run("Bundled example", A.example_choices()[i], None, "mV", cfg, m, True, True, "mV")
            wall = time.perf_counter() - t0
            pred, sec = E.reconstruct(pp["z"], pp["stats"], cfg, m)
            pcc, _, _ = E.T.metrics_np(pred, pp["z"])
            run = E._model_entry(cfg, m)["run"]
            ref = np.load(os.path.join(CFG.RUNS_DIR, run, stored[m]))["pcc"][e["test_windows"]]
            app_pcc = pcc[:, pred_leads].mean(1); ref_pcc = ref[:, pred_leads].mean(1)
            rows.append(dict(ecg_id=e["ecg_id"], superclass=e["superclass"], config=cfg, method=m,
                             preprocess_max_abs_diff_vs_dataset=dz, window_pcc_app=np.round(app_pcc, 4).tolist(),
                             window_pcc_stored=np.round(ref_pcc, 4).tolist(),
                             max_abs_dpcc_window=float(np.abs(app_pcc - ref_pcc).max()), model_sec=round(sec, 2),
                             request_wall_sec=round(wall, 2), judge_real=None if jt is None or "real 12-lead ECG" not in jt else jt["real 12-lead ECG"].tolist(),
                             judge_recon=None if jt is None else jt.iloc[:, 2].tolist()))
            plt_close = __import__("matplotlib.pyplot", fromlist=["close"]).close(fig)
            print(rows[-1], flush=True)
out = pd.DataFrame(rows)
suffix = f"_threads{a.threads}" if a.threads else ""
out.to_csv(os.path.join(SPACE, "tools", f"smoke_results{suffix}.csv"), index=False)
print(out[["ecg_id", "config", "method", "preprocess_max_abs_diff_vs_dataset", "max_abs_dpcc_window", "model_sec", "request_wall_sec"]].to_string(index=False))
