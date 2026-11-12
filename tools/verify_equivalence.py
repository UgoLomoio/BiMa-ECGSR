"""Numerical equivalence of this repository's code with the stored study outputs (read-only use of the run dirs).

1. BiMa-SR II and DiBiMa-SR II (one-step, DDIM-6): reload best.pth with the new code (mamba_ssm backend, GPU, bf16
   autocast, eval batch 128 and generator seed 0 exactly as the training script's test pass), reconstruct the first
   N PTB-XL test windows and compare with the stored pred_ptbxl_test*.npy (float16) and metrics_*.npz.
2. Pure-PyTorch Mamba backend on CPU (float32) for the first N_CPU windows vs the stored predictions.
3. The 12-lead judge: predict on the first N true PTB-XL test windows and compare window and record probabilities with
   judge12/pred_true_ptbxl_test.npz.
Usage: python tools/verify_equivalence.py [--n 64] [--n-cpu 4] [--out docs/verification/equivalence.json]
Requires > MIN_FREE_MIB free GPU memory (default 4000); otherwise everything runs on the CPU (torch backend).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.training import train_sr as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--n-cpu", type=int, default=4)
ap.add_argument("--min-free-mib", type=int, default=4000)
ap.add_argument("--warmup", action="store_true",
                help="run one batch-64 pass of each GPU model first (reproduces the kernel-autotune history of the training\n"
                     "process, where validation at batch 64 preceded the batch-128 test pass)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
a.out = a.out or os.path.join(REPO, "docs", "verification", "equivalence%s.json" % ("_warmup" if a.warmup else ""))
import torch  # noqa: E402


def free_mib():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return 0


fm = free_mib(); use_gpu = torch.cuda.is_available() and fm > a.min_free_mib
print(f"[gpu] free {fm} MiB -> {'GPU' if use_gpu else 'CPU only'}", flush=True)
X, S = T.load_cohort("ptbxl_test")
EB = 128                                  # eval batch of the study queue (--eval-bs 128)
nb = max(EB, a.n)                         # one full first batch so the noise draw equals the stored run's
res = dict(n_windows_compared=a.n, eval_batch=EB, gpu_free_mib=fm)


def stored(run, key):
    p = np.load(os.path.join(CFG.RUNS_DIR, run, f"pred_{key}.npy"), mmap_mode="r")
    m = np.load(os.path.join(CFG.RUNS_DIR, run, f"metrics_{key}.npz"))
    return np.asarray(p[:nb], np.float32), {k: m[k][:nb] for k in ("pcc", "mae", "rmse")}


def load(mode, backend, dev):
    Wr = T.make_wrapper()
    net = T.build_net(mode, 1, backend=backend)
    sd = torch.load(os.path.join(CFG.RUNS_DIR, ("bima_sr_II" if mode == "regression" else "dibima_sr_II"), "best.pth"),
                    map_location="cpu", weights_only=True)
    missing = net.load_state_dict(sd, strict=True)
    return Wr(net.to(dev), [1], mode).eval()


def cmp(pred, ref, M, Mref, n):
    d = np.abs(pred[:n] - ref[:n]); pred_leads = [c for c in range(12) if c != 1]
    return dict(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                max_abs_diff_predicted_leads=float(d[:, pred_leads].max()),
                frac_exact_float16=float((pred[:n].astype(np.float16) == ref[:n].astype(np.float16)).mean()),
                max_abs_diff_pcc=float(np.abs(M[0][:n] - Mref["pcc"][:n]).max()),
                max_abs_diff_mae=float(np.abs(M[1][:n] - Mref["mae"][:n]).max()))


tmp = os.path.join("/tmp", f"bima_eq_{os.getpid()}.npy")
if use_gpu:
    for mode, run, samplers in (("regression", "bima_sr_II", {"": None}), ("diffusion", "dibima_sr_II", T.SAMPLERS)):
        model = load(mode, "mamba_ssm", "cuda"); ddpm = T.build_scheduler(20) if mode == "diffusion" else None
        if a.warmup:
            T.predict_array(model, X, S, [1], 64, ddpm, 1, seed=5, amp=True, n=64)
        for sname, steps in samplers.items():
            key = f"ptbxl_test_{sname}" if sname else "ptbxl_test"
            torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            M = T.predict_array(model, X, S, [1], EB, ddpm, steps, seed=0, amp=True, n=nb, out_path=tmp)
            P = np.asarray(np.load(tmp), np.float32); os.remove(tmp)
            ref, Mref = stored(run, key)
            r = cmp(P, ref, M, Mref, a.n); r.update(sec=time.time() - t0, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            r["max_abs_diff_all_first_batch"] = float(np.abs(P - ref).max())
            res[f"{run}:{sname or 'regression'}:gpu_mamba_ssm" + (":warmup" if a.warmup else "")] = r
            print(run, sname or "regression", json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
        del model; torch.cuda.empty_cache()

# pure-PyTorch backend on CPU (float32, no autocast)
torch.set_num_threads(min(8, os.cpu_count()))
n_cpu = a.n_cpu
z = torch.from_numpy(np.asarray(X[:n_cpu], np.float32)); st = torch.from_numpy(S[:n_cpu])
inp = T.norm_inputs_t(z, st, [1])
for mode, run, samplers in (("regression", "bima_sr_II", {"": None}), ("diffusion", "dibima_sr_II", {"onestep": 1})):
    model = load(mode, "torch", "cpu"); ddpm = T.build_scheduler(20) if mode == "diffusion" else None
    for sname, steps in samplers.items():
        key = f"ptbxl_test_{sname}" if sname else "ptbxl_test"
        ref, Mref = stored(run, key)
        if mode == "diffusion":   # reuse the exact stored-run noise: first CUDA draw of generator seed 0, batch 128
            if use_gpu:
                g = torch.Generator(device="cuda").manual_seed(0)
                noise = torch.randn((EB, 12, X.shape[-1]), device="cuda", generator=g)[:n_cpu].cpu()
            else:
                noise = None
        t0 = time.time()
        with torch.no_grad():
            if mode == "regression":
                p = model(inp)
            elif noise is not None:
                tb = torch.full((n_cpu,), 19, dtype=torch.long)
                p = model(inp, noise, tb)          # one-step = x0 prediction at t = T-1 from pure noise
            else:
                p = None
        sec = time.time() - t0
        if p is None:
            res[f"{run}:{sname}:cpu_torch"] = "skipped (needs the CUDA noise draw)"; continue
        p = p.float(); p[:, [1]] = z[:, [1]]
        P = p.numpy(); M = T.metrics_np(P, z.numpy())
        r = cmp(P, ref, M, Mref, n_cpu); r.update(sec=sec, n=n_cpu, ms_per_window=sec / n_cpu * 1000)
        res[f"{run}:{sname or 'regression'}:cpu_torch"] = r
        print(run, sname or "regression", "cpu", json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}), flush=True)

# judge
from bima_ecgsr.classification.predict import load_classifier, predict, aggregate_records  # noqa: E402
import pandas as pd  # noqa: E402
dev = "cuda" if use_gpu else "cpu"
judge, info = load_classifier(CFG.judge_ckpt(), device=dev)
tj = np.load(os.path.join(CFG.CLS_DIR, "judge12", "pred_true_ptbxl_test.npz"), allow_pickle=True)
meta = pd.read_csv(os.path.join(CFG.DATA_DIR, "meta_ptbxl_test.csv"), usecols=["cohort", "record_id"]).iloc[:a.n].copy()
meta["cohort"] = "ptbxl_test"
for bs in (64, 256):
    nn_ = max(a.n, bs) if bs == 256 else a.n
    pw = predict(judge, X[:nn_], batch_size=bs)[:a.n]
    dw = np.abs(pw - tj["p_win"][:a.n].astype(np.float32))
    keys, pr, codes = aggregate_records(pw, meta)
    cnt = np.bincount(codes); full = cnt == 2                  # records whose both windows are inside the first n
    rid = tj["record_id"].astype(str); pos = {r: i for i, r in enumerate(rid)}
    idx = np.array([pos[str(r)] for r in keys.record_id])
    dr = np.abs(pr[full] - tj["p_rec"][idx[full]])
    res[f"judge12:batch{bs}:{dev}"] = dict(max_abs_diff_window_prob_vs_stored_float16=float(dw.max()),
                                           max_abs_diff_record_prob=float(dr.max()), n_records_complete=int(full.sum()),
                                           note="stored p_win is float16; p_rec float32 (bf16 autocast on GPU)")
    print("judge", bs, res[f"judge12:batch{bs}:{dev}"], flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print("[saved]", a.out)
