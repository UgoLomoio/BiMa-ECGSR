"""Parameter counts and inference latency of BiMa-SR / DiBiMa-SR (all trained configurations) and the linear filter.

Unit: one 10-s, 500-Hz record = 2 non-overlapping 5-s windows (batch of 2), the model input of the paper.
GPU: mamba_ssm backend, bf16 autocast (as in the study's test pass), 20 warm-up + 100 timed calls, CUDA-synchronised;
     throughput at batch 128 windows. CPU: pure-PyTorch Mamba backend, float32, torch threads 2 and all cores,
     3 warm-up + 10 timed calls. Linear filter: numpy, per predicted lead sum of k 101-tap convolutions (np.convolve),
     timed with coefficient arrays of the fitted shape. Reports medians. Run on an otherwise idle GPU.
Usage: python tools/benchmark_latency.py [--out results/final/efficiency.csv]
"""
import argparse, os, sys, time, json
import numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.training import train_sr as T  # noqa: E402
import torch  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CONFIGS = ["II", "I", "I,II", "I,II,V2", "I,II,V1", "I,II,V1,V4"]
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(CFG.RESULTS_DIR, "final", "efficiency.csv"))
ap.add_argument("--configs", default="+".join(c.replace(",", "_") for c in CONFIGS))
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
cfgs = [c.replace("_", ",") for c in a.configs.split("+")]
NCPU = os.cpu_count()


def timeit(fn, warm, reps, cuda):
    for _ in range(warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def load(mode, cfg, backend, dev):
    in_idx = [LEADS.index(l) for l in cfg.split(",")]; tag = cfg.replace(",", "_")
    run = ("bima_sr_" if mode == "regression" else "dibima_sr_") + tag
    f = os.path.join(CFG.RUNS_DIR, run, "best.pth")
    if not os.path.exists(f):
        return None, None, None
    net = T.build_net(mode, len(in_idx), backend=backend)
    net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
    return T.make_wrapper()(net.to(dev), in_idx, mode).eval(), in_idx, run


rows = []
rng = np.random.default_rng(0)
for cfg in cfgs:
    for mode, samplers in (("regression", {"": None}), ("diffusion", T.SAMPLERS)):
        for sname, steps in samplers.items():
            m, in_idx, run = load(mode, cfg, "mamba_ssm", "cuda")
            if m is None:
                continue
            k = len(in_idx); ddpm = T.build_scheduler(20) if mode == "diffusion" else None
            n_params = sum(p.numel() for p in m.parameters())
            method = "bima_sr" if mode == "regression" else f"dibima_sr_{sname}"
            gen = torch.Generator(device="cuda").manual_seed(0)
            r = dict(config=cfg, method=method, run=run, n_params=n_params, nfe=steps or 1)
            with torch.no_grad():
                for bs, key, warm, reps in ((2, "gpu_ms_per_record", 20, 100), (128, "gpu_ms_per_128_windows", 5, 20)):
                    x = torch.randn(bs, k, 2500, device="cuda")
                    def f():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            T.predict_batch(m, x, ddpm, steps=steps or 1, gen=gen)
                    r[key] = timeit(f, warm, reps, True)
                r["gpu_windows_per_s"] = 128 / (r["gpu_ms_per_128_windows"] / 1e3)
                del m; torch.cuda.empty_cache()
                mc, _, _ = load(mode, cfg, "torch", "cpu"); genc = torch.Generator().manual_seed(0)
                x = torch.randn(2, k, 2500)
                for th in (2, NCPU):
                    torch.set_num_threads(th)
                    r[f"cpu_ms_per_record_threads{th}"] = timeit(
                        lambda: T.predict_batch(mc, x, ddpm, steps=steps or 1, gen=genc), 3, 10, False)
                r["cpu_threads_all"] = NCPU
            rows.append(r); print(json.dumps(r), flush=True)
    # linear filter: per predicted lead, sum over k inputs of a 101-tap FIR (np.convolve, 'same')
    in_idx = [LEADS.index(l) for l in cfg.split(",")]; k = len(in_idx); n_pred = 12 - k
    W = rng.standard_normal((n_pred, k, 101)); x = rng.standard_normal((2, k, 2500))
    def fir():
        for w in range(2):
            for p in range(n_pred):
                sum(np.convolve(x[w, j], W[p, j], mode="same") for j in range(k))
    rows.append(dict(config=cfg, method="linear_fir", run=f"baselines_{cfg.replace(',', '_')}", n_params=int(W.size),
                     nfe=1, cpu_ms_per_record_threads1=timeit(fir, 3, 20, False)))
    print(json.dumps(rows[-1]), flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False); print("wrote", a.out)
