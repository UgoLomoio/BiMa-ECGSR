"""Reconstruction-fidelity statistics for one input configuration.
Usage: python -m bima_ecgsr.evaluation.sr_stats --config I,II,V2
Pooled median PCC = median over window x predicted-lead pairs; MAE = mean over windows and predicted leads (z units).
95% CIs: record bootstrap (B=1000; seeds ptbxl_test 0, georgia 1, china 2), both windows of a record resampled together,
implemented as weighted statistics with multinomial record counts. Output: $BIMA_RESULTS_DIR/sr_<tag>_overall.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from bima_ecgsr import config as CFG  # noqa: E402
DATA = CFG.DATA_DIR; RUNS = CFG.RUNS_DIR
OUT = CFG.RESULTS_DIR
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SEED = {"ptbxl_test": 0, "georgia": 1, "china": 2}; B = 1000
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="II"); a = ap.parse_args()
inp = a.config.split(","); tag = "_".join(inp); pred = [i for i, l in enumerate(LEADS) if l not in inp]
prec = [i for i in pred if i >= 6]
methods = {"bima_sr": f"bima_sr_{tag}/metrics_{{c}}.npz", "dibima_sr_onestep": f"dibima_sr_{tag}/metrics_{{c}}_onestep.npz",
           "dibima_sr_ddim6": f"dibima_sr_{tag}/metrics_{{c}}_ddim6.npz", "linear_fir": f"baselines_{tag}/metrics_{{c}}_fir.npz",
           "einthoven_fir": f"baselines_{tag}/metrics_{{c}}_einthoven_fir.npz"}
def wmedian_boot(v, rec_of, counts):
    o = np.argsort(v, kind="stable"); vs = v[o]; ro = rec_of[o]; est = np.median(v); res = np.empty(len(counts))
    for b, c in enumerate(counts):
        w = c[ro].astype(np.float64); cw = np.cumsum(w); res[b] = vs[np.searchsorted(cw, 0.5 * cw[-1])]
    return est, np.percentile(res, [2.5, 97.5])
def wmean_boot(v_rec_sum, n_rec, counts):
    est = v_rec_sum.sum() / n_rec.sum(); bs = (counts @ v_rec_sum) / (counts @ n_rec); return est, np.percentile(bs, [2.5, 97.5])
rows = []
for coh in SEED:
    meta = pd.read_csv(f"{DATA}/meta_{coh}.csv"); codes, uniq = pd.factorize(meta.record_id.astype(str), sort=False)
    R = len(uniq); rng = np.random.default_rng(SEED[coh]); counts = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(B)])
    for m, pat in methods.items():
        f = os.path.join(RUNS, pat.format(c=coh))
        if not os.path.exists(f): continue
        d = np.load(f); assert d["pcc"].shape[0] == len(meta)
        for grp, idx in [("all_pred", pred), ("prec_pred", prec)]:
            if not idx: continue
            pcc = d["pcc"][:, idx]; mae = d["mae"][:, idx]
            v = pcc.ravel(); rec_of = np.repeat(codes, len(idx))
            med, (lo, hi) = wmedian_boot(v, rec_of, counts)
            ms = np.bincount(codes, weights=mae.sum(1), minlength=R); nn = np.bincount(codes, minlength=R) * len(idx)
            mm, (mlo, mhi) = wmean_boot(ms, nn.astype(float), counts)
            rows.append(dict(config=a.config, cohort=coh, method=m, group=grp, n_windows=len(meta), n_records=R, pcc_median=med,
                             pcc_lo=lo, pcc_hi=hi, mae_mean=mm, mae_lo=mlo, mae_hi=mhi))
        print(coh, m, flush=True)
df = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True); df.to_csv(f"{OUT}/sr_{tag}_overall.csv", index=False)
print(df[df.group == "all_pred"].round(4).to_string(index=False))
