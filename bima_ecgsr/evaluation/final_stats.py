"""Reconstruction-fidelity statistics for every input configuration (paper tables and figures).

Usage: python -m bima_ecgsr.evaluation.final_stats [--configs II,I,I+II,...] [--B 1000]

Definitions (z units, per-window per-lead metrics from <run>/metrics_<cohort>[_suffix].npz):
  pooled median PCC  = median over all window x lead pairs of the leads in the group (headline statistic)
  per-record PCC     = mean over the group's leads, averaged over the record's windows (used for paired tests)
  MAE / RMSE         = mean over window x lead pairs
Groups: all_pred (all predicted leads), limb_pred (predicted limb leads), prec_pred (predicted chest leads V1-V6).
95% CIs: record bootstrap (B=1000; seeds ptbxl_test 0, georgia 1, china 2), both windows of a record resampled
together, implemented as weighted statistics with multinomial record counts (identical resamples to sr_stats.py
and to the classification evaluation).

Outputs ($BIMA_RESULTS_DIR/final/):
  sr_overall.csv     config, cohort, method, group, pooled median PCC [CI], MAE [CI], RMSE, per-record PCC median
  sr_per_lead.csv    config, cohort, method, lead, role (input/predicted), median PCC, mean MAE
  sr_per_class.csv   config, cohort, method, label, n_records, pooled median PCC (all_pred / prec_pred), MAE
  sr_paired.csv      per-record paired differences (BiMa-SR minus comparator): mean [CI], fraction improved, Wilcoxon p
"""
import os, sys, argparse, time
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bima_ecgsr import config as CFG  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SEED = {"ptbxl_test": 0, "georgia": 1, "china": 2}
CONFIGS = ["II", "I", "I,II", "I,II,V2", "I,II,V1", "I,II,V1,V4"]
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
PANEL = ["AFIB", "AFLT", "1AVB", "PAC", "PVC", "SBRAD", "STACH", "WPW", "LVH", "RVH", "CRBBB", "CLBBB"]
MIN_POS = 10


def method_files(tag):
    return {"bima_sr": f"bima_sr_{tag}/metrics_{{c}}.npz",
            "dibima_sr_onestep": f"dibima_sr_{tag}/metrics_{{c}}_onestep.npz",
            "dibima_sr_ddim6": f"dibima_sr_{tag}/metrics_{{c}}_ddim6.npz",
            "linear_fir": f"baselines_{tag}/metrics_{{c}}_fir.npz",
            "einthoven_fir": f"baselines_{tag}/metrics_{{c}}_einthoven_fir.npz"}


def wmedian_boot(v, rec_of, counts):
    o = np.argsort(v, kind="stable"); vs = v[o]; ro = rec_of[o]
    res = np.empty(len(counts))
    for b, c in enumerate(counts):
        cw = np.cumsum(c[ro].astype(np.float64)); res[b] = vs[np.searchsorted(cw, 0.5 * cw[-1])]
    return float(np.median(v)), np.percentile(res, [2.5, 97.5])


def wmean_boot(rec_sum, rec_n, counts):
    est = rec_sum.sum() / rec_n.sum(); bs = (counts @ rec_sum) / (counts @ rec_n)
    return float(est), np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(c.replace(",", "+") for c in CONFIGS))
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    cfgs = [c.replace("+", ",") for c in a.configs.split(",")]
    out = os.path.join(CFG.RESULTS_DIR, "final"); os.makedirs(out, exist_ok=True)
    overall, per_lead, per_class, paired = [], [], [], []
    for coh, seed in SEED.items():
        meta = pd.read_csv(f"{CFG.DATA_DIR}/meta_{coh}.csv")
        codes, uniq = pd.factorize(meta.record_id.astype(str), sort=False); R = len(uniq)
        rng = np.random.default_rng(seed)
        counts = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(a.B)])
        rec_n_w = np.bincount(codes, minlength=R).astype(np.float64)
        labs = [l for l in (SUPER + PANEL) if l in meta.columns and not (coh == "china" and l == "AFLT")]
        rec_lab = {l: (meta.groupby(codes)[l].max().values == 1) for l in labs}
        for cfg in cfgs:
            inp = cfg.split(","); tag = "_".join(inp)
            pred = [i for i, l in enumerate(LEADS) if l not in inp]
            groups = {"all_pred": pred, "limb_pred": [i for i in pred if i < 6], "prec_pred": [i for i in pred if i >= 6]}
            recpcc = {}
            for m, pat in method_files(tag).items():
                f = os.path.join(CFG.RUNS_DIR, pat.format(c=coh))
                if not os.path.exists(f):
                    continue
                t0 = time.time(); d = np.load(f); assert d["pcc"].shape == (len(meta), 12), (f, d["pcc"].shape)
                for g, idx in groups.items():
                    if not idx:
                        continue
                    pcc = d["pcc"][:, idx].astype(np.float64); mae = d["mae"][:, idx].astype(np.float64)
                    rmse = d["rmse"][:, idx].astype(np.float64)
                    med, (lo, hi) = wmedian_boot(pcc.ravel(), np.repeat(codes, len(idx)), counts)
                    mae_rs = np.bincount(codes, mae.sum(1), minlength=R); n_rs = rec_n_w * len(idx)
                    mmae, (mlo, mhi) = wmean_boot(mae_rs, n_rs, counts)
                    rp = np.bincount(codes, np.nanmean(pcc, 1), minlength=R) / rec_n_w
                    recpcc[(m, g)] = rp
                    recpcc[(m, g, "mae")] = np.bincount(codes, mae.mean(1), minlength=R) / rec_n_w
                    overall.append(dict(config=cfg, cohort=coh, method=m, group=g, n_leads=len(idx), n_records=R,
                                        n_windows=len(meta), pcc_pooled_median=med, pcc_ci_low=lo, pcc_ci_high=hi,
                                        mae_mean=mmae, mae_ci_low=mlo, mae_ci_high=mhi, rmse_mean=float(rmse.mean()),
                                        pcc_per_record_median=float(np.nanmedian(rp))))
                    if g == "all_pred":
                        for l in labs:
                            sel = rec_lab[l][codes]
                            if rec_lab[l].sum() < MIN_POS:
                                continue
                            for gg in ("all_pred", "prec_pred"):
                                ii = groups[gg]
                                if not ii:
                                    continue
                                per_class.append(dict(config=cfg, cohort=coh, method=m, label=l, group=gg,
                                                      n_records=int(rec_lab[l].sum()),
                                                      pcc_pooled_median=float(np.nanmedian(d["pcc"][sel][:, ii])),
                                                      mae_mean=float(d["mae"][sel][:, ii].mean())))
                for li, ln in enumerate(LEADS):
                    per_lead.append(dict(config=cfg, cohort=coh, method=m, lead=ln,
                                         role="input" if ln in inp else "predicted",
                                         pcc_median=float(np.nanmedian(d["pcc"][:, li])),
                                         mae_mean=float(d["mae"][:, li].mean())))
                print(f"{coh} {cfg} {m} {time.time() - t0:.1f}s", flush=True)
            for comp in ["linear_fir", "einthoven_fir", "dibima_sr_onestep", "dibima_sr_ddim6"]:
                for g in groups:
                    if ("bima_sr", g) not in recpcc or (comp, g) not in recpcc:
                        continue
                    for metric, key_a, key_b, better in [("pcc", ("bima_sr", g), (comp, g), 1),
                                                         ("mae", ("bima_sr", g, "mae"), (comp, g, "mae"), -1)]:
                        diff = recpcc[key_a] - recpcc[key_b]
                        bs = (counts @ diff) / R
                        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                        paired.append(dict(config=cfg, cohort=coh, group=g, metric=metric, a="bima_sr", b=comp,
                                           n_records=R, mean_diff=float(diff.mean()),
                                           ci_low=float(np.percentile(bs, 2.5)), ci_high=float(np.percentile(bs, 97.5)),
                                           frac_a_better=float(np.mean(better * diff > 0)), wilcoxon_p=float(p)))
    for name, rows in [("sr_overall", overall), ("sr_per_lead", per_lead), ("sr_per_class", per_class), ("sr_paired", paired)]:
        pd.DataFrame(rows).to_csv(os.path.join(out, f"{name}.csv"), index=False)
        print(name, len(rows))


if __name__ == "__main__":
    main()
