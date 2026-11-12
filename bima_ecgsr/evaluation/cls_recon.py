"""Diagnostic fidelity of reconstructed 12-lead ECGs: ONE classifier, the 12-lead judge trained on real PTB-XL ECGs only,
run on real 12-lead ECGs and on reconstructed 12-lead ECGs (BiMa-SR, DiBiMa-SR one-step, linear filter, Einthoven+filter).
Record-level AUROC (mean of the two 5-s windows), record bootstrap B=1000 with the classifier track's seeds
(ptbxl_test 0, georgia 1, china 2) over the meta record order; the same resamples for every input type -> paired deltas.
Usage: python -m bima_ecgsr.evaluation.cls_recon --config II [--cohorts ptbxl_test,georgia,china]
Outputs: $BIMA_RESULTS_DIR/reconstruction_classification/auroc_recon_<tag>.csv, auroc_recon_deltas_<tag>.csv
"""
import os, sys, argparse, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.classification.predict import load_classifier, predict, aggregate_records, record_labels  # noqa: E402
from bima_ecgsr.classification.metrics import bootstrap_counts, bootstrap_auroc  # noqa: E402
from bima_ecgsr.classification.classifier import LABELS  # noqa: E402
DATA = CFG.DATA_DIR; RUNS = CFG.RUNS_DIR
CRUNS = CFG.CLS_DIR
OUT = os.path.join(CFG.RESULTS_DIR, "reconstruction_classification")
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]; PANEL = [l for l in LABELS if l not in SUPER]
SEED = {"ptbxl_test": 0, "georgia": 1, "china": 2}
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--cohorts", default="ptbxl_test,georgia,china")
a = ap.parse_args(); cfg = a.config; tag = cfg.replace(",", "_")
judge, _ = load_classifier(f"{CRUNS}/judge12/best.pt")
rows, deltas = [], []
for coh in a.cohorts.split(","):
    meta = pd.read_csv(f"{DATA}/meta_{coh}.csv"); meta["cohort"] = coh
    keys, _, codes = aggregate_records(np.zeros((len(meta), 1)), meta)
    Y = record_labels(meta, codes=codes)
    counts = bootstrap_counts(len(keys), 1000, SEED[coh])
    excl = ("AFLT",) if coh == "china" else ()
    groups = [("superclass", SUPER), ("panel", PANEL)] if coh == "ptbxl_test" else [("panel", PANEL)]
    # inputs to the judge
    srcs = {"true_12lead": f"{DATA}/X_{coh}.npy"}
    cand = {f"bima_sr": f"{RUNS}/bima_sr_{tag}/pred_{coh}.npy", "dibima_sr_onestep": f"{RUNS}/dibima_sr_{tag}/pred_{coh}_onestep.npy",
            "linear_fir": f"{RUNS}/baselines_{tag}/pred_{coh}_fir.npy", "einthoven_fir": f"{RUNS}/baselines_{tag}/pred_{coh}_einthoven_fir.npy"}
    srcs.update({k: v for k, v in cand.items() if os.path.exists(v)})
    S = {}
    for name, p in srcs.items():
        X = np.load(p, mmap_mode="r"); assert X.shape[0] == len(meta), (name, X.shape)
        _, S[name], _ = aggregate_records(predict(judge, X, batch_size=256), meta)
    tj = np.load(f"{CRUNS}/judge12/pred_true_{coh}.npz", allow_pickle=True)
    assert np.abs(tj["p_rec"] - S["true_12lead"]).max() < 0.02  # consistency with the classifier track (bf16)
    B = {}
    for gname, labs in groups:
        idx = [LABELS.index(l) for l in labs]
        for name, s in S.items():
            r, Bm, names = bootstrap_auroc(Y[:, idx], s[:, idx], labs, counts, exclude=excl)
            for x in r: rows.append(dict(config=cfg, cohort=coh, group=gname, input=name, **x))
            B[(gname, name)] = np.nanmean(Bm, 1)
        for x_, y_ in [("bima_sr", "true_12lead"), ("dibima_sr_onestep", "true_12lead"), ("linear_fir", "true_12lead"),
                       ("einthoven_fir", "true_12lead"), ("bima_sr", "linear_fir"), ("bima_sr", "einthoven_fir"),
                       ("bima_sr", "dibima_sr_onestep")]:
            if (gname, x_) in B and (gname, y_) in B:
                d = B[(gname, x_)] - B[(gname, y_)]
                deltas.append(dict(config=cfg, cohort=coh, group=gname, a=x_, b=y_, delta_macro_boot_mean=float(np.mean(d)),
                                   ci_low=float(np.percentile(d, 2.5)), ci_high=float(np.percentile(d, 97.5)),
                                   p_two_sided=float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))))
    print(coh, "done", flush=True)
R = pd.DataFrame(rows); D = pd.DataFrame(deltas)
os.makedirs(OUT, exist_ok=True); R.to_csv(f"{OUT}/auroc_recon_{tag}.csv", index=False); D.to_csv(f"{OUT}/auroc_recon_deltas_{tag}.csv", index=False)
m = R[R.label == "macro"].pivot_table(index=["cohort", "group"], columns="input", values="auroc")
print(m.round(4).to_string()); print(D.round(4).to_string(index=False))
