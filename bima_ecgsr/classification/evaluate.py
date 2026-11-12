"""Evaluate every trained classifier on TRUE ECGs (PTB-XL test, Georgia, China) at record level.

  python -m bima_ecgsr.classification.evaluate [--names judge12] [--B 1000]

For each classifier <name> (best.pt = best validation epoch):
  * window probabilities -> record probabilities (mean over the record's windows, predict.aggregate_records)
    saved to <runs>/<name>/pred_true_<cohort>.npz  (record keys, p_rec [R,17], y_rec [R,17], labels)
  * record-level AUROC with record-bootstrap 95% percentile CIs (B = 1000):
      PTB-XL test : 5 superclasses (records with has_superclass == 1) + macro;
                    panel labels with >= 10 positive records + macro
      Georgia     : panel labels only (-1 = unknown excluded per label), >= 10 positive records
      China       : panel labels only, -1 excluded, AFLT never scored (invalid label), >= 10 positives
Bootstrap resamples are drawn once per cohort (seed: ptbxl_test 0, georgia 1, china 2) over the record
order of meta_<cohort>.csv (first appearance), so all classifiers -- and later the reconstructed-ECG
evaluations that use metrics.bootstrap_counts with the same seed/order -- share identical resamples.
Outputs: <runs>/results/auroc_true_records.csv (long), auroc_true_summary.csv (macro per classifier/cohort/group);
copies in <repo>/results/classification/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root (script use)
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.classification.classifier import LABELS, PANEL, SUPERCLASSES  # noqa: E402
from bima_ecgsr.classification.metrics import bootstrap_auroc, bootstrap_counts  # noqa: E402
from bima_ecgsr.classification.predict import aggregate_records, load_classifier, predict, record_labels  # noqa: E402

DATA = CFG.DATA_DIR
RUNS = CFG.CLS_DIR
NAMES = ["judge12"]
COHORTS = {"ptbxl_test": ("X_ptbxl_test.npy", "meta_ptbxl_test.csv", 0),
           "georgia": ("X_georgia.npy", "meta_georgia.csv", 1),
           "china": ("X_china.npy", "meta_china.csv", 2)}
EXCLUDE = {"china": ("AFLT",)}
MIN_POS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=",".join(NAMES))
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--gpu-mem-gb", type=float, default=4.0)
    args = ap.parse_args()
    names = [n for n in args.names.split(",") if os.path.exists(os.path.join(args.runs, n, "best.pt"))]
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gb * 2**30 / total), 0)
    resdir = os.path.join(args.runs, "results"); os.makedirs(resdir, exist_ok=True)

    rows = []
    for cohort, (xf, mf, seed) in COHORTS.items():
        meta = pd.read_csv(os.path.join(args.data, mf), low_memory=False, dtype={"record_id": str})
        X = np.load(os.path.join(args.data, xf), mmap_mode="r")
        keys, _, codes = aggregate_records(np.zeros((len(meta), 1), np.float32), meta)
        Yr = record_labels(meta, codes=codes)
        counts = bootstrap_counts(len(keys), args.B, seed)
        print(f"[{cohort}] {len(meta)} windows, {len(keys)} records", flush=True)
        for name in names:
            out_npz = os.path.join(args.runs, name, f"pred_true_{cohort}.npz")
            ck = os.path.join(args.runs, name, "best.pt")
            if os.path.exists(out_npz) and os.path.getmtime(out_npz) > os.path.getmtime(ck):
                z = np.load(out_npz, allow_pickle=True); pr = z["p_rec"]; info = json.loads(str(z["info"]))
            else:
                model, info = load_classifier(ck)
                pw = predict(model, X, lead_idx=info["lead_idx"], batch_size=256)
                _, pr, _ = aggregate_records(pw, meta)
                np.savez_compressed(out_npz, cohort=np.array(keys["cohort"].astype(str)),
                                    record_id=np.array(keys["record_id"].astype(str)), p_rec=pr, y_rec=Yr,
                                    p_win=pw.astype(np.float16), labels=np.array(LABELS),
                                    info=json.dumps({k: info[k] for k in ("name", "input_leads", "lead_idx")}))
                del model; torch.cuda.empty_cache()
            groups = [("panel", PANEL)] if cohort != "ptbxl_test" else [("superclass", SUPERCLASSES), ("panel", PANEL)]
            for gname, labs in groups:
                jj = [LABELS.index(l) for l in labs]
                rr, _, _ = bootstrap_auroc(Yr[:, jj], pr[:, jj], list(labs), counts, min_pos=MIN_POS,
                                           exclude=EXCLUDE.get(cohort, ()))
                for r in rr:
                    rows.append(dict(classifier=name, input_leads=info["input_leads"], cohort=cohort, group=gname,
                                     n_records=len(keys), **r))
                mac = [r for r in rr if r["label"] == "macro"][0]
                print(f"  {name:15s} {gname:10s} macro {mac['auroc']:.4f} [{mac['ci_low']:.4f}, {mac['ci_high']:.4f}]",
                      flush=True)
    df = pd.DataFrame(rows)
    p1 = os.path.join(resdir, "auroc_true_records.csv"); df.to_csv(p1, index=False, float_format="%.5f")
    summ = df[df.label == "macro"].rename(columns={"n_pos": "n_labels", "status": "labels_evaluated"}).drop(columns="n_neg")
    p2 = os.path.join(resdir, "auroc_true_summary.csv"); summ.to_csv(p2, index=False, float_format="%.5f")
    cdir = os.path.join(CFG.RESULTS_DIR, "classification"); os.makedirs(cdir, exist_ok=True)
    for p in (p1, p2):
        shutil.copy2(p, cdir)
    print("[saved]", p1, p2, flush=True)


if __name__ == "__main__":
    main()
