# 12-lead ECG classifier (the judge)

One classifier, trained on real 12-lead PTB-XL ECGs, is run unchanged on real 12-lead ECGs and on reconstructed 12-lead
ECGs; the AUROC difference measures how much diagnostic information a reconstruction preserves.

Data: `$BIMA_DATA_DIR` (see [PIPELINE.md](PIPELINE.md)). Runs: `$BIMA_CLS_DIR/<name>/` (the study run is `judge12`).

| module | purpose |
|---|---|
| `bima_ecgsr/classification/classifier.py` | xresnet1d-style 1D-ResNet (k=7, widths 64/128/192/256, 2 blocks per stage, concat-pool head), 3.36 M params, `in_channels` configurable, 17 sigmoid outputs (`LABELS` = NORM, MI, STTC, CD, HYP + AFIB, AFLT, 1AVB, PAC, PVC, SBRAD, STACH, WPW, LVH, RVH, CRBBB, CLBBB) |
| `bima_ecgsr/classification/train_classifier.py` | training on PTB-XL folds 1-8 (ecg 10565 excluded), masked BCE, early stopping on fold-9 record-level superclass macro AUROC; checkpoint each epoch, `--resume` |
| `bima_ecgsr/classification/predict.py` | `load_classifier`, `predict(model, X, lead_idx=None)` -> window probabilities, `aggregate_records(p_win, meta)` -> record probabilities (mean over windows, keyed by cohort + record_id), `record_labels` |
| `bima_ecgsr/classification/metrics.py` | AUROC (Mann-Whitney, equal to sklearn) and record-bootstrap AUROC (weighted, shared resamples) |
| `bima_ecgsr/classification/evaluate.py` | real-ECG evaluation on PTB-XL test / Georgia / China, CSVs in `results/classification/` |
| `bima_ecgsr/evaluation/cls_recon.py` | the same classifier on reconstructed ECGs, paired bootstrap deltas, CSVs in `results/reconstruction_classification/` |
| `scripts/run_classifier.sh` | trains the judge (resumable) and runs `evaluate` |

## Input handling (training and prediction)
`predict.zscore_windows`: each window, each lead: (x - mean_t) / std_t (population std); leads with std < 1e-6 are only
mean-centred (flat-lead guard); NaN/inf -> 0. It is applied to every training batch and in `predict()`, so real
windows (already z-scored), destandardised mV windows (`z*std+mean`) and reconstructed ECGs are treated the same way
(max probability difference between z-score input and mV input on 512 PTB-XL test windows: 0.004, from bf16/float16
rounding). Consequence: absolute amplitudes and inter-lead amplitude ratios are not visible to the classifier (this
mainly costs HYP).

For reconstructed 12-lead ECGs `Xrec` [N,12,2500]:
```python
from bima_ecgsr import config
from bima_ecgsr.classification.predict import load_classifier, predict, aggregate_records
m, info = load_classifier(config.judge_ckpt())
p_rec = aggregate_records(predict(m, Xrec, lead_idx=info['lead_idx']), meta)[1]
```
For paired bootstrap tests against the real-ECG results use `metrics.bootstrap_counts(n_records, 1000, seed)` with
seed 0 (ptbxl_test), 1 (georgia), 2 (china) over the record order of `meta_<cohort>.csv` (first appearance): these are
the resamples used in `results/classification/auroc_true_*.csv`. Real-ECG record probabilities are in
`<run>/pred_true_<cohort>.npz`.

## Training recipe
AdamW (lr 1e-3, wd 1e-2), OneCycle cosine over 30 epochs (pct_start 0.3), batch 128, bf16 autocast, grad clip 1.0,
random 2250-sample crop (train only), seed 0, patience 5. Peak allocated GPU memory about 0.45 GB, about 12 s per epoch.
A validation-only schedule check (15-epoch OneCycle; 15 epochs with wd 0.05 and a 2000-sample crop; lr 3e-4) gave
fold-9 macro AUROC 0.9150 / 0.9151 / 0.9127, against 0.9153 for the default recipe, so the default recipe was kept.

## Evaluation rules
Record-level probability = mean of the record's windows. PTB-XL test: superclasses on records with has_superclass = 1;
panel labels with >= 10 positive records (AFLT 7 and WPW 8 are skipped). Georgia / China: panel only, -1 excluded per
label, labels with < 10 positive records skipped (Georgia WPW, 2), China AFLT never scored. 95% CI = percentile record
bootstrap, B = 1000.
