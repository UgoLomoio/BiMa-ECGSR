#!/usr/bin/env bash
# Train the 12-lead judge (real 12-lead PTB-XL ECGs) and evaluate it on real ECGs (PTB-XL test, Georgia, China).
# Re-running this script resumes an unfinished run and skips a finished one. NOT started automatically.
#   bash scripts/run_classifier.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
PY=${BIMA_PYTHON:-python}
RUNS=${BIMA_CLS_DIR:-/media/gtimpano/Data/UGO/dibima_cls_v2_runs}
mkdir -p "$RUNS"; cd "$REPO"
n=judge12
if [ -f "$RUNS/$n/summary.json" ]; then echo "[skip] $n finished"
else
  for attempt in 1 2 3; do
    echo "[run] $n attempt=$attempt $(date)"
    "$PY" -m bima_ecgsr.classification.train_classifier --name "$n" --input-leads all --runs "$RUNS" --resume >> "$RUNS/$n.stdout.log" 2>&1 && break
    echo "[warn] $n exited non-zero; resuming"
  done
fi
"$PY" -m bima_ecgsr.classification.evaluate --runs "$RUNS" >> "$RUNS/evaluate.stdout.log" 2>&1
echo "[all done] $(date)"
