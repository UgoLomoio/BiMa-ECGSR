#!/usr/bin/env bash
# Post-training analysis for every input configuration whose runs exist: fidelity statistics with record-bootstrap
# CIs (bima_ecgsr.evaluation.sr_stats) and judge AUROC on reconstructed ECGs (bima_ecgsr.evaluation.cls_recon),
# then the figures. NOT started automatically. Needs the GPU for the judge (cls_recon).
#   bash scripts/run_all.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
PY=${BIMA_PYTHON:-python}
RUNS=${BIMA_RUNS_DIR:-/media/gtimpano/Data/UGO/dibima_sr12_v2_runs}
cd "$REPO"
for cfg in II I I,II I,II,V1 I,II,V2 I,II,V1,V4; do
  tag=${cfg//,/_}
  if [ -f "$RUNS/bima_sr_$tag/summary.csv" ]; then
    echo "[stats] $cfg"; "$PY" -m bima_ecgsr.evaluation.sr_stats --config "$cfg"
    echo "[judge] $cfg"; "$PY" -m bima_ecgsr.evaluation.cls_recon --config "$cfg"
  else
    echo "[skip] $cfg (bima_sr_$tag not finished)"
  fi
done
"$PY" figures/graphical_abstract.py
"$PY" figures/figure1.py
