#!/bin/bash
# Smoke test of bima_ecgsr.training.train_sr (scratch outputs; reduced steps/val/test). Per run:
#  A) 2 epochs x 40 steps, simulated crash after epoch 0 (rc 3 expected)
#  B) --resume: continues at epoch 1, then tests on the first N_TEST windows of each cohort
#  C) --resume again: must skip training and every finished test cohort
# Usage: SCR=/path/to/scratch bash scripts/smoke_sr.sh [name:mode:leads ...]
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
PY=${BIMA_PYTHON:-python}
SCR=${SCR:?set SCR to a scratch output root (never the study runs directory)}
LOGD=${BIMA_LOG_DIR:-$REPO/scripts/logs}/smoke; mkdir -p "$LOGD"
N_TEST=${N_TEST:-256}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
C="--amp --accum 4 --bs 32 --eval-bs 64 --epochs 2 --max-steps 40 --log-every 10 --n-val 128 --n-test $N_TEST --out-root $SCR"
one() { local name=$1 mode=$2 leads=$3
  $PY -m bima_ecgsr.training.train_sr --mode $mode --input-leads $leads --name $name $C --exit-after-epoch 0 > "$LOGD/$name.A.log" 2>&1; echo "$name A rc=$?"
  $PY -m bima_ecgsr.training.train_sr --mode $mode --input-leads $leads --name $name $C --resume > "$LOGD/$name.B.log" 2>&1; echo "$name B rc=$?"
  $PY -m bima_ecgsr.training.train_sr --mode $mode --input-leads $leads --name $name $C --resume > "$LOGD/$name.C.log" 2>&1; echo "$name C rc=$?"
}
for spec in "${@:-bima_sr_II:regression:II dibima_sr_II:diffusion:II bima_sr_I:regression:I bima_sr_I_II:regression:I,II bima_sr_I_II_V1:regression:I,II,V1 bima_sr_I_II_V2:regression:I,II,V2 bima_sr_I_II_V1_V4:regression:I,II,V1,V4}"; do
  for s in $spec; do IFS=: read n m l <<< "$s"; one $n $m $l; done
done
