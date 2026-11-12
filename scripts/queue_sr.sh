#!/bin/bash
# Training queue (sequential): reduced-lead -> 12-lead reconstruction (PTB-XL train; test PTB-XL fold 10, Georgia, China).
# NOT started automatically. Launch from your own shell, from any directory:
#   nohup bash scripts/queue_sr.sh > /dev/null 2>&1 &
# Jobs: BiMa-SR (regression) for II, I, I+II, I+II+V2, I+II+V1+V4 and I+II+V1; DiBiMa-SR (diffusion, T=20
# cosine+zero-terminal-SNR, x0-prediction; one-step and DDIM-6 from N(0,I)) for II; then the CPU baselines job
# (multi-input FIR + Einthoven) for every configuration.
# Recipe: 20 epochs, AdamW 1e-3 cosine -> 1e-5, wd 1e-6, batch 32 = 4 x 8 micro-batches, bf16, clip 1; model selection
# on val_select_1024 (pooled median PCC over predicted leads). Each GPU job waits for >= MIN_FREE_MIB free GPU memory,
# and is retried up to 30x with --resume (training resumes from <run>/ckpt.pt at the end of the last finished epoch;
# finished test cohorts are skipped via done_*.json).
# Outputs: $BIMA_RUNS_DIR/<run>/ (run names bima_sr_<tag>, dibima_sr_<tag>, baselines_<tag>; tag = leads joined by '_')
# Logs: $BIMA_LOG_DIR/{status,<run>}.log (default scripts/logs/)
# Run a subset: JOBS="bima_sr_I_II_V1 baselines_I_II_V1" bash scripts/queue_sr.sh
# Interpreter: BIMA_PYTHON (default: python on PATH; needs torch + mamba_ssm + diffusers, see requirements.txt)
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
PY=${BIMA_PYTHON:-python}
LOGD=${BIMA_LOG_DIR:-$REPO/scripts/logs}; mkdir -p "$LOGD"
OUT_ROOT=${BIMA_RUNS_DIR:-/media/gtimpano/Data/UGO/dibima_sr12_v2_runs}
MIN_FREE_MIB=${MIN_FREE_MIB:-6000}
EPOCHS=${EPOCHS:-20}
JOBS=${JOBS:-"bima_sr_II dibima_sr_II bima_sr_I bima_sr_I_II bima_sr_I_II_V2 bima_sr_I_II_V1_V4 baselines bima_sr_I_II_V1 baselines_I_II_V1"}
COMMON="--amp --accum 4 --bs 32 --eval-bs 128 --epochs $EPOCHS --n-val 1024 --out-root $OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
st() { echo "[$(date '+%F %T')] $*" >> "$LOGD/status.log"; }

run() {  # name mode leads [extra args]
  local name=$1 mode=$2 leads=$3; shift 3
  for attempt in $(seq 1 30); do
    if [ "$mode" != baselines ]; then while [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do sleep 120; done; fi
    st "$name attempt $attempt start (free $(free_mib) MiB)"
    $PY -m bima_ecgsr.training.train_sr --mode $mode --input-leads $leads --name $name $COMMON --resume "$@" >> "$LOGD/$name.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { st "$name done"; return 0; }
    st "$name attempt $attempt failed rc=$rc; retry in 180 s"; sleep 180
  done
  st "$name gave up"; return 1
}

job() {
  case $1 in
    bima_sr_II)         run bima_sr_II         regression II ;;
    dibima_sr_II)       run dibima_sr_II       diffusion  II --T 20 --val-steps 6 ;;
    bima_sr_I)          run bima_sr_I          regression I ;;
    bima_sr_I_II)       run bima_sr_I_II       regression I,II ;;
    bima_sr_I_II_V2)    run bima_sr_I_II_V2    regression I,II,V2 ;;
    bima_sr_I_II_V1_V4) run bima_sr_I_II_V1_V4 regression I,II,V1,V4 ;;
    bima_sr_I_II_V1)    run bima_sr_I_II_V1    regression I,II,V1 ;;     # 5-electrode bedside telemetry
    baselines)  # CPU only: FIR for every config of the first round; + Einthoven/Goldberger limb leads where I and II are inputs
      for c in II I I,II I,II,V2 I,II,V1,V4; do run baselines_${c//,/_} baselines $c; done ;;
    baselines_I_II_V1)  run baselines_I_II_V1  baselines  I,II,V1 ;;
    *) st "unknown job $1" ;;
  esac
}

st "queue start (jobs: $JOBS)"
for j in $JOBS; do job $j; done
st "queue all done"
