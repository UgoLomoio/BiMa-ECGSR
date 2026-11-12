#!/bin/bash
# Follow-up queue: BiMa-SR for the telemetry configuration I+II+V1 (5-electrode bedside monitoring) plus its
# FIR/Einthoven baseline. Same recipe as queue_sr.sh (these two jobs are also available there as
# JOBS="baselines_I_II_V1 bima_sr_I_II_V1"). Waits until the main queue logs "queue all done" (a NEW occurrence after
# this script starts), so it can be launched while the main queue is running. NOT started automatically:
#   nohup bash scripts/queue_sr_extra.sh > /dev/null 2>&1 &
# Set NO_WAIT=1 to start immediately.
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
LOGD=${BIMA_LOG_DIR:-$REPO/scripts/logs}; mkdir -p "$LOGD"
n0=$(grep -c "queue all done" "$LOGD/status.log" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] extra queue armed (waiting for main queue; done-count now $n0)" >> "$LOGD/status.log"
if [ -z "$NO_WAIT" ]; then
  while [ "$(grep -c 'queue all done' "$LOGD/status.log")" -le "$n0" ]; do sleep 300; done
fi
JOBS="baselines_I_II_V1 bima_sr_I_II_V1" bash "$HERE/queue_sr.sh"
