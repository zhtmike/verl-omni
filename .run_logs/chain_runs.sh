#!/usr/bin/env bash
# Wait for run 4 to finish, then launch run 5 (also from-scratch, no resume).
# Both runs use the local script with the same overrides; only the log file differs.
set -u

REPO=/home/public/yx/verl-omni
RUN4_PID=1480395
RUN5_LOG=$REPO/.run_logs/flowgrpo_e190d17_run5_fromscratch.log
SCRIPT=$REPO/examples/flowgrpo_trainer/run_qwen_image_ocr_lora_local.sh

echo "[$(date -u +%H:%M:%S)] chain_runs.sh: waiting for run 4 (pid $RUN4_PID) to exit..." > "$REPO/.run_logs/chain_runs.log"

# Tail-call wait: polls every 60s. Avoids holding the parent shell hostage on `wait`,
# which only works for direct children.
while kill -0 "$RUN4_PID" 2>/dev/null; do
  sleep 60
done

echo "[$(date -u +%H:%M:%S)] chain_runs.sh: run 4 exited. Cleaning up before run 5..." >> "$REPO/.run_logs/chain_runs.log"

# Best-effort cleanup of leftover ray/vllm child processes so run 5 doesn't hit
# EADDRINUSE / leftover GPU memory like run 2 did.
pkill -9 -f "main_flowgrpo|ray::|raylet|gcs_server|vLLM|TaskRunner" 2>/dev/null || true
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | awk 'NF{print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 5
rm -f /dev/shm/psm_* /dev/shm/sem.* /dev/shm/cuda* /dev/shm/vllm* /dev/shm/ray* 2>/dev/null || true
rm -rf /tmp/ray /tmp/vllm_* 2>/dev/null || true
sleep 5

echo "[$(date -u +%H:%M:%S)] chain_runs.sh: launching run 5 (from-scratch)..." >> "$REPO/.run_logs/chain_runs.log"

cd "$REPO"
exec bash "$SCRIPT" trainer.resume_mode=disable > "$RUN5_LOG" 2>&1
