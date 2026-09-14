#!/usr/bin/env bash
# Detached helper: wait nine hours, then upload a completed V7.7 multitask run.
set -euo pipefail

OUT_DIR="/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/multitask_v7_7/pi05_rotation_v7_7_five_task_h100_no_history"
REPO_ID="wdqdp11/tac_vla"
REMOTE_PATH="pi05_rotation_v7_7_five_task_h100_no_history"
RUN_NAME="pi05_rotation_v7_7_five_task_h100_no_history"
TRAIN_TMUX="train2"
LOG_FILE="${OUT_DIR}/modelscope_upload_monitor.log"

log() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG_FILE"
}

final_step() {
    awk -F ': ' '/"num_steps"/ {gsub(/,/, "", $2); gsub(/ /, "", $2); print $2; exit}' \
        "${OUT_DIR}/config.json"
}

training_finished() {
    local step train2_state
    step="$(final_step)"
    [[ "$step" =~ ^[1-9][0-9]*$ ]] || {
        log "invalid or missing num_steps in ${OUT_DIR}/config.json"
        return 1
    }
    if tmux has-session -t "$TRAIN_TMUX" 2>/dev/null; then
        train2_state="present"
    else
        train2_state="absent"
    fi
    [[ -d "${OUT_DIR}/${step}/full_params" ]] || {
        log "train2=${train2_state}; final deployable checkpoint ${step}/full_params is not present"
        return 1
    }
    if pgrep -f "scripts/train_vla_multitask_v7_7.py.*--run-name ${RUN_NAME}" >/dev/null; then
        log "train2=${train2_state}; training process is still running"
        return 1
    fi
    log "train2=${train2_state}; final checkpoint ${step}/full_params exists and training process has exited"
}

upload_once() {
    modelscope upload "$REPO_ID" "$OUT_DIR" "$REMOTE_PATH" \
        --repo-type model \
        --max-workers 8 \
        --use-cache \
        --disable-tqdm \
        --commit-message "Upload V7.7 five-task multitask run" >>"$LOG_FILE" 2>&1 &
    local upload_pid=$!
    local elapsed_minutes=0
    while kill -0 "$upload_pid" 2>/dev/null; do
        sleep 60
        elapsed_minutes=$((elapsed_minutes + 1))
        if (( elapsed_minutes % 30 == 0 )) && kill -0 "$upload_pid" 2>/dev/null; then
            log "upload still running after ${elapsed_minutes} minutes; local size=$(du -sh "$OUT_DIR" | awk '{print $1}')"
        fi
    done
    wait "$upload_pid"
}

log "monitor started in tmux ${TRAIN_TMUX}; first completion check is in 9 hours"
sleep 9h

while ! training_finished; do
    log "training is not complete; retrying train2 completion check in 30 minutes"
    sleep 30m
done

log "training complete; starting ModelScope upload to ${REPO_ID}/${REMOTE_PATH}"
while true; do
    if upload_once; then
        log "upload succeeded; monitor window is exiting"
        exit 0
    fi
    log "upload failed; retrying in 30 minutes"
    sleep 30m
done
