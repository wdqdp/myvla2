#!/usr/bin/env bash
# Detached helper: wait three hours, then upload the completed V7.3 Stage-A run.
set -euo pipefail

OUT_DIR="/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_3_no_history"
REPO_ID="wdqdp11/tac_vla"
REMOTE_PATH="stage_a_action/pi05_delta_tac_rotation_phase_v7_3_no_history"
RUN_NAME="pi05_delta_tac_rotation_phase_v7_3_no_history"
LOG_FILE="${OUT_DIR}/modelscope_upload_monitor.log"

log() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG_FILE"
}

final_step() {
    awk -F ': ' '/"num_steps"/ {gsub(/,/, "", $2); gsub(/ /, "", $2); print $2; exit}' \
        "${OUT_DIR}/config.json"
}

training_finished() {
    local step
    step="$(final_step)"
    [[ "$step" =~ ^[1-9][0-9]*$ ]] || {
        log "invalid or missing num_steps in ${OUT_DIR}/config.json"
        return 1
    }
    [[ -d "${OUT_DIR}/${step}/params" ]] || return 1
    ! pgrep -f "scripts/train_vla_stage_a_openpi.py.*--run-name ${RUN_NAME}" >/dev/null
}

upload_once() {
    modelscope upload "$REPO_ID" "$OUT_DIR" "$REMOTE_PATH" \
        --repo-type model \
        --max-workers 8 \
        --use-cache \
        --disable-tqdm \
        --commit-message "Upload V7.3 Stage-A run" >>"$LOG_FILE" 2>&1 &
    local upload_pid=$!
    local elapsed_minutes=0
    while kill -0 "$upload_pid" 2>/dev/null; do
        sleep 60
        elapsed_minutes=$((elapsed_minutes + 1))
        if (( elapsed_minutes % 30 == 0 )) && kill -0 "$upload_pid" 2>/dev/null; then
            log "upload is still running after ${elapsed_minutes} minutes"
        fi
    done
    wait "$upload_pid"
}

log "monitor started; first completion check is in 3 hours"
sleep 3h

while ! training_finished; do
    log "training is not complete; retrying completion check in 30 minutes"
    sleep 30m
done

log "training complete; starting ModelScope upload to ${REMOTE_PATH}"
while true; do
    if upload_once; then
        log "upload succeeded; tmux monitor is exiting"
        exit 0
    fi
    log "upload failed or is incomplete; retrying in 30 minutes"
    sleep 30m
done
