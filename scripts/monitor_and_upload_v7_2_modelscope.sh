#!/usr/bin/env bash
# Detached helper: wait six hours, then upload the completed V7.2 Stage-A run.
set -euo pipefail

OUT_DIR="/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_2_no_history"
REPO_ID="wdqdp11/tac_vla"
REMOTE_PATH="stage_a_action/ta_tac_rotation_phase_v7_2_no_history"
RUN_NAME="pi05_delta_tac_rotation_phase_v7_2_no_history"
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

log "monitor started; first completion check is in 6 hours"
sleep 6h

while ! training_finished; do
    log "training is not complete; retrying completion check in 30 minutes"
    sleep 30m
done

log "training complete; starting ModelScope upload to ${REMOTE_PATH}"
while true; do
    if modelscope upload "$REPO_ID" "$OUT_DIR" "$REMOTE_PATH" \
        --repo-type model \
        --exclude '5000/**' \
        --max-workers 8 \
        --use-cache \
        --disable-tqdm \
        --commit-message "Upload V7.2 final Stage-A checkpoint and metadata" >>"$LOG_FILE" 2>&1; then
        log "upload succeeded; tmux monitor is exiting"
        exit 0
    fi
    log "upload failed or is incomplete; retrying in 30 minutes"
    sleep 30m
done
