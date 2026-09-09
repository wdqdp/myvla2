#!/usr/bin/env bash
# Detached tmux helper: wait eight hours, then upload the completed V7.1 model.
set -euo pipefail

OUT_DIR="/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs/stage_a_action/pi05_delta_tac_rotation_phase_v7_1_no_history"
REPO_ID="wdqdp11/tac_vla"
REMOTE_PATH="tac_vla/stage_a_action/pi05_delta_tac_rotation_phase_v7_1_no_history"
RUN_NAME="pi05_delta_tac_rotation_phase_v7_1_no_history"
LOG_FILE="${OUT_DIR}/modelscope_upload_monitor.log"

log() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG_FILE"
}

training_finished() {
    [[ -d "${OUT_DIR}/15000/params" ]] || return 1
    ! pgrep -f "scripts/train_vla_stage_a_openpi.py.*--run-name ${RUN_NAME}" >/dev/null
}

log "monitor started; first completion check is in 8 hours"
sleep 8h

while ! training_finished; do
    log "training is not complete; retrying completion check in 30 minutes"
    sleep 30m
done

log "training complete; starting ModelScope upload"
while true; do
    if modelscope upload "$REPO_ID" "$OUT_DIR" "$REMOTE_PATH" \
        --repo-type model \
        --exclude '5000/**' \
        --exclude '10000/**' \
        --max-workers 8 \
        --use-cache \
        --disable-tqdm \
        --commit-message "Upload ${RUN_NAME} final checkpoint and metadata" >>"$LOG_FILE" 2>&1; then
        log "upload succeeded; tmux monitor is exiting"
        exit 0
    fi
    log "upload failed; retrying in 30 minutes"
    sleep 30m
done
