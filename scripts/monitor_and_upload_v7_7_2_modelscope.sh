#!/usr/bin/env bash
# Wait five hours for V7.7.2 training, then upload the requested local outputs.
set -euo pipefail

ROOT="/data1/qxh/tac_vla_new/tac_data/demon_data/black_box/outputs"
STAGE_A_DIR="${ROOT}/stage_a_action/pi05_delta_tac_rotation_phase_v7_4_2_no_history"
MULTITASK_DIR="${ROOT}/multitask_v7_7_2"
RUN_DIR="${MULTITASK_DIR}/pi05_rotation_v7_7_2_five_task_h100_no_history"
REPO_ID="wdqdp11/tac_vla"
LOG_FILE="/tmp/v7_7_2_modelscope_monitor.log"

log() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG_FILE"
}

training_finished() {
    local expected_step latest_step
    if pgrep -f 'scripts/train_vla_multitask_v7_7_2.py' >/dev/null; then
        log "V7.7.2 training process is still running"
        return 1
    fi
    if [[ ! -f "${RUN_DIR}/config.json" ]]; then
        log "missing run config: ${RUN_DIR}/config.json"
        return 1
    fi
    expected_step="$(jq -r '.num_steps // empty' "${RUN_DIR}/config.json")"
    if [[ ! "${expected_step}" =~ ^[1-9][0-9]*$ ]]; then
        log "invalid num_steps in ${RUN_DIR}/config.json: ${expected_step}"
        return 1
    fi
    if [[ ! -d "${RUN_DIR}/${expected_step}/full_params" ]]; then
        log "training process has exited, but final ${expected_step}/full_params is not present"
        return 1
    fi
    latest_step="$(jq -sr '[.[] | select(.step != null) | .step] | max // 0' "${RUN_DIR}/metrics.jsonl")"
    if (( latest_step < expected_step )); then
        log "latest metric step=${latest_step}; waiting for final metric step=${expected_step}"
        return 1
    fi
    log "V7.7.2 training finished at step ${expected_step}; final full_params exists"
    return 0
}

upload_one() {
    local source_dir="$1" remote_path="$2" message="$3"
    local pid started now elapsed cache_size source_size
    modelscope upload "${REPO_ID}" "${source_dir}" "${remote_path}" \
        --repo-type model \
        --max-workers 8 \
        --use-cache \
        --disable-tqdm \
        --commit-message "${message}" >>"${LOG_FILE}" 2>&1 &
    pid=$!
    started="$(date +%s)"
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 60
        if ! kill -0 "${pid}" 2>/dev/null; then
            break
        fi
        now="$(date +%s)"
        elapsed=$(( (now - started) / 60 ))
        if (( elapsed > 0 && elapsed % 30 == 0 )); then
            source_size="$(du -sh "${source_dir}" 2>/dev/null | awk '{print $1}')"
            cache_size="$(du -sh "${source_dir}/.ms_upload_cache" 2>/dev/null | awk '{print $1}')"
            log "upload still running: ${source_dir} -> ${REPO_ID}/${remote_path}; elapsed=${elapsed}min; source=${source_size:-unknown}; upload_cache=${cache_size:-not present}"
            tail -n 8 "${LOG_FILE}" | sed 's/^/  /' | tee -a "${LOG_FILE}"
        fi
    done
    wait "${pid}"
}

log "monitor started; first train2 completion check is scheduled in 5 hours"
sleep 5h

while ! training_finished; do
    log "training is not ready for upload; checking again in 30 minutes"
    sleep 30m
done

if [[ ! -d "${STAGE_A_DIR}/10000/params" ]]; then
    log "missing V7.4.2 Stage-A 10000-step params: ${STAGE_A_DIR}/10000/params"
    exit 1
fi
if [[ ! -d "${MULTITASK_DIR}" ]]; then
    log "missing V7.7.2 output directory: ${MULTITASK_DIR}"
    exit 1
fi

while true; do
    if upload_one "${STAGE_A_DIR}" \
        "stage_a_action/pi05_delta_tac_rotation_phase_v7_4_2_no_history" \
        "Upload V7.4.2 Stage-A run"; then
        log "Stage-A upload succeeded"
        break
    fi
    log "Stage-A upload failed; retrying in 30 minutes"
    sleep 30m
done

while true; do
    if upload_one "${MULTITASK_DIR}" "outputs/multitask_v7_7_2" \
        "Upload V7.7.2 multitask outputs"; then
        log "V7.7.2 output upload succeeded; monitor exiting"
        exit 0
    fi
    log "V7.7.2 output upload failed; retrying in 30 minutes"
    sleep 30m
done
