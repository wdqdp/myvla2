#!/usr/bin/env bash
# Train the three V7.10 qpos-history ablations sequentially on the same GPUs.
set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${V7_10_PYTHON:-${project_root}/openpi/.venv/bin/python}"
training_script="${project_root}/scripts/train_vla_multitask_v7_10.py"
print_only=0
training_args=()

usage() {
    printf '%s\n' \
        'Usage: bash scripts/run_v7_10_multitask.sh [--dry-run] [TRAINING_ARGS...]' \
        '' \
        'Train 0s, then 1s, then 2s; stop immediately if any training process fails.' \
        'Common training arguments (e.g. --num-steps 2000 --fsdp-devices 1)' \
        'are passed to all three runs. CUDA_VISIBLE_DEVICES is inherited.' \
        '--dry-run prints the three commands without creating logs or starting training.' \
        'Group-specific index/profile/run-name flags are managed by this script.' \
        'V7_10_LOG_DIR optionally sets the pipeline log root.' \
        'V7_10_PYTHON optionally sets the Python interpreter.'
}

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --dry-run)
            print_only=1
            ;;
        --)
            shift
            training_args+=("$@")
            break
            ;;
        *)
            training_args+=("$1")
            ;;
    esac
    shift
done

for argument in "${training_args[@]}"; do
    case "${argument%%=*}" in
        --history-window|--index-file|--run-name|--data-profile|--prompt-profile)
            printf 'Error: %s is group-specific; omit it from the sequential runner.\n' "${argument%%=*}" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$python_bin" ]]; then
    printf 'Python interpreter is not executable: %s\n' "$python_bin" >&2
    exit 2
fi

export USE_TF="${USE_TF:-0}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/qxh/.cache/openpi}"
export PYTHONUNBUFFERED=1
cd -- "$project_root"

if (( ! print_only )); then
    log_root="${V7_10_LOG_DIR:-${project_root}/outputs/multitask_v7_10_pipeline}"
    mkdir -p -- "$log_root"
    log_dir="$(mktemp -d "${log_root}/sequence_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
    printf 'Pipeline logs: %s\nCUDA_VISIBLE_DEVICES=%s\n' "$log_dir" "${CUDA_VISIBLE_DEVICES:-not set}"
fi

for window in 0s 1s 2s; do
    command=("$python_bin" -u "$training_script" --history-window "$window" "${training_args[@]}")
    printf '\n[V7.10 %s] ' "$window"
    printf '%q ' "${command[@]}"
    printf '\n'
    if (( print_only )); then
        continue
    fi
    if "${command[@]}" 2>&1 | tee -- "${log_dir}/${window}.log"; then
        printf '[V7.10 %s] completed at %s\n' "$window" "$(date --iso-8601=seconds)"
    else
        return_code=$?
        printf '[V7.10 %s] failed (exit %s); pipeline stopped. Log: %s/%s.log\n' \
            "$window" "$return_code" "$log_dir" "$window" >&2
        exit "$return_code"
    fi
done

if (( ! print_only )); then
    printf '\nAll V7.10 runs completed: 0s -> 1s -> 2s. Logs: %s\n' "$log_dir"
fi
