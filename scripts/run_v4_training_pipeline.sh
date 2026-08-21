#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VLA_PYTHON="${VLA_PYTHON:-/home/test/anaconda3/envs/openpi/bin/python}"
DATASET_DIR="/data1/tac_data/lerobot_data/tactile_vla_rotation_v4"
PROFILE_DIR="/data1/outputs/vla/rotation_v4"
TRAINING_INDEX_DIR=""
REASONING_DIR=""
NORM_STATS_DIR="/data1/outputs/vla/assets/tactile_vla_rotation_v4"
BASE_CHECKPOINT="/home/test/.cache/modelscope/hub/models/hairuoliu/pi05_base/params"
STAGE_A_CHECKPOINT="/data1/outputs/vla/stage_a_action/pi05_delta_tac_rotation_v4/10000"
STAGE_A_OUTPUT_DIR="/data1/outputs/vla/stage_a_action"
STAGE_B_OUTPUT_DIR="/data1/outputs/vla/stage_b_v4"
STAGE_A_RUN_NAME="pi05_delta_tac_rotation_v4"
STAGE_B_RUN_NAME="pi05_stage_b_rotation_v4"
FROM_STEP="training-index"
TO_STEP="stage-b-dry-run"
EXECUTE=false
OVERWRITE_INDEX=false

usage() {
    cat <<'EOF'
Usage: run_v4_training_pipeline.sh [options]

Continue after run_v4_data_pipeline.sh. The safe default only prints shell-quoted
commands. Pass --execute to build the training index/norm stats and run the Stage
A one-batch plus Stage B four-batch dry-runs. This script never starts training.

Steps: training-index, norm, stage-a-dry-run, stage-b-dry-run

Options:
  --execute                    Execute selected steps (default: print only)
  --dry-run                    Print selected commands only
  --from-step NAME             Start at NAME
  --to-step NAME               Stop after NAME
  --vla-python PATH            Python executable for myvla2
  --dataset-dir PATH           V4 LeRobot dataset
  --profile-dir PATH           V4 profile/artifact root
  --training-index-dir PATH    Index and need output directory (default: profile root)
  --reasoning-dir PATH         Reasoning manifest directory (default: profile/reasoning_manifests)
  --norm-stats-dir PATH        V4 norm stats output directory
  --base-checkpoint PATH       Raw pi05_base checkpoint (dry-run records the argument only)
  --stage-a-checkpoint PATH    Future Stage A step 10000 checkpoint for Stage B
  --stage-a-output-dir PATH    Stage A output root (dry-run does not write it)
  --stage-b-output-dir PATH    Stage B output root (dry-run does not write it)
  --stage-a-run-name NAME
  --stage-b-run-name NAME
  --overwrite-index            Allow replacing an existing V4 training index/need manifests
  -h, --help
EOF
}

require_value() {
    local option="$1"
    local count="$2"
    if ((count < 2)); then
        printf 'Missing value for %s\n' "${option}" >&2
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --execute) EXECUTE=true; shift ;;
        --dry-run) EXECUTE=false; shift ;;
        --from-step) require_value "$1" "$#"; FROM_STEP="$2"; shift 2 ;;
        --to-step) require_value "$1" "$#"; TO_STEP="$2"; shift 2 ;;
        --vla-python) require_value "$1" "$#"; VLA_PYTHON="$2"; shift 2 ;;
        --dataset-dir) require_value "$1" "$#"; DATASET_DIR="$2"; shift 2 ;;
        --profile-dir) require_value "$1" "$#"; PROFILE_DIR="$2"; shift 2 ;;
        --training-index-dir) require_value "$1" "$#"; TRAINING_INDEX_DIR="$2"; shift 2 ;;
        --reasoning-dir) require_value "$1" "$#"; REASONING_DIR="$2"; shift 2 ;;
        --norm-stats-dir) require_value "$1" "$#"; NORM_STATS_DIR="$2"; shift 2 ;;
        --base-checkpoint) require_value "$1" "$#"; BASE_CHECKPOINT="$2"; shift 2 ;;
        --stage-a-checkpoint) require_value "$1" "$#"; STAGE_A_CHECKPOINT="$2"; shift 2 ;;
        --stage-a-output-dir) require_value "$1" "$#"; STAGE_A_OUTPUT_DIR="$2"; shift 2 ;;
        --stage-b-output-dir) require_value "$1" "$#"; STAGE_B_OUTPUT_DIR="$2"; shift 2 ;;
        --stage-a-run-name) require_value "$1" "$#"; STAGE_A_RUN_NAME="$2"; shift 2 ;;
        --stage-b-run-name) require_value "$1" "$#"; STAGE_B_RUN_NAME="$2"; shift 2 ;;
        --overwrite-index) OVERWRITE_INDEX=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

step_index() {
    case "$1" in
        training-index) printf '1' ;;
        norm) printf '2' ;;
        stage-a-dry-run) printf '3' ;;
        stage-b-dry-run) printf '4' ;;
        *) printf 'Invalid step: %s\n' "$1" >&2; return 1 ;;
    esac
}

FROM_INDEX="$(step_index "${FROM_STEP}")"
TO_INDEX="$(step_index "${TO_STEP}")"
if ((FROM_INDEX > TO_INDEX)); then
    printf -- '--from-step must not be after --to-step\n' >&2
    exit 2
fi

if [[ -z "${TRAINING_INDEX_DIR}" ]]; then
    TRAINING_INDEX_DIR="${PROFILE_DIR}"
fi
if [[ -z "${REASONING_DIR}" ]]; then
    REASONING_DIR="${PROFILE_DIR}/reasoning_manifests"
fi

SELECTION_FILE="${PROFILE_DIR}/selection.json"
PROFILE_FILE="${PROFILE_DIR}/profile.json"
SPLIT_FILE="${PROFILE_DIR}/splits.json"
ACTION_MANIFEST_FILE="${PROFILE_DIR}/action_frame_manifest.jsonl"
INDEX_FILE="${TRAINING_INDEX_DIR}/v4_training_index.json"

selected_step() {
    local index="$1"
    ((index >= FROM_INDEX && index <= TO_INDEX))
}

run_step() {
    local index="$1"
    local name="$2"
    shift 2
    if ! selected_step "${index}"; then
        return 0
    fi
    printf '[%s] ' "${name}"
    printf '%q ' "$@"
    printf '\n'
    if [[ "${EXECUTE}" == true ]]; then
        "$@"
    fi
}

printf 'mode=%s from=%s to=%s\n' \
    "$([[ "${EXECUTE}" == true ]] && printf execute || printf dry-run)" \
    "${FROM_STEP}" "${TO_STEP}"
printf 'note=%s\n' 'Stage A and Stage B commands are batch dry-runs only; no training is started.'

training_index_cmd=(
    "${VLA_PYTHON}" "${SCRIPT_DIR}/prepare_v4_training_index.py"
    --dataset-dir "${DATASET_DIR}"
    --selection-file "${SELECTION_FILE}"
    --profile-file "${PROFILE_FILE}"
    --split-file "${SPLIT_FILE}"
    --action-manifest-file "${ACTION_MANIFEST_FILE}"
    --reasoning-manifest-dir "${REASONING_DIR}"
    --output-dir "${TRAINING_INDEX_DIR}"
    --seed 42
    --status-negative-ratio 3.0
    --failure-window-length 15
)
if [[ "${OVERWRITE_INDEX}" == true ]]; then
    training_index_cmd+=(--overwrite)
fi
run_step 1 training-index "${training_index_cmd[@]}"

norm_cmd=(
    "${VLA_PYTHON}" "${SCRIPT_DIR}/compute_vla_norm_stats.py"
    --dataset-dir "${DATASET_DIR}"
    --split-file "${SPLIT_FILE}"
    --index-file "${INDEX_FILE}"
    --data-profile rotation_v4
    --output-dir "${NORM_STATS_DIR}"
    --seed 42
    --action-horizon 30
    --delta-action-dims 7
)
run_step 2 norm "${norm_cmd[@]}"

stage_a_dry_run_cmd=(
    "${VLA_PYTHON}" "${SCRIPT_DIR}/train_vla_stage_a_openpi.py"
    --dataset-dir "${DATASET_DIR}"
    --data-profile rotation_v4
    --prompt-profile minimal_v1
    --index-file "${INDEX_FILE}"
    --split-file "${SPLIT_FILE}"
    --norm-stats-dir "${NORM_STATS_DIR}"
    --output-dir "${STAGE_A_OUTPUT_DIR}"
    --run-name "${STAGE_A_RUN_NAME}"
    --checkpoint "${BASE_CHECKPOINT}"
    --batch-size 8
    --num-workers 0
    --action-horizon 30
    --action-dim 32
    --state-history-len 60
    --state-history-dim 7
    --history-hidden-dim 256
    --use-state-history
    --train-lora-only
    --dry-run
)
run_step 3 stage-a-dry-run "${stage_a_dry_run_cmd[@]}"

stage_b_dry_run_cmd=(
    "${VLA_PYTHON}" "${SCRIPT_DIR}/train_vla_stage_b_v3.py"
    --dataset-dir "${DATASET_DIR}"
    --data-profile rotation_v4
    --prompt-profile minimal_v1
    --grammar-profile v3_full_v1
    --index-file "${INDEX_FILE}"
    --split-file "${SPLIT_FILE}"
    --reasoning-manifest-dir "${REASONING_DIR}"
    --norm-stats-dir "${NORM_STATS_DIR}"
    --stage-a-checkpoint "${STAGE_A_CHECKPOINT}"
    --output-dir "${STAGE_B_OUTPUT_DIR}"
    --run-name "${STAGE_B_RUN_NAME}"
    --batch-size 8
    --num-workers 0
    --action-horizon 30
    --action-dim 32
    --state-history-len 60
    --state-history-dim 7
    --history-hidden-dim 256
    --use-state-history
    --reasoning-window-frames 15
    --status-negative-ratio 3.0
    --dry-run
)
run_step 4 stage-b-dry-run "${stage_b_dry_run_cmd[@]}"
