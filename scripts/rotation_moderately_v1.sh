#!/usr/bin/env bash
# Rotation-only tactile VLA training and inference entry point.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TRAIN_PYTHON="${VLA_TRAIN_PYTHON:-${PROJECT_ROOT}/openpi/.venv/bin/python}"
RUNTIME_PYTHON="${VLA_RUNTIME_PYTHON:-python3}"

DATA_PROFILE="rotation_moderately_success_v1"
PROMPT_PROFILE="minimal_v1"
DATASET_DIR="/data1/tac_data/lerobot_data/tactile_vla_v3"
PROFILE_DIR="/data1/outputs/vla/${DATA_PROFILE}"
INDEX_FILE="${PROFILE_DIR}/vla_indices_v3.json"
SPLIT_FILE="${PROFILE_DIR}/splits.json"
REASONING_DIR="${PROFILE_DIR}/reasoning"
NORM_DIR="/data1/outputs/vla/assets/tactile_vla_rotation_moderately_success_v1"
BASE_CHECKPOINT="/home/test/.cache/modelscope/hub/models/hairuoliu/pi05_base/params"
STAGE_A_ROOT="/data1/outputs/vla/stage_a_action"
STAGE_A_RUN_NAME="pi05_delta_tac_rotation_moderately_v1"
STAGE_A_RUN="${STAGE_A_ROOT}/${STAGE_A_RUN_NAME}"
STAGE_A_STEP="${STAGE_A_RUN}/10000"
STAGE_B_ROOT="/data1/outputs/vla/stage_b_v3"
STAGE_B_RUN_NAME="pi05_stage_b_rotation_moderately_v1"
STAGE_B_RUN="${STAGE_B_ROOT}/${STAGE_B_RUN_NAME}"
MERGED_CHECKPOINT="${STAGE_B_RUN}/merged_best"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/.cache/huggingface/datasets}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data1/outputs/openpi_cache}"

usage() {
    cat <<'EOF'
Usage:
  scripts/rotation_moderately_v1.sh <command> [extra arguments]

Preparation:
  prepare             Build the immutable split, V3 index, manifests and hashes.
  norm                Recompute train-only delta-action normalization statistics.
  preflight           Validate Python, data, profile, norm stats and accelerator.
  print-config        Print all resolved paths and versioned profiles.

Training:
  stage-a-dry         Inspect one Stage A batch and its minimal_v1 prompt.
  stage-a             Train Stage A for 10,000 total updates from pi05_base.
  stage-b-dry         Inspect action/need/failure/plan Stage B batches.
  stage-b             Train Stage B for 4,000 total updates from Stage A step 10000.
  merge               Merge the selected best Stage B delta with Stage A step 10000.

Inference:
  warmup              Load merged_best and warm up all server inference paths.
  serve               Serve the merged closed-loop policy on port 8000.
  client              Run the robot closed-loop client; extra arguments are forwarded.
  serve-stage-a       Serve Stage A for forced-recovery action-only comparison.
  serve-stage-b       Serve merged Stage B for action-only comparison.
  ablation-client     Run the manual NORMAL -> RECOVERY comparison client.

Examples:
  scripts/rotation_moderately_v1.sh prepare --overwrite
  scripts/rotation_moderately_v1.sh norm
  scripts/rotation_moderately_v1.sh stage-a
  scripts/rotation_moderately_v1.sh stage-b
  scripts/rotation_moderately_v1.sh merge
  scripts/rotation_moderately_v1.sh serve --host 0.0.0.0 --port 8000
  scripts/rotation_moderately_v1.sh client --host TRAINING_HOST --port 8000
  scripts/rotation_moderately_v1.sh ablation-client --host TRAINING_HOST --port 8000 \
    --rotation-direction right --noise-seed 42

Resume explicitly when needed:
  scripts/rotation_moderately_v1.sh stage-a --resume
  scripts/rotation_moderately_v1.sh stage-b --resume

Environment overrides:
  VLA_TRAIN_PYTHON       OpenPI training/server Python executable.
  VLA_RUNTIME_PYTHON     ROS client Python executable (default: python3).
  VLA_ALLOW_CPU_TRAINING Set to 1 only for intentional CPU training.
EOF
}

run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Missing required file: %s\n' "$1" >&2
        exit 2
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        printf 'Missing required directory: %s\n' "$1" >&2
        exit 2
    fi
}

check_train_python() {
    if [[ ! -x "${TRAIN_PYTHON}" ]]; then
        printf 'Training Python is not executable: %s\n' "${TRAIN_PYTHON}" >&2
        printf 'Set VLA_TRAIN_PYTHON to the OpenPI environment Python.\n' >&2
        exit 2
    fi
}

check_profile() {
    require_dir "${DATASET_DIR}"
    require_file "${PROFILE_DIR}/profile.json"
    require_file "${PROFILE_DIR}/artifact_manifest.json"
    require_file "${PROFILE_DIR}/action_frame_manifest.json"
    require_file "${INDEX_FILE}"
    require_file "${SPLIT_FILE}"
    require_file "${REASONING_DIR}/train.jsonl"
    require_file "${REASONING_DIR}/val.jsonl"
    require_file "${REASONING_DIR}/test.jsonl"
}

check_norm() {
    require_file "${NORM_DIR}/norm_stats.json"
    require_file "${NORM_DIR}/summary.json"
}

verify_artifacts() {
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${TRAIN_PYTHON}" -c '
import json
import sys
from tactile_vla.vla.artifacts import artifact_identity, validate_norm_stats_identity

index_path, norm_summary = sys.argv[1:]
index = json.load(open(index_path, encoding="utf-8"))
identity = artifact_identity(
    index,
    index_path=index_path,
    prompt_profile="minimal_v1",
    requested_data_profile="rotation_moderately_success_v1",
)
validate_norm_stats_identity(norm_summary, identity, context="pipeline preflight")
counts = identity["action_indices_identity"]
if counts["all"]["count"] != 98233 or counts["train"]["count"] != 80619:
    raise ValueError(f"Unexpected action index counts: {counts}")
print("Artifact identity: OK")
print(f"index_sha256={identity['"'"'index_sha256'"'"']}")
print(f"action_frame_manifest_hash={identity['"'"'action_frame_manifest_hash'"'"']}")
' "${INDEX_FILE}" "${NORM_DIR}/summary.json"
}

check_accelerator() {
    local backend
    backend="$("${TRAIN_PYTHON}" -c 'import jax; print(jax.default_backend())')"
    printf 'JAX backend: %s\n' "${backend}"
    if [[ "${backend}" == "cpu" && "${VLA_ALLOW_CPU_TRAINING:-0}" != "1" ]]; then
        printf 'Refusing long training on CPU. Fix the GPU/TPU environment or set VLA_ALLOW_CPU_TRAINING=1.\n' >&2
        exit 2
    fi
}

print_config() {
    cat <<EOF
project_root=${PROJECT_ROOT}
train_python=${TRAIN_PYTHON}
runtime_python=${RUNTIME_PYTHON}
data_profile=${DATA_PROFILE}
prompt_profile=${PROMPT_PROFILE}
dataset_dir=${DATASET_DIR}
profile_dir=${PROFILE_DIR}
index_file=${INDEX_FILE}
split_file=${SPLIT_FILE}
reasoning_dir=${REASONING_DIR}
norm_dir=${NORM_DIR}
base_checkpoint=${BASE_CHECKPOINT}
stage_a_run=${STAGE_A_RUN}
stage_a_step=${STAGE_A_STEP}
stage_b_run=${STAGE_B_RUN}
merged_checkpoint=${MERGED_CHECKPOINT}
EOF
}

prepare() {
    check_train_python
    run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/prepare_rotation_moderately_profile.py" \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${PROFILE_DIR}" \
        --seed 42 \
        --action-horizon 30 \
        --reasoning-window-frames 15 \
        --status-negative-ratio 3 \
        "$@"
}

norm() {
    check_train_python
    check_profile
    run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/compute_vla_norm_stats.py" \
        --dataset-dir "${DATASET_DIR}" \
        --data-profile "${DATA_PROFILE}" \
        --index-file "${INDEX_FILE}" \
        --split-file "${SPLIT_FILE}" \
        --output-dir "${NORM_DIR}" \
        --seed 42 \
        --action-horizon 30 \
        --delta-action-dims 7 \
        "$@"
}

stage_a_common() {
    check_train_python
    check_profile
    check_norm
    run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/train_vla_stage_a_openpi.py" \
        --dataset-dir "${DATASET_DIR}" \
        --data-profile "${DATA_PROFILE}" \
        --prompt-profile "${PROMPT_PROFILE}" \
        --index-file "${INDEX_FILE}" \
        --split-file "${SPLIT_FILE}" \
        --norm-stats-dir "${NORM_DIR}" \
        --output-dir "${STAGE_A_ROOT}" \
        --run-name "${STAGE_A_RUN_NAME}" \
        --checkpoint "${BASE_CHECKPOINT}" \
        --batch-size 8 \
        --num-steps 10000 \
        --lr 5e-5 \
        --lr-final 5e-7 \
        --lr-transition-steps 7000 \
        --save-interval 1000 \
        --keep-period 5000 \
        --action-horizon 30 \
        --action-dim 32 \
        --state-history-len 60 \
        --state-history-dim 7 \
        --history-hidden-dim 256 \
        --use-state-history \
        --train-lora-only \
        "$@"
}

stage_b_common() {
    check_train_python
    check_profile
    check_norm
    require_dir "${STAGE_A_STEP}"
    run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/train_vla_stage_b_v3.py" \
        --dataset-dir "${DATASET_DIR}" \
        --data-profile "${DATA_PROFILE}" \
        --prompt-profile "${PROMPT_PROFILE}" \
        --grammar-profile v3_full_v1 \
        --index-file "${INDEX_FILE}" \
        --split-file "${SPLIT_FILE}" \
        --reasoning-manifest-dir "${REASONING_DIR}" \
        --norm-stats-dir "${NORM_DIR}" \
        --stage-a-checkpoint "${STAGE_A_STEP}" \
        --output-dir "${STAGE_B_ROOT}" \
        --run-name "${STAGE_B_RUN_NAME}" \
        --batch-size 8 \
        --num-steps 4000 \
        --lr 1e-4 \
        --eval-interval 500 \
        --save-interval 1000 \
        --keep-period 1000 \
        --action-loss-degradation-limit 0.10 \
        --reasoning-window-frames 15 \
        --status-negative-ratio 3 \
        --action-horizon 30 \
        --action-dim 32 \
        --state-history-len 60 \
        --state-history-dim 7 \
        --history-hidden-dim 256 \
        --use-state-history \
        "$@"
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${command}" in
    help|-h|--help)
        usage
        ;;
    print-config)
        print_config
        ;;
    prepare)
        prepare "$@"
        ;;
    norm)
        norm "$@"
        ;;
    preflight)
        check_train_python
        require_dir "${BASE_CHECKPOINT}"
        check_profile
        check_norm
        verify_artifacts
        print_config
        check_accelerator
        printf 'Preflight passed.\n'
        ;;
    stage-a-dry)
        stage_a_common --dry-run --num-workers 0 --max-frames 8 "$@"
        ;;
    stage-a)
        require_dir "${BASE_CHECKPOINT}"
        check_accelerator
        stage_a_common "$@"
        ;;
    stage-b-dry)
        # Dataset/prompt dry-run intentionally does not load model weights.
        check_train_python
        check_profile
        check_norm
        run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/train_vla_stage_b_v3.py" \
            --dataset-dir "${DATASET_DIR}" \
            --data-profile "${DATA_PROFILE}" \
            --prompt-profile "${PROMPT_PROFILE}" \
            --grammar-profile v3_full_v1 \
            --index-file "${INDEX_FILE}" \
            --split-file "${SPLIT_FILE}" \
            --reasoning-manifest-dir "${REASONING_DIR}" \
            --norm-stats-dir "${NORM_DIR}" \
            --batch-size 8 \
            --num-steps 4000 \
            --eval-interval 500 \
            --save-interval 1000 \
            --keep-period 1000 \
            --dry-run --num-workers 0 "$@"
        ;;
    stage-b)
        check_accelerator
        stage_b_common "$@"
        ;;
    merge)
        check_train_python
        require_dir "${STAGE_A_STEP}"
        require_dir "${STAGE_B_RUN}/best/delta_params"
        run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/merge_v3_stage_b_delta.py" \
            --stage-a-checkpoint "${STAGE_A_STEP}" \
            --stage-b-delta "${STAGE_B_RUN}/best" \
            --output "${MERGED_CHECKPOINT}" \
            "$@"
        ;;
    warmup)
        check_train_python
        check_norm
        require_dir "${MERGED_CHECKPOINT}/params"
        run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/serve_tactile_vla_policy_v3.py" \
            --checkpoint "${MERGED_CHECKPOINT}" \
            --norm-stats-dir "${NORM_DIR}" \
            --dry-run "$@"
        ;;
    serve)
        check_train_python
        check_norm
        require_dir "${MERGED_CHECKPOINT}/params"
        run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/serve_tactile_vla_policy_v3.py" \
            --checkpoint "${MERGED_CHECKPOINT}" \
            --norm-stats-dir "${NORM_DIR}" \
            --host 0.0.0.0 --port 8000 "$@"
        ;;
    client)
        run "${RUNTIME_PYTHON}" \
            "${PROJECT_ROOT}/openpi/inference/agilex/inference/agilex_inference_tactile_vla_sync_single.py" \
            "$@"
        ;;
    serve-stage-a)
        check_train_python
        check_norm
        require_dir "${STAGE_A_STEP}"
        run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/serve_tactile_vla_action_ablation.py" \
            --checkpoint-kind stage-a \
            --checkpoint "${STAGE_A_STEP}" \
            --norm-stats-dir "${NORM_DIR}" \
            --host 0.0.0.0 --port 8000 "$@"
        ;;
    serve-stage-b)
        check_train_python
        check_norm
        require_dir "${MERGED_CHECKPOINT}/params"
        run "${TRAIN_PYTHON}" "${SCRIPT_DIR}/serve_tactile_vla_action_ablation.py" \
            --checkpoint-kind stage-b \
            --checkpoint "${MERGED_CHECKPOINT}" \
            --norm-stats-dir "${NORM_DIR}" \
            --host 0.0.0.0 --port 8000 "$@"
        ;;
    ablation-client)
        run "${RUNTIME_PYTHON}" \
            "${PROJECT_ROOT}/openpi/inference/agilex/inference/agilex_inference_forced_recovery_ablation.py" \
            "$@"
        ;;
    *)
        printf 'Unknown command: %s\n\n' "${command}" >&2
        usage >&2
        exit 2
        ;;
esac
