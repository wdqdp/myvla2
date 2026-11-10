#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${NO_PLAN_PYTHON:-${PROJECT_DIR}/openpi/.venv/bin/python}"
OUTPUT_ROOT="${NO_PLAN_OUTPUT_ROOT:-/data1/qxh/tac_vla_new/tac_data/demon_data}"
TASK="${1:-}"
if [[ -n "$TASK" ]]; then shift; fi
case "$TASK" in
    book) VERSION=v9_7 ;;
    sponge) VERSION=v10_3 ;;
    bottle) VERSION=v11_4 ;;
    *) printf 'Usage: %s {book|sponge|bottle} [--gpus 0,1] [--fsdp 2] [--dry-run]\n' "$0" >&2; exit 2 ;;
esac

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
FSDP=1
DRY_RUN=false
usage() {
    cat <<USAGE
Usage: $(basename "$0") $TASK [--gpus GPU_IDS] [--fsdp N] [--dry-run]

  --gpus IDS   Comma-separated physical GPU IDs (default: CUDA_VISIBLE_DEVICES or 0,1,2,3)
  --fsdp N     FSDP devices for both stages (default: 1)
  --dry-run    Print commands and paths without running training
  -h, --help   Show this help

The pipeline trains Stage A for 10000 steps, verifies its checkpoint, then starts Stage B.
NO_PLAN_PYTHON overrides the Python executable; NO_PLAN_OUTPUT_ROOT overrides the model-output root.
USAGE
}
while (($#)); do
    case "$1" in
        --gpus|--fsdp)
            if (($# < 2)) || [[ "$2" == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2; exit 2
            fi
            if [[ "$1" == --gpus ]]; then GPUS="$2"; else FSDP="$2"; fi
            shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'GPU IDs must be comma-separated nonnegative integers: %s\n' "$GPUS" >&2; exit 2
fi
IFS=, read -r -a GPU_ARRAY <<< "$GPUS"
declare -A GPU_SEEN=()
for gpu in "${GPU_ARRAY[@]}"; do
    normalized_gpu=$((10#$gpu))
    if [[ -n "${GPU_SEEN[$normalized_gpu]+present}" ]]; then
        printf 'Duplicate GPU ID: %s\n' "$gpu" >&2; exit 2
    fi
    GPU_SEEN[$normalized_gpu]=1
done
if [[ ! "$FSDP" =~ ^[0-9]+$ ]] || ((10#$FSDP < 1)); then
    printf 'FSDP must be a positive integer: %s\n' "$FSDP" >&2; exit 2
fi
FSDP=$((10#$FSDP))
GPU_COUNT=${#GPU_ARRAY[@]}
if ((GPU_COUNT % FSDP != 0 || 8 % GPU_COUNT != 0)); then
    printf 'GPU count %d must divide batch size 8 and be divisible by FSDP %d.\n' "$GPU_COUNT" "$FSDP" >&2; exit 2
fi

STAGE_A_OUTPUT="${OUTPUT_ROOT}/${TASK}/outputs/stage_a_action"
STAGE_A_RUN="pi05_${TASK}_${VERSION}_stage_a_direct_tactile"
STAGE_A_CHECKPOINT="${STAGE_A_OUTPUT}/${STAGE_A_RUN}/10000"
STAGE_B_OUTPUT="${OUTPUT_ROOT}/${TASK}/outputs/multitask_${VERSION}"
STAGE_B_RUN="pi05_${TASK}_${VERSION}_direct_tactile_h100"
STAGE_A_CMD=("$PYTHON_BIN" "${SCRIPT_DIR}/train_vla_stage_a_no_plan.py" --task "$TASK"
             --batch-size 8 --fsdp-devices "$FSDP" --output-dir "$STAGE_A_OUTPUT" --run-name "$STAGE_A_RUN")
STAGE_B_CMD=("$PYTHON_BIN" "${SCRIPT_DIR}/train_vla_multitask_no_plan.py" --task "$TASK"
             --batch-size 8 --fsdp-devices "$FSDP" --output-dir "$STAGE_B_OUTPUT"
             --run-name "$STAGE_B_RUN" --stage-a-checkpoint "$STAGE_A_CHECKPOINT")
export CUDA_VISIBLE_DEVICES="$GPUS"
export USE_TF=0
cd -- "$PROJECT_DIR"
printf 'Task: %s %s; GPUs: %s; FSDP: %s\n' "$TASK" "$VERSION" "$GPUS" "$FSDP"
printf 'Stage A: '; printf '%q ' "${STAGE_A_CMD[@]}"; printf '\n'
printf 'Stage B: '; printf '%q ' "${STAGE_B_CMD[@]}"; printf '\n'
if [[ "$DRY_RUN" == true ]]; then exit 0; fi
if [[ ! -x "$PYTHON_BIN" ]]; then printf 'Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 1; fi
"${STAGE_A_CMD[@]}"
if [[ ! -f "${STAGE_A_CHECKPOINT}/params/_METADATA" || ! -f "${STAGE_A_OUTPUT}/${STAGE_A_RUN}/config.json" ]]; then
    printf 'Stage A finished but its checkpoint or config is missing: %s\n' "$STAGE_A_CHECKPOINT" >&2; exit 1
fi
printf 'Stage A finished. Starting Stage B from %s\n' "$STAGE_A_CHECKPOINT"
"${STAGE_B_CMD[@]}"
