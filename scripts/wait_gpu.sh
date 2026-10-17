#!/usr/bin/env bash
# Poll one or all GPUs, then replace this process with the requested command.
set -euo pipefail

GPU=0
MAX_MEMORY=500
INTERVAL=1

usage() {
    cat <<'EOF'
用法：bash scripts/wait_gpu.sh [选项] -- 命令 [参数...]

  -g, --gpu ID          GPU 编号、完整 UUID 或 all（检查所有卡；默认：0）
  --max-memory MIB      显存占用低于此值时启动（默认：500 MiB）
  --interval SECONDS   检查间隔，正整数秒（默认：1）
  -h, --help           显示帮助

示例：
  bash scripts/wait_gpu.sh --gpu 0 -- python scripts/train.py --batch-size 8
  bash scripts/wait_gpu.sh --gpu all -- python scripts/train.py --batch-size 8
  nohup bash scripts/wait_gpu.sh --gpu 1 -- python scripts/train.py > train.log 2>&1 &

命令使用当前目录和环境；请先激活训练所需的 Python 环境。
--gpu all 会选择查询结果中第一张满足条件的 GPU，只启动一次命令。
启动时通过 CUDA_VISIBLE_DEVICES 将选中的卡设为程序内的 cuda:0。
启动条件只依据显存占用，不会预留 GPU；其他等待者仍可能同时启动。
命令执行一次，退出后不会自动重试。Ctrl+C 可取消前台等待。
EOF
}

die() { printf '错误：%s\n' "$*" >&2; exit 2; }
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }

while (($#)); do
    case "$1" in
        -g|--gpu|--max-memory|--interval)
            (($# >= 2)) || die "$1 缺少参数"
            case "$1" in
                -g|--gpu) GPU="$2" ;;
                --max-memory) MAX_MEMORY="$2" ;;
                --interval) INTERVAL="$2" ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) die "未知选项 $1；请用 -- 分隔训练命令，或查看 --help" ;;
    esac
done

(($#)) || die '请在 -- 后提供要执行的命令'
[[ "$GPU" == all || "$GPU" =~ ^[0-9]+$ || "$GPU" =~ ^GPU-[[:xdigit:]-]+$ ]] || die 'GPU 必须是 all、单个编号或完整 GPU UUID'
[[ "$MAX_MEMORY" =~ ^[0-9]{1,9}$ ]] || die '--max-memory 必须是正整数 MiB'
[[ "$INTERVAL" =~ ^[0-9]{1,9}$ ]] || die '--interval 必须是正整数秒'
MAX_MEMORY=$((10#$MAX_MEMORY))
INTERVAL=$((10#$INTERVAL))
((MAX_MEMORY > 0)) || die '--max-memory 必须大于 0'
((INTERVAL > 0)) || die '--interval 必须大于 0'
command -v nvidia-smi >/dev/null 2>&1 || die '找不到 nvidia-smi，请在 GPU 机器或容器中运行'
command -v "$1" >/dev/null 2>&1 || die "找不到要执行的命令：$1"

QUERY_ARGS=()
if [[ "$GPU" != all ]]; then
    QUERY_ARGS+=(-i "$GPU")
fi
QUERY_ARGS+=(--query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)

log "等待 GPU $GPU：显存占用 < ${MAX_MEMORY} MiB；每 ${INTERVAL} 秒检查"
printf '待执行命令：' >&2
printf '%q ' "$@" >&2
printf '\n' >&2
LAST_STATUS=''
LAST_LOG_SECONDS=-30
FIRST_QUERY=true

while true; do
    if ROWS=$(nvidia-smi "${QUERY_ARGS[@]}" 2>&1); then
        SUMMARY=()
        while IFS=, read -r INDEX GPU_UUID USED EXTRA; do
            INDEX="${INDEX//[[:space:]]/}"
            GPU_UUID="${GPU_UUID//[[:space:]]/}"
            USED="${USED//[[:space:]]/}"
            if [[ "$INDEX" =~ ^[0-9]+$ && "$GPU_UUID" =~ ^GPU-[[:xdigit:]-]+$ && "$USED" =~ ^[0-9]{1,9}$ && -z "$EXTRA" ]]; then
                USED=$((10#$USED))
                if ((USED < MAX_MEMORY)); then
                    log "GPU $INDEX ($GPU_UUID) 已满足条件（${USED} MiB），开始执行"
                    # UUIDs avoid differences between CUDA and nvidia-smi ordering.
                    export CUDA_VISIBLE_DEVICES="$GPU_UUID"
                    exec "$@"
                fi
                SUMMARY+=("GPU $INDEX: ${USED} MiB;")
            else
                SUMMARY+=('显存查询返回无效数据;')
            fi
        done <<< "$ROWS"
        STATUS="${SUMMARY[*]} 继续等待"
    else
        if [[ "$FIRST_QUERY" == true ]]; then
            die "无法查询 GPU $GPU，请检查 GPU 编号、驱动和容器 GPU 权限：$ROWS"
        fi
        STATUS="GPU 查询失败，继续等待：$ROWS"
    fi
    FIRST_QUERY=false

    # Report status changes immediately; otherwise emit a heartbeat every 30s.
    if [[ "$STATUS" != "$LAST_STATUS" ]] || ((SECONDS - LAST_LOG_SECONDS >= 30)); then
        log "$STATUS"
        LAST_STATUS="$STATUS"
        LAST_LOG_SECONDS=$SECONDS
    fi
    sleep "$INTERVAL"
done
