#!/usr/bin/env bash
set -euo pipefail

# Public GPT-L / NAR-XL ImageNet-256 300-epoch recipe.
# Usage: bash train_c2i_mtarpp.sh FEATURE_DIR TEACHER_CKPT OUTPUT_DIR [TRAIN_ARGS...]
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if (( $# < 3 )); then
    echo "Usage: $0 FEATURE_DIR TEACHER_CKPT OUTPUT_DIR [TRAIN_ARGS...]" >&2
    exit 2
fi
FEATURE_DIR=$1
TEACHER_CKPT=$2
OUTPUT_DIR=$3
shift 3
for required in "$TEACHER_CKPT" "$FEATURE_DIR/imagenet256_codes" \
    "$FEATURE_DIR/imagenet256_labels" "$FEATURE_DIR/imagenet256_scores"; do
    [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
# Resolve paths before switching to the repository root (including spaces).
FEATURE_DIR=$(cd "$FEATURE_DIR" && pwd)
TEACHER_CKPT=$(cd "$(dirname "$TEACHER_CKPT")" && pwd)/$(basename "$TEACHER_CKPT")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#GPU_IDS[@]}}
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
COMMAND=("${TORCHRUN:-torchrun}" --nnodes=1 --nproc_per_node="$NPROC_PER_NODE"
    --master_port="${MASTER_PORT:-29502}" --module mtarpp.train
    --code-path "$FEATURE_DIR" --results-dir "$OUTPUT_DIR/logs"
    --cloud-save-path "$OUTPUT_DIR/checkpoints" --no-local-save
    --dataset imagenet_code --gpt-model GPT-L --gpt-type c2i
    --teacher-model NAR-XL --teacher-ckpt "$TEACHER_CKPT"
    --student-target-layers 24 --teacher-target-layers 36 --projector-dim 2048
    --distill-weight 1.0 --future-feature-pred-layer 24
    --future-feature-offsets 1,16 --future-feature-loss-weights 1.0,1.0
    --loss-weight-contrast 0.2 --contrast-temp 0.07
    --contrast-dropout-p 0.2 --contrast-num-samples 2048
    --max-student-drop-rate 0.5 --student-drop-switch-ratio 0.8
    --max-physical-drop-rate 0.5 --physical-drop-switch-ratio 1.0
    --semantic-drop-temperature 0.35
    --image-size 256 --downsample-size 16 --vocab-size 16384
    --num-classes 1000 --cls-token-num 1 --max-seq-len 257
    --dropout-p 0.1 --token-dropout-p 0.1 --drop-path-rate 0.0
    --epochs 300 --global-batch-size 256 --lr 1e-4 --weight-decay 0.05
    --beta1 0.9 --beta2 0.95 --max-grad-norm 1.0
    --mixed-precision bf16 --global-seed 0 --num-workers 24 --log-every 100
    --checkpoint-steps 1200000,1250000,1300000,1350000,1400000,1450000,1500000
    --debug-first-steps 3 --ema "$@")
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
[[ ${DRY_RUN:-0} == 1 ]] && exit 0
cd "$PROJECT_ROOT"
"${COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"
