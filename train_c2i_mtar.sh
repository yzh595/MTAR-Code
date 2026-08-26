#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_PATH=${1:-${CODE_PATH:-}}
RESULTS_DIR=${2:-${RESULTS_DIR:-"$PROJECT_ROOT/results"}}

if [[ -z "$CODE_PATH" ]]; then
    echo "Usage: $0 CODE_PATH [RESULTS_DIR]" >&2
    exit 1
fi

: "${TORCHRUN:=torchrun}"
: "${CUDA_DEVICES:=0}"
: "${TRAIN_MASTER_PORT:=33842}"

TRAIN_SCRIPT="$PROJECT_ROOT/autoregressive/train/train_c2i_MTAR.py"
[[ -s "$TRAIN_SCRIPT" ]] || { echo "Missing $TRAIN_SCRIPT" >&2; exit 1; }
[[ -d "$CODE_PATH/imagenet256_codes" ]] || { echo "Missing token directory" >&2; exit 1; }
[[ -d "$CODE_PATH/imagenet256_labels" ]] || { echo "Missing label directory" >&2; exit 1; }
[[ -d "$CODE_PATH/imagenet256_scores" ]] || { echo "Missing semantic-score directory" >&2; exit 1; }

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
mkdir -p "$RESULTS_DIR"
cd "$PROJECT_ROOT"

"$TORCHRUN" \
    --nproc_per_node=1 \
    --master_port="$TRAIN_MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --code-path "$CODE_PATH" \
    --results-dir "$RESULTS_DIR" \
    --dataset imagenet_code \
    --gpt-model GPT-B \
    --gpt-type c2i \
    --image-size 256 \
    --downsample-size 16 \
    --vocab-size 16384 \
    --num-classes 1000 \
    --cls-token-num 1 \
    --epochs 300 \
    --global-batch-size 256 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --beta1 0.9 \
    --beta2 0.95 \
    --max-grad-norm 1.0 \
    --mixed-precision bf16 \
    --dropout-p 0.1 \
    --token-dropout-p 0.1 \
    --drop-path-rate 0.0 \
    --max-drop-rate 0.50 \
    --semantic-drop-temperature 0.35 \
    --loss-weight-contrast 0.2 \
    --contrast-temp 0.07 \
    --contrast-dropout-p 0.2 \
    --contrast-num-samples 2048 \
    --global-seed 42 \
    --num-workers 24 \
    --log-every 100 \
    --ckpt-every 5000 \
    --extra-ckpt-step -1 \
    --ema
