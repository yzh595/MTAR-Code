#!/usr/bin/env bash
set -euo pipefail

# Shared MTAR / MTAR++ launcher.
# MTAR:   bash train_c2i_mtar.sh FEATURE_DIR [OUTPUT_DIR]
# MTAR++: METHOD=MTAR++ TEACHER_CKPT=/path/to/NAR-XL.pt bash train_c2i_mtar.sh FEATURE_DIR OUTPUT_DIR
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_PATH=${1:-${CODE_PATH:-}}
RESULTS_DIR=${2:-${RESULTS_DIR:-"$PROJECT_ROOT/results"}}
[[ -n "$CODE_PATH" ]] || { echo "Usage: $0 CODE_PATH [RESULTS_DIR]" >&2; exit 2; }
for directory in imagenet256_codes imagenet256_labels imagenet256_scores; do
    [[ -d "$CODE_PATH/$directory" ]] || { echo "Missing $CODE_PATH/$directory" >&2; exit 1; }
done
CODE_PATH=$(cd "$CODE_PATH" && pwd)
mkdir -p "$RESULTS_DIR"
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)

case "${METHOD:-MTAR}" in
    MTAR|mtar)
        TRAIN_SCRIPT="$PROJECT_ROOT/autoregressive/train/train_c2i_MTAR.py"
        DEFAULT_MODEL=GPT-B
        DEFAULT_SEED=42
        DEFAULT_DEVICES=0
        METHOD_ARGS=(--max-drop-rate 0.50 --ckpt-every 5000 --extra-ckpt-step -1)
        ;;
    MTAR++|mtar++|mtarpp)
        TRAIN_SCRIPT="$PROJECT_ROOT/autoregressive/train/train_c2i_MATR++.py"
        DEFAULT_MODEL=GPT-L
        DEFAULT_SEED=0
        DEFAULT_DEVICES=0,1
        [[ -s "${TEACHER_CKPT:-}" ]] || { echo "Set TEACHER_CKPT to the NAR-XL checkpoint." >&2; exit 1; }
        TEACHER_CKPT=$(cd "$(dirname "$TEACHER_CKPT")" && pwd)/$(basename "$TEACHER_CKPT")
        METHOD_ARGS=(
            --teacher-model NAR-XL --teacher-ckpt "$TEACHER_CKPT"
            --student-target-layers 24 --teacher-target-layers 36 --projector-dim 2048
            --distill-weight 1.0 --future-feature-pred-layer 24
            --future-feature-offsets 1,16 --future-feature-loss-weights 1.0,1.0
            --max-student-drop-rate 0.5 --student-drop-switch-ratio 0.8
            --max-physical-drop-rate 0.5 --physical-drop-switch-ratio 1.0
            --max-seq-len 257 --cloud-save-path "$RESULTS_DIR/checkpoints" --no-local-save
            --checkpoint-steps 1200000,1250000,1300000,1350000,1400000,1450000,1500000
            --debug-first-steps 3
        )
        ;;
    *) echo "Unknown METHOD: $METHOD (use MTAR or MTAR++)" >&2; exit 2 ;;
esac

[[ -s "$TRAIN_SCRIPT" ]] || { echo "Missing $TRAIN_SCRIPT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-$DEFAULT_DEVICES}}
IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${NUM_GPUS:-${#GPU_ARRAY[@]}}
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
COMMAND=(
    "${TORCHRUN:-torchrun}" --nproc_per_node="$NUM_GPUS"
    --master_port="${TRAIN_MASTER_PORT:-33842}" "$TRAIN_SCRIPT"
    --code-path "$CODE_PATH" --results-dir "$RESULTS_DIR"
    --dataset imagenet_code --gpt-model "${GPT_MODEL:-$DEFAULT_MODEL}" --gpt-type c2i
    --image-size 256 --downsample-size 16 --vocab-size 16384
    --num-classes 1000 --cls-token-num 1
    --epochs "${EPOCHS:-300}" --global-batch-size "${GLOBAL_BATCH_SIZE:-256}" --lr "${LR:-1e-4}"
    --weight-decay 0.05 --beta1 0.9 --beta2 0.95 --max-grad-norm 1.0
    --mixed-precision bf16 --dropout-p 0.1 --token-dropout-p 0.1 --drop-path-rate 0.0
    --semantic-drop-temperature 0.35 --loss-weight-contrast 0.2
    --contrast-temp 0.07 --contrast-dropout-p 0.2 --contrast-num-samples 2048
    --global-seed "${GLOBAL_SEED:-$DEFAULT_SEED}" --num-workers 24 --log-every 100
    --ema "${METHOD_ARGS[@]}"
)
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
[[ ${DRY_RUN:-0} == 1 ]] && exit 0
cd "$PROJECT_ROOT"
"${COMMAND[@]}" 2>&1 | tee "$RESULTS_DIR/training.log"
