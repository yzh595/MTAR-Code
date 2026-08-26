#!/usr/bin/env bash
set -euo pipefail

# Extract aligned VQ codes, ImageNet labels, and DINOv3 semantic scores.
#
# Usage:
#   bash dinov3/extract_dinov3_scores.sh DATA_PATH CODE_PATH VQ_CKPT [DINO_MODEL]
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1 bash dinov3/extract_dinov3_scores.sh \
#       /path/to/imagenet/train \
#       /path/to/features_dinov3_large_flip_all \
#       /path/to/vq_ds16_c2i.pt \
#       facebook/dinov3-vitl16-pretrain-lvd1689m

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_PATH=${1:-${DATA_PATH:-}}
CODE_PATH=${2:-${CODE_PATH:-}}
VQ_CKPT=${3:-${VQ_CKPT:-}}
DINO_MODEL_NAME=${4:-${DINO_MODEL_NAME:-facebook/dinov3-vitl16-pretrain-lvd1689m}}

if [[ -z "$DATA_PATH" || -z "$CODE_PATH" || -z "$VQ_CKPT" ]]; then
    echo "Usage: $0 DATA_PATH CODE_PATH VQ_CKPT [DINO_MODEL]" >&2
    exit 1
fi

EXTRACTOR="$PROJECT_ROOT/dinov3/extract_dinov3_scores.py"
for required in "$DATA_PATH" "$VQ_CKPT" "$EXTRACTOR"; do
    [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
for required in \
    "$PROJECT_ROOT/dataset/augmentation.py" \
    "$PROJECT_ROOT/dataset/build.py" \
    "$PROJECT_ROOT/tokenizer/tokenizer_image/vq_model.py"; do
    [[ -s "$required" ]] || { echo "Missing required code: $required" >&2; exit 1; }
done

TORCHRUN=${TORCHRUN:-torchrun}
MASTER_PORT=${MASTER_PORT:-29501}
IMAGE_SIZE=${IMAGE_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-16}
GLOBAL_SEED=${GLOBAL_SEED:-0}
TEN_CROP=${TEN_CROP:-0}
CROP_RANGE=${CROP_RANGE:-1.1}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi
IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${NUM_GPUS:-${#GPU_ARRAY[@]}}
(( NUM_GPUS > 0 )) || { echo "NUM_GPUS must be positive" >&2; exit 1; }

mkdir -p "$CODE_PATH"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

EXTRA_ARGS=()
if [[ "$TEN_CROP" == "1" ]]; then
    EXTRA_ARGS+=(--ten-crop --crop-range "$CROP_RANGE")
fi

echo "DINOv3 semantic-score extraction"
echo "  images: $DATA_PATH"
echo "  output: $CODE_PATH"
echo "  image size: $IMAGE_SIZE"
echo "  augmentations: $([[ "$TEN_CROP" == "1" ]] && echo ten-crop || echo original+horizontal-flip)"
echo "  DINOv3: $DINO_MODEL_NAME"
echo "  GPUs: $CUDA_VISIBLE_DEVICES (processes=$NUM_GPUS)"

"$TORCHRUN" \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --node_rank=0 \
    --master_port="$MASTER_PORT" \
    "$EXTRACTOR" \
    --data-path "$DATA_PATH" \
    --code-path "$CODE_PATH" \
    --vq-model VQ-16 \
    --vq-ckpt "$VQ_CKPT" \
    --dino-model-name "$DINO_MODEL_NAME" \
    --dataset imagenet \
    --image-size "$IMAGE_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --global-seed "$GLOBAL_SEED" \
    "${EXTRA_ARGS[@]}"

for output_dir in \
    "$CODE_PATH/imagenet${IMAGE_SIZE}_codes" \
    "$CODE_PATH/imagenet${IMAGE_SIZE}_labels" \
    "$CODE_PATH/imagenet${IMAGE_SIZE}_scores"; do
    [[ -d "$output_dir" ]] || { echo "Missing output directory: $output_dir" >&2; exit 1; }
done

echo "Extraction complete: $CODE_PATH"
