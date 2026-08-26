#!/usr/bin/env bash
set -euo pipefail

# Sample ImageNet-256 images with the MTAR checkpoint loader.
#
# Usage:
#   bash sample_c2i_mtar.sh GPT_CKPT VQ_CKPT [CFG] [SAMPLE_DIR] [NUM_SAMPLES]
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash sample_c2i_mtar.sh \
#       /path/to/0250000.pt \
#       /path/to/vq_ds16_c2i.pt \
#       2.25 \
#       ./fid_samples \
#       50000

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPT_CKPT=${1:-}
VQ_CKPT=${2:-}
CFG_SCALE=${3:-2.0}
SAMPLE_DIR=${4:-"$PROJECT_ROOT/fid_samples"}
NUM_SAMPLES=${5:-50000}

if [[ -z "$GPT_CKPT" || -z "$VQ_CKPT" ]]; then
    echo "Usage: $0 GPT_CKPT VQ_CKPT [CFG] [SAMPLE_DIR] [NUM_SAMPLES]" >&2
    exit 1
fi

SAMPLE_SCRIPT="$PROJECT_ROOT/autoregressive/sample/sample_c2i_ddp_MTAR.py"
GENERATE_SCRIPT="$PROJECT_ROOT/autoregressive/models/generate.py"
for required in "$GPT_CKPT" "$VQ_CKPT" "$SAMPLE_SCRIPT" "$GENERATE_SCRIPT"; do
    [[ -s "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done

TORCHRUN=${TORCHRUN:-torchrun}
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29663}
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-32}
GLOBAL_SEED=${GLOBAL_SEED:-0}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$SAMPLE_DIR"
cd "$PROJECT_ROOT"

echo "MTAR sampling"
echo "  checkpoint: $GPT_CKPT"
echo "  VQ checkpoint: $VQ_CKPT"
echo "  CFG: $CFG_SCALE"
echo "  samples: $NUM_SAMPLES"
echo "  output root: $SAMPLE_DIR"
echo "  GPUs: $CUDA_VISIBLE_DEVICES (processes=$NUM_GPUS)"

"$TORCHRUN" \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --node_rank=0 \
    --master_port="$MASTER_PORT" \
    "$SAMPLE_SCRIPT" \
    --gpt-model GPT-B \
    --gpt-ckpt "$GPT_CKPT" \
    --gpt-type c2i \
    --vq-model VQ-16 \
    --vq-ckpt "$VQ_CKPT" \
    --image-size 256 \
    --image-size-eval 256 \
    --downsample-size 16 \
    --num-classes 1000 \
    --cfg-scale "$CFG_SCALE" \
    --temperature 1.0 \
    --top-k 0 \
    --top-p 1.0 \
    --sample-dir "$SAMPLE_DIR" \
    --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
    --num-fid-samples "$NUM_SAMPLES" \
    --global-seed "$GLOBAL_SEED" \
    --precision bf16 \
    --compile

CKPT_NAME=$(basename "$GPT_CKPT")
CKPT_STEM=${CKPT_NAME%.pt}
CKPT_STEM=${CKPT_STEM%.pth}
printf -v CFG_CANON '%.15g' "$CFG_SCALE"
# argparse stores CFG as float, so an integral value appears as e.g. "2.0" in
# the sampler's folder name rather than "2".
if [[ "$CFG_CANON" != *.* && "$CFG_CANON" != *e* && "$CFG_CANON" != *E* ]]; then
    CFG_CANON="${CFG_CANON}.0"
fi
NPZ_PATH="$SAMPLE_DIR/GPT-B-${CKPT_STEM}-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-${CFG_CANON}-seed-${GLOBAL_SEED}.npz"
[[ -s "$NPZ_PATH" ]] || {
    echo "Sampling finished but the expected NPZ was not found: $NPZ_PATH" >&2
    exit 1
}

echo "Sampling complete: $NPZ_PATH"
