#!/usr/bin/env bash
set -euo pipefail

# Historical random-class-label, NTP-only LlamaGen sampling protocol.
# Usage: bash sample_c2i_mtarpp.sh GPT_CKPT VQ_CKPT [CFG] [OUTPUT_DIR] [NUM_SAMPLES]
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if (( $# < 2 )); then
    echo "Usage: $0 GPT_CKPT VQ_CKPT [CFG] [OUTPUT_DIR] [NUM_SAMPLES]" >&2
    exit 2
fi
GPT_CKPT=$1
VQ_CKPT=$2
CFG=${3:-1.80}
OUTPUT_DIR=${4:-./fid_samples/mtarpp}
NUM_SAMPLES=${5:-50000}
for required in "$GPT_CKPT" "$VQ_CKPT"; do
    [[ -s "$required" ]] || { echo "Missing checkpoint: $required" >&2; exit 1; }
done
GPT_CKPT=$(cd "$(dirname "$GPT_CKPT")" && pwd)/$(basename "$GPT_CKPT")
VQ_CKPT=$(cd "$(dirname "$VQ_CKPT")" && pwd)/$(basename "$VQ_CKPT")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
COMMAND=("${TORCHRUN:-torchrun}" --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
    --master_port="${MASTER_PORT:-29503}" --module mtarpp.sample
    --gpt-model GPT-L --gpt-type c2i --gpt-ckpt "$GPT_CKPT"
    --vq-model VQ-16 --vq-ckpt "$VQ_CKPT"
    --image-size 256 --image-size-eval 256 --downsample-size 16
    --num-classes 1000 --cfg-scale "$CFG" --temperature 1.0 --top-k 0 --top-p 1.0
    --sample-dir "$OUTPUT_DIR" --per-proc-batch-size "${SAMPLE_BATCH_SIZE:-32}"
    --num-fid-samples "$NUM_SAMPLES" --global-seed 0 --precision bf16 --compile)
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
[[ ${DRY_RUN:-0} == 1 ]] && exit 0
cd "$PROJECT_ROOT"
"${COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/sampling.log"
