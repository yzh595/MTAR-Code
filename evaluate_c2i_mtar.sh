#!/usr/bin/env bash
set -euo pipefail

# Evaluate an MTAR sample NPZ with the bundled OpenAI-style evaluator.
#
# Usage:
#   bash evaluate_c2i_mtar.sh REFERENCE_NPZ SAMPLE_NPZ [EVALUATION_LOG]
#
# Run inside an evaluation environment containing TensorFlow, NumPy and SciPy.
# To select a Conda environment without activating it first:
#   EVAL_CONDA_ENV=lla bash evaluate_c2i_mtar.sh REF_NPZ SAMPLE_NPZ

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REFERENCE_NPZ=${1:-}
SAMPLE_NPZ=${2:-}
EVALUATION_LOG=${3:-}

if [[ -z "$REFERENCE_NPZ" || -z "$SAMPLE_NPZ" ]]; then
    echo "Usage: $0 REFERENCE_NPZ SAMPLE_NPZ [EVALUATION_LOG]" >&2
    exit 1
fi

EVALUATOR="$PROJECT_ROOT/evaluations/c2i/evaluator.py"
for required in "$REFERENCE_NPZ" "$SAMPLE_NPZ" "$EVALUATOR"; do
    [[ -s "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done

SAMPLE_ABS=$(cd "$(dirname "$SAMPLE_NPZ")" && pwd)/$(basename "$SAMPLE_NPZ")
REFERENCE_ABS=$(cd "$(dirname "$REFERENCE_NPZ")" && pwd)/$(basename "$REFERENCE_NPZ")
METRICS_FILE="${SAMPLE_ABS%.npz}.txt"
if [[ -z "$EVALUATION_LOG" ]]; then
    EVALUATION_LOG="${SAMPLE_ABS%.npz}_evaluation.log"
fi
mkdir -p "$(dirname "$EVALUATION_LOG")"

if [[ -n "${EVAL_CONDA_ENV:-}" ]]; then
    if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
        CONDA_BIN="$CONDA_EXE"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BIN=$(command -v conda)
    elif [[ -x /opt/conda/bin/conda ]]; then
        CONDA_BIN=/opt/conda/bin/conda
    else
        echo "EVAL_CONDA_ENV is set, but conda was not found" >&2
        exit 1
    fi
    EVAL_COMMAND=("$CONDA_BIN" run --no-capture-output --name "$EVAL_CONDA_ENV" python)
else
    EVAL_COMMAND=("${EVAL_PYTHON:-python}")
fi

# The evaluator caches classify_image_graph_def.pb relative to its working
# directory, so keep that cache inside evaluations/c2i.
cd "$PROJECT_ROOT/evaluations/c2i"
"${EVAL_COMMAND[@]}" "$EVALUATOR" "$REFERENCE_ABS" "$SAMPLE_ABS" \
    2>&1 | tee "$EVALUATION_LOG"

[[ -s "$METRICS_FILE" ]] || {
    echo "Evaluator did not create metrics file: $METRICS_FILE" >&2
    exit 1
}
for metric in "Inception Score:" "FID:" "sFID:" "Precision:" "Recall:"; do
    grep -Fq "$metric" "$METRICS_FILE" || {
        echo "Incomplete evaluation: missing $metric" >&2
        exit 1
    }
done

echo "Evaluation complete"
echo "  metrics: $METRICS_FILE"
echo "  log: $EVALUATION_LOG"
