#!/bin/bash
#SBATCH --job-name=kvcache
#SBATCH --partition=short
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --account=edu
#SBATCH --output=slurm-%j-%x.out

# Required env vars (pass via --export):
#   METHOD  — one of: streaming_llm, h2o, full_cache, paged_attention
#   CONFIG  — path to yaml config, e.g. configs/experiment1_short.yaml
#
# Optional env vars:
#   START_SIZE    — StreamingLLM attention sink size (default: 4)
#   RECENT_SIZE   — StreamingLLM sliding window size (default: 1024)
#   MAX_MODEL_LEN — PagedAttention max sequence length (auto-set by config if not specified)
#
# Example:
#   sbatch --export=ALL,METHOD=h2o,CONFIG=configs/experiment1_short.yaml slurm/run_experiment.sbatch
#   bash slurm/submit_all.sh streaming_llm

UNI="${USER}"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
REPO="${SCRATCH}/HPML-KV-Cache-Mangement"

module purge
module load cuda/12.3

source "${SCRATCH}/.venv/bin/activate"
cd "${REPO}"

echo "=== KV Cache Experiment ==="
echo "User: ${UNI}, Method: ${METHOD}, Config: ${CONFIG}"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [ -z "${METHOD}" ] || [ -z "${CONFIG}" ]; then
    echo "ERROR: METHOD and CONFIG must be set via --export"
    exit 1
fi

EXTRA_ARGS=""
if [ "${METHOD}" = "streaming_llm" ]; then
    START_SIZE=${START_SIZE:-4}
    RECENT_SIZE=${RECENT_SIZE:-1024}
    EXTRA_ARGS="--start_size ${START_SIZE} --recent_size ${RECENT_SIZE}"
fi

if [ "${METHOD}" = "paged_attention" ]; then
    # Auto-set max_model_len based on config if not overridden
    if [ -z "${MAX_MODEL_LEN}" ]; then
        case "${CONFIG}" in
            *experiment1_short*)       MAX_MODEL_LEN=4096  ;;
            *experiment2_long_explain*) MAX_MODEL_LEN=34000 ;;
            *experiment2_long*)        MAX_MODEL_LEN=34000 ;;
            *experiment2_summarization*) MAX_MODEL_LEN=18000 ;;
            *)                         MAX_MODEL_LEN=34000 ;;
        esac
    fi
    EXTRA_ARGS="${EXTRA_ARGS} --max_model_len ${MAX_MODEL_LEN}"
    echo "PagedAttention: max_model_len=${MAX_MODEL_LEN}"
fi

python scripts/run_experiment1.py \
    --config "${CONFIG}" \
    --method "${METHOD}" \
    ${EXTRA_ARGS}

echo "=== Done ==="