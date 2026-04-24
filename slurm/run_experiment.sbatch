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
#   METHOD  — one of: streaming_llm, streaming_llm_vllm, h2o, full_cache, paged_attention
#   CONFIG  — path to yaml config, e.g. configs/experiment1_short.yaml
#
# Optional env vars:
#   START_SIZE              — StreamingLLM attention sink size (default: 4)
#   RECENT_SIZE             — StreamingLLM sliding window size (default: 1024)
#   MAX_MODEL_LEN           — vLLM max sequence length (default: 32768)
#   GPU_MEMORY_UTILIZATION  — vLLM GPU memory fraction (default: 0.90)
#   BLOCK_SIZE              — vLLM KV block size in tokens (default: 16)
#
# Example:
#   sbatch --export=ALL,METHOD=h2o,CONFIG=configs/experiment1_short.yaml slurm/run_experiment.sbatch
#   sbatch --export=ALL,METHOD=streaming_llm_vllm,CONFIG=configs/experiment2_long.yaml slurm/run_experiment.sbatch

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
if [ "${METHOD}" = "streaming_llm_vllm" ]; then
    START_SIZE=${START_SIZE:-4}
    RECENT_SIZE=${RECENT_SIZE:-1024}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
    GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
    BLOCK_SIZE=${BLOCK_SIZE:-16}
    EXTRA_ARGS="--start_size ${START_SIZE} --recent_size ${RECENT_SIZE} --max_model_len ${MAX_MODEL_LEN} --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} --block_size ${BLOCK_SIZE}"
fi

python scripts/run_experiment1.py \
    --config "${CONFIG}" \
    --method "${METHOD}" \
    ${EXTRA_ARGS}

echo "=== Done ==="