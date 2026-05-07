#!/bin/bash
#SBATCH --job-name=kvcache_mixed
#SBATCH --partition=short
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --account=edu
#SBATCH --output=slurm-%j-%x.out

# Required env vars:
#   METHOD - one of: full_cache, h2o, streaming_llm, paged_attention, streaming_llm_vllm
#
# Optional env vars:
#   CONFIG                  - mixed workload config path
#   TRACE                   - trace JSONL path
#   GENERATE_TRACE          - 1 to generate trace before running
#   SMOKE_TEST              - 1 to run first 3 trace requests
#   NUM_REQUESTS            - cap total trace requests
#   START_SIZE              - StreamingLLM attention sink size
#   RECENT_SIZE             - StreamingLLM/H2O recent window size
#   HH_SIZE                 - H2O heavy-hitter budget
#   MAX_MODEL_LEN           - vLLM max sequence length
#   GPU_MEMORY_UTILIZATION  - vLLM GPU memory fraction
#   BLOCK_SIZE              - vLLM KV block size

UNI="${USER}"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
REPO="${SCRATCH}/HPML-KV-Cache-Mangement"

module purge
module load cuda/12.3

unset CUDA_VISIBLE_DEVICES

source "${HOME}/.venv/bin/activate"
cd "${REPO}"

CONFIG=${CONFIG:-configs/experiment_mixed_workload.yaml}
TRACE=${TRACE:-results/traces/mixed_workload_trace.jsonl}

echo "=== Mixed KV Cache Workload ==="
echo "User: ${UNI}, Method: ${METHOD}, Config: ${CONFIG}, Trace: ${TRACE}"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [ -z "${METHOD}" ]; then
    echo "ERROR: METHOD must be set via --export"
    exit 1
fi

if [ "${GENERATE_TRACE}" = "1" ]; then
    python -m data.realistic_trace \
        --config "${CONFIG}" \
        --output "${TRACE}"
fi

EXTRA_ARGS=""

if [ "${METHOD}" = "streaming_llm" ]; then
    START_SIZE=${START_SIZE:-4}
    RECENT_SIZE=${RECENT_SIZE:-1024}
    EXTRA_ARGS="${EXTRA_ARGS} --start_size ${START_SIZE} --recent_size ${RECENT_SIZE}"
fi

if [ "${METHOD}" = "h2o" ]; then
    HH_SIZE=${HH_SIZE:-64}
    RECENT_SIZE=${RECENT_SIZE:-64}
    EXTRA_ARGS="${EXTRA_ARGS} --hh_size ${HH_SIZE} --recent_size ${RECENT_SIZE}"
fi

if [ "${METHOD}" = "paged_attention" ] || [ "${METHOD}" = "streaming_llm_vllm" ]; then
    export VLLM_USE_V1=${VLLM_USE_V1:-0}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-34000}
    GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
    BLOCK_SIZE=${BLOCK_SIZE:-16}
    EXTRA_ARGS="${EXTRA_ARGS} --max_model_len ${MAX_MODEL_LEN} --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} --block_size ${BLOCK_SIZE}"
    echo "vLLM: max_model_len=${MAX_MODEL_LEN}, gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}, block_size=${BLOCK_SIZE}, VLLM_USE_V1=${VLLM_USE_V1}"
fi

if [ "${METHOD}" = "streaming_llm_vllm" ]; then
    START_SIZE=${START_SIZE:-4}
    RECENT_SIZE=${RECENT_SIZE:-1024}
    EXTRA_ARGS="${EXTRA_ARGS} --start_size ${START_SIZE} --recent_size ${RECENT_SIZE}"
fi

if [ "${SMOKE_TEST}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --smoke_test"
fi

if [ -n "${NUM_REQUESTS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --num_requests ${NUM_REQUESTS}"
fi

python scripts/run_experiment_mixed_workload.py \
    --config "${CONFIG}" \
    --trace "${TRACE}" \
    --method "${METHOD}" \
    ${EXTRA_ARGS}

echo "=== Done ==="
