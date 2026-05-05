#!/bin/bash
#SBATCH --job-name=vllm_serving
#SBATCH --partition=short
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --account=edu
#SBATCH --output=slurm-%j-serving.out

# True vLLM concurrent serving benchmark using AsyncLLMEngine.
# Compares paged_attention vs streaming_llm_vllm under concurrent load.
#
# Required env vars:
#   METHOD — paged_attention | streaming_llm_vllm
#
# Optional env vars:
#   ARRIVAL_RATES    — space-separated list (default: "0.5 1.0 2.0 4.0")
#   MAX_MODEL_LEN    — default: 34000
#   RECENT_SIZE      — StreamingLLM window size (default: 1024)
#   VLLM_USE_V1      — set to 1 for experiment 5 (v1 engine)
#
# Example (experiment 4 — both v0):
#   sbatch --export=ALL,METHOD=paged_attention slurm/run_vllm_serving_workload.sbatch
#   sbatch --export=ALL,METHOD=streaming_llm_vllm slurm/run_vllm_serving_workload.sbatch
#
# Example (experiment 5 — v1 paged_attention):
#   sbatch --export=ALL,METHOD=paged_attention,VLLM_USE_V1=1 slurm/run_vllm_serving_workload.sbatch

UNI="${USER}"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
REPO="${SCRATCH}/HPML-KV-Cache-Mangement"

module purge
module load cuda/12.3
unset CUDA_VISIBLE_DEVICES

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=${VLLM_USE_V1:-0}

source "${HOME}/.venv/bin/activate"
cd "${REPO}"

echo "=== vLLM Serving Benchmark ==="
echo "Method: ${METHOD}, VLLM_USE_V1=${VLLM_USE_V1}"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

ARRIVAL_RATES=${ARRIVAL_RATES:-"0.5 1.0 2.0 4.0"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-34000}
RECENT_SIZE=${RECENT_SIZE:-1024}

V1_FLAG=""
if [ "${VLLM_USE_V1}" = "1" ]; then
    V1_FLAG="--vllm_use_v1"
fi

python scripts/run_vllm_serving_workload.py \
    --config configs/experiment_mixed_workload.yaml \
    --method "${METHOD}" \
    --arrival_rates ${ARRIVAL_RATES} \
    --max_model_len ${MAX_MODEL_LEN} \
    --recent_size ${RECENT_SIZE} \
    ${V1_FLAG}

echo "=== Done ==="