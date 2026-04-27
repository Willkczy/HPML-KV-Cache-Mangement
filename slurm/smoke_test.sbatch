#!/bin/bash
#SBATCH --job-name=kvcache_smoke
#SBATCH --partition=short
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=00:20:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --account=edu
#SBATCH --output=slurm-%j-smoke.out

# Run 3 samples only — validate the full stack before submitting long jobs.
#
# Usage (default: streaming_llm):
#   sbatch slurm/smoke_test.sbatch
#
# Usage (other method):
#   sbatch --export=ALL,METHOD=h2o slurm/smoke_test.sbatch

METHOD=${METHOD:-streaming_llm}

UNI="${USER}"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
REPO="${SCRATCH}/HPML-KV-Cache-Mangement"

module purge
module load cuda/12.3

source "${HOME}/.venv/bin/activate"
cd "${REPO}"

echo "=== Smoke test: ${METHOD} — MMLU (3 samples) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

EXTRA_ARGS=""
if [ "${METHOD}" = "streaming_llm" ]; then
    EXTRA_ARGS="--start_size 4 --recent_size 256"
fi
if [ "${METHOD}" = "streaming_llm_vllm" ]; then
    EXTRA_ARGS="--start_size 4 --recent_size 1024 --max_model_len 32768 --gpu_memory_utilization 0.90 --block_size 16"
fi

python scripts/run_experiment1.py \
    --config configs/experiment1_short.yaml \
    --method "${METHOD}" \
    ${EXTRA_ARGS} \
    --smoke_test

echo "=== Smoke test done ==="