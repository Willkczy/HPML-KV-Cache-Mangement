#!/bin/bash
#SBATCH --job-name=sllm_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --account=edu
#SBATCH --output=slurm-%j-smoke.out

# Run 3 samples only — validate the full stack before submitting long jobs.

UNI="th3192"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
REPO="${SCRATCH}/HPML-KV-Cache-Mangement"

module purge
module load cuda/12.3
module load python/3.11

source "${SCRATCH}/.venv/bin/activate"
cd "${REPO}"

echo "=== Smoke test: StreamingLLM — MMLU (3 samples) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python scripts/run_experiment1.py \
    --config configs/experiment1_short.yaml \
    --method streaming_llm \
    --start_size 4 \
    --recent_size 256 \
    --smoke_test

echo "=== Smoke test done ==="
