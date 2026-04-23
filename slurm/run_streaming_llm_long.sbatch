#!/bin/bash
#SBATCH --job-name=sllm_long
#SBATCH --partition=edu1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --account=edu
#SBATCH --output=slurm-%j-sllm_long.out

UNI="th3192"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
REPO="${SCRATCH}/HPML-KV-Cache-Mangement"

module purge
module load cuda/12.3
module load python/3.11

source "${SCRATCH}/.venv/bin/activate"
cd "${REPO}"

echo "=== Job: StreamingLLM — LongBench v2 (short answer) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python scripts/run_experiment1.py \
    --config configs/experiment2_long.yaml \
    --method streaming_llm \
    --start_size 4 \
    --recent_size 1024

echo "=== Done ==="
