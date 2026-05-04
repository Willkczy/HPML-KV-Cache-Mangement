#!/bin/bash
#SBATCH --job-name=sllm_sweep
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --account=edu
#SBATCH --output=slurm-%j-sweep.out

# Sweep recent_size values on MMLU short and LongBench short answer.
# Results saved to results/sweep_streaming_llm/<config>/<recent_size>.json

UNI="${USER}"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
REPO="${SCRATCH}/HPML-KV-Cache-Mangement"

module purge
module load cuda/12.3
unset CUDA_VISIBLE_DEVICES

source "${HOME}/.venv/bin/activate"
cd "${REPO}"

echo "=== StreamingLLM window size sweep ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python scripts/sweep_streaming_llm.py \
    --configs configs/experiment1_short.yaml \
             configs/experiment2_long.yaml \
    --window_sizes 64 256 512

echo "=== Sweep done ==="