#!/bin/bash
#SBATCH --job-name=download_qwen
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --account=edu
#SBATCH --output=slurm-%j-download.out

UNI="${USER}"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
MODEL_DIR="${SCRATCH}/models/qwen2.5-7b"

module purge

source "${HOME}/.venv/bin/activate"

echo "Downloading Qwen2.5-7B to ${MODEL_DIR} ..."
huggingface-cli download Qwen/Qwen2.5-7B \
    --local-dir "${MODEL_DIR}" \
    --local-dir-use-symlinks False

echo "Download complete: ${MODEL_DIR}"
