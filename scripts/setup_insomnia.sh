#!/bin/bash
# Run this ONCE on the login node after cloning the repo.
# Sets up venv and symlinks in scratch so all configs work unchanged.

set -e

UNI="${USER}"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
VENV="${SCRATCH}/.venv"

echo "=== Creating scratch directories ==="
mkdir -p "${SCRATCH}/models"
mkdir -p "${SCRATCH}/results"

# Symlink ~/models -> scratch/models so configs (local_path: ~/models/...) work as-is
if [ ! -L "${HOME}/models" ]; then
    ln -s "${SCRATCH}/models" "${HOME}/models"
    echo "Linked ~/models -> ${SCRATCH}/models"
fi

echo "=== Loading modules ==="
module purge
module load anaconda/2023.09   # provides Python 3.11 — required for vllm
module load cuda/12.3

echo "=== Creating virtual environment in scratch ==="
python -m venv "${VENV}"       # uses Python 3.11 from anaconda
source "${VENV}/bin/activate"

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cu121
# Install all requirements (vllm==0.8.5 pinned in requirements.txt)
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ${VENV}/bin/activate"
echo "Next step: sbatch slurm/download_model.sbatch"
