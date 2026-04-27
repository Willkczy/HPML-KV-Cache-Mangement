#!/bin/bash
# Run this ONCE on the login node after cloning the repo.
# Venv goes in HOME (personal 50 GB quota).
# Models and repo stay in SCRATCH (shared class quota).

set -e

UNI="${USER}"
SCRATCH="/insomnia001/depts/edu/users/${UNI}"
VENV="${HOME}/.venv"   # HOME has personal quota; SCRATCH is shared class quota (limited)

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

echo "=== Creating virtual environment in HOME ==="
python -m venv "${VENV}"       # uses Python 3.11 from anaconda
source "${VENV}/bin/activate"

echo "=== Installing dependencies ==="
pip install --upgrade pip --no-cache-dir
pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
# Install all requirements (vllm==0.8.5 pinned in requirements.txt)
pip install -r requirements.txt --no-cache-dir

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ${VENV}/bin/activate"
echo "Next step: sbatch slurm/download_model.sbatch"
