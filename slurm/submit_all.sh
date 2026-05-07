#!/bin/bash
# Submit all 4 experiment configs for a given method.
#
# Usage:
#   bash slurm/submit_all.sh <method> [gpu_type]
#
# Methods:  streaming_llm | h2o | full_cache | paged_attention
# GPU type: l40s | l40 | h100 | A6000 | 1 (any)  — default: l40s
#
# Example:
#   bash slurm/submit_all.sh h2o l40s
#   bash slurm/submit_all.sh streaming_llm h100

METHOD=${1:?"Usage: bash slurm/submit_all.sh <method> [gpu_type]"}
GPU_TYPE=${2:-l40s}
GRES="gpu:${GPU_TYPE}:1"

CONFIGS=(
    "configs/experiment1_short.yaml"
    "configs/experiment2_long.yaml"
    "configs/experiment2_long_explain.yaml"
    "configs/experiment2_summarization.yaml"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting all experiments for method: ${METHOD} (GPU: ${GRES})"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    JOB=$(sbatch --partition=short --gres=${GRES} \
        --export=ALL,METHOD="${METHOD}",CONFIG="${CONFIG}" \
        "${SCRIPT_DIR}/run_experiment.sbatch")
    echo "  ${JOB} — ${CONFIG}"
done

echo ""
echo "Monitor with: squeue -u \${USER}"