#!/bin/bash
# Submit all 4 experiment configs for a given method.
#
# Usage:
#   bash slurm/submit_all.sh <method>
#
# Methods: streaming_llm | h2o | full_cache | paged_attention
#
# Example:
#   bash slurm/submit_all.sh h2o

METHOD=${1:?"Usage: bash slurm/submit_all.sh <method>"}

CONFIGS=(
    "configs/experiment1_short.yaml"
    "configs/experiment2_long.yaml"
    "configs/experiment2_long_explain.yaml"
    "configs/experiment2_summarization.yaml"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting all experiments for method: ${METHOD}"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    JOB=$(sbatch --partition=short --gres=gpu:A6000:1 \
        --export=ALL,METHOD="${METHOD}",CONFIG="${CONFIG}" \
        "${SCRIPT_DIR}/run_experiment.sbatch")
    echo "  ${JOB} — ${CONFIG}"
done

echo ""
echo "Monitor with: squeue -u \${USER}"