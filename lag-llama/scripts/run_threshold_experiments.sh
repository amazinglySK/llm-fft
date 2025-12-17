#!/usr/bin/env bash
set -euo pipefail

# === USAGE ===
# ./run_threshold_experiments.sh <wandb_entity> <wandb_project>
#
# Example:
# ./run_threshold_experiments.sh myusername lag-llama-experiments
#
# This script runs pretraining experiments sequentially with different energy thresholds

if [ "$#" -lt 2 ]; then
  echo "❌ Error: You must provide Weights & Biases entity and project name."
  echo "Usage: $0 <wandb_entity> <wandb_project>"
  exit 1
fi

WANDB_ENTITY="$1"
WANDB_PROJECT="$2"

# Array of energy thresholds to test
THRESHOLDS=(0.7 0.75 0.8 0.85 0.9 0.95)

echo "=================================================="
echo "Running sequential pretraining experiments"
echo "Entity: $WANDB_ENTITY"
echo "Project: $WANDB_PROJECT"
echo "Thresholds: ${THRESHOLDS[@]}"
echo "=================================================="

# Run experiments sequentially
for THRESHOLD in "${THRESHOLDS[@]}"
do
    echo ""
    echo "=================================================="
    echo "Starting experiment with threshold: $THRESHOLD"
    echo "=================================================="
    
    ./pretrain_filtered.sh "$WANDB_ENTITY" "$WANDB_PROJECT" "$THRESHOLD"
    
    echo ""
    echo "=================================================="
    echo "Completed experiment with threshold: $THRESHOLD"
    echo "=================================================="
    echo ""
done

echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
