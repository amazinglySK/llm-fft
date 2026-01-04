#!/usr/bin/env bash
set -euo pipefail

# === USAGE ===
# ./run_harmonic_experiments.sh <wandb_entity> <wandb_project>
#
# Example:
# ./run_harmonic_experiments.sh myusername lag-llama-experiments
#
# This script runs pretraining experiments sequentially with different harmonic filter orders

if [ "$#" -lt 3 ]; then
  echo "❌ Error: You must provide Weights & Biases entity, project name, and energy threshold."
  echo "Usage: $0 <wandb_entity> <wandb_project> <energy_threshold>"
  exit 1
fi

WANDB_ENTITY="$1"
WANDB_PROJECT="$2"
ENERGY_THRESHOLD="$3"  
# Array of harmonic filter orders to test
HARMONICS=(2 4 6 8)

echo "=================================================="
echo "Running sequential pretraining experiments"
echo "Entity: $WANDB_ENTITY"
echo "Project: $WANDB_PROJECT"
echo "Harmonics: ${HARMONICS[@]}"
echo "Energy Threshold: $ENERGY_THRESHOLD"
echo "=================================================="

# Run experiments sequentially
for HARMONIC in "${HARMONICS[@]}"
do
    echo ""
    echo "=================================================="
    echo "Starting experiment with harmonic filter order: $HARMONIC"
    echo "=================================================="
    
    ./scripts/pretrain_filtered_small.sh "$WANDB_ENTITY" "$WANDB_PROJECT" $HARMONIC $ENERGY_THRESHOLD
    
    echo ""
    echo "=================================================="
    echo "Completed experiment with harmonic filter order: $HARMONIC"
    echo "=================================================="
    echo ""
done

echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
