#!/usr/bin/env bash
set -euo pipefail

# === USAGE ===
# ./pretrain_systematic_grid.sh <wandb_entity> <wandb_project> [accelerator]
#
# This script performs a systematic grid search over n_layer (1-8) and n_head (1-9)
# training each combination with 3 different seeds.
#
# Example:
# ./pretrain_systematic_grid.sh myusername lag-llama-grid-search

if [ "$#" -lt 2 ]; then
  echo "❌ Error: You must provide Weights & Biases entity and project name."
  echo "Usage: $0 <wandb_entity> <wandb_project> [accelerator]"
  exit 1
fi

WANDB_ENTITY="$1"
WANDB_PROJECT="$2"
HARMONIC=1.0
ENERGY_THRESHOLD=0.9
ACCELERATOR_FLAG=()

if [ "$#" -ge 3 ]; then
    ACCELERATOR="$3"
elif [ -n "${ACCELERATOR:-}" ]; then
    ACCELERATOR="${ACCELERATOR}"
else
    ACCELERATOR="gpu"
fi

if [ "$ACCELERATOR" = "cpu" ]; then
    ACCELERATOR_FLAG=(--accelerator cpu)
fi

mkdir -p experiments
mkdir -p experiments/seeds
mkdir -p experiments/results

# Grid search parameters
MIN_N_LAYER=3
MAX_N_LAYER=8
MIN_N_HEAD=3
MAX_N_HEAD=9
NUM_SEEDS=5

echo "======================================"
echo "Generating config files..."
echo "======================================"
python scripts/generate_grid_configs.py \
    --base_config configs/lag_llama.json \
    --output_dir configs/grid_configs \
    --min_n_layer $MIN_N_LAYER \
    --max_n_layer $MAX_N_LAYER \
    --min_n_head $MIN_N_HEAD \
    --max_n_head $MAX_N_HEAD

echo ""
echo "======================================"
echo "Starting Systematic Grid Search"
echo "n_layer range: ${MIN_N_LAYER} to ${MAX_N_LAYER}"
echo "n_head range: ${MIN_N_HEAD} to ${MAX_N_HEAD}"
echo "Seeds per configuration: ${NUM_SEEDS}"
echo "Total configurations: $((($MAX_N_LAYER - $MIN_N_LAYER + 1) * ($MAX_N_HEAD - $MIN_N_HEAD + 1)))"
echo "Total runs: $((($MAX_N_LAYER - $MIN_N_LAYER + 1) * ($MAX_N_HEAD - $MIN_N_HEAD + 1) * $NUM_SEEDS))"
echo "======================================"
echo ""

# Counter for tracking progress
TOTAL_CONFIGS=$(( ($MAX_N_LAYER - $MIN_N_LAYER + 1) * ($MAX_N_HEAD - $MIN_N_HEAD + 1) ))
CURRENT_CONFIG=0

# Iterate through all combinations of n_layer and n_head
for N_LAYER in $(seq $MIN_N_LAYER $MAX_N_LAYER)
do
    for N_HEAD in $(seq $MIN_N_HEAD $MAX_N_HEAD)
    do
        CURRENT_CONFIG=$((CURRENT_CONFIG + 1))
        EXP_NAME="grid_search_l${N_LAYER}_h${N_HEAD}"
        CONFIG_FILE="configs/grid_configs/grid_l${N_LAYER}_h${N_HEAD}.json"
        FILENAME="experiments/seeds/${EXP_NAME}"

        echo "======================================"
        echo "Configuration ${CURRENT_CONFIG}/${TOTAL_CONFIGS}: n_layer=${N_LAYER}, n_head=${N_HEAD}"
        echo "Using config: ${CONFIG_FILE}"
        echo "======================================"

        # Create or read seeds
        if [ -f $FILENAME ]; then
            echo "${FILENAME} already exists."

            SEEDS=()
            while read -r LINE; do
                SEEDS+=("$LINE")
            done < $FILENAME
            echo "Found ${#SEEDS[@]} seeds for training."
        else
            # Write seeds
            echo "${FILENAME} created. Writing seeds."
            touch $FILENAME
            for (( i = 0; i < $NUM_SEEDS; i++ )) 
            do 
                SEED=$((RANDOM + 1))
                echo $SEED >> $FILENAME
            done

            # Read them
            SEEDS=()
            while read -r LINE; do
                SEEDS+=("$LINE")
            done < $FILENAME
        fi

        # Train with each seed
        SEED_COUNT=0
        for SEED in "${SEEDS[@]}"
        do
            SEED_COUNT=$((SEED_COUNT + 1))
            EXPERIMENT_NAME="${EXP_NAME}_seed_${SEED}"

            echo "Training ${EXPERIMENT_NAME} (run ${SEED_COUNT}/${NUM_SEEDS})..."

            python run.py \
            -e $EXP_NAME -d "datasets" --seed $SEED \
            -r "experiments/results" \
            --batch_size 512 -m 500 -n 64 \
            --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" --wandb_tags "lagllama" "grid_search" "l${N_LAYER}" "h${N_HEAD}" \
            --all_datasets "electricity_hourly" "solar_10_minutes" "wind_farms_without_missing" "uber_tlc_hourly" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "cpu_limit_minute" "function_delay_minute" "instances_minute" "memory_usage_minute" "requests_minute" "ett_h1" "ett_m1" "AirQualityUCI" "weather" "pedestrian_counts" "exchange_rate" "ett_m2" \
            --test_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "requests_minute" \
            --num_workers 4 --args_from_dict_path $CONFIG_FILE --search_batch_size \
            --lr 0.0001 \
            --fits_then_cps \
            --filter_h_order $HARMONIC \
            --filter_energy_threshold $ENERGY_THRESHOLD \
            --early_stopping_patience 30 \
            --num_validation_windows 10 \
            --evaluate_train_split \
            --verbose_processor_info \
            "${ACCELERATOR_FLAG[@]}"

            echo "Completed ${EXPERIMENT_NAME}"
        done

        echo "======================================"
        echo "Finished all runs for n_layer=${N_LAYER}, n_head=${N_HEAD}"
        echo "Progress: ${CURRENT_CONFIG}/${TOTAL_CONFIGS} configurations completed"
        echo "======================================"
        echo ""
    done
done

echo "======================================"
echo "✅ Grid search completed!"
echo "Total configurations trained: ${TOTAL_CONFIGS}"
echo "Total runs: $((TOTAL_CONFIGS * NUM_SEEDS))"
echo "======================================"
