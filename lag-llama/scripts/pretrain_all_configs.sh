#!/usr/bin/env bash
set -euo pipefail

# === USAGE ===
# ./pretrain_all_configs.sh <wandb_entity> <wandb_project> [accelerator]
#
# Example:
# ./pretrain_all_configs.sh myusername lag-llama-config-experiment

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

# Define configs to iterate through
CONFIGS=("lag_llama_conf_1" "lag_llama_conf_2" "lag_llama_conf_3")

NUM_SEEDS=3

# Iterate through each config
for CONFIG_NAME in "${CONFIGS[@]}"
do
    EXP_NAME="pretraining_${CONFIG_NAME}"
    FILENAME="experiments/seeds/${EXP_NAME}"
    CONFIGPATH="configs/${CONFIG_NAME}.json"

    echo "======================================"
    echo "Running experiments with: ${CONFIG_NAME}"
    echo "======================================"

    # Create seeds
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
    for SEED in "${SEEDS[@]}"
    do
        EXPERIMENT_NAME="${EXP_NAME}_seed_${SEED}"

        echo "Training ${EXPERIMENT_NAME}..."

        python run.py \
        -e $EXP_NAME -d "datasets" --seed $SEED \
        -r "experiments/results" \
        --batch_size 512 -m 500 -n 64 \
        --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" --wandb_tags "lagllama" "${CONFIG_NAME}" \
        --all_datasets "electricity_hourly" "solar_10_minutes" "wind_farms_without_missing" "uber_tlc_hourly" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "cpu_limit_minute" "function_delay_minute" "instances_minute" "memory_usage_minute" "requests_minute" "ett_h1" "ett_m1" "AirQualityUCI" "weather" "pedestrian_counts" "exchange_rate" "ett_m2" \
        --test_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "requests_minute" \
        --num_workers 4 --args_from_dict_path $CONFIGPATH --search_batch_size \
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
    echo "Finished all runs for ${CONFIG_NAME}"
    echo "======================================"
    echo ""
done

echo "✅ All configurations completed!"
