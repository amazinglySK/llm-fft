# PLEASE FOLLOW THE BELOW INSTRUCTIONS FIRST

# 1. Install the requirements. It is recommend to use a new Anaconda environment with Python 3.10.8. Execute the below command (remove the #)
# !pip install -r requirements.txt

# 2. Please download https://drive.google.com/file/d/1JrDWMZyoPsc6d1wAAjgm3PosbGus-jCE/view?usp=sharing and use the below command to download the non-monash datasets (remove the #)
# tar -xvzf nonmonash_datasets.tar.gz -C datasets

# 3. Edit the Weights and Biases arguments on line 59 of this script

#!/usr/bin/env bash
set -euo pipefail

# === USAGE ===
# ./pretrain_with_fits_then_cps.sh <wandb_entity> <wandb_project> <energy_threshold> [accelerator]
#
# Example:
# ./pretrain_with_fits_then_cps.sh myusername lag-llama-fits-experiment 0.9

if [ "$#" -lt 4 ]; then
  echo "❌ Error: You must provide Weights & Biases entity, project name, harmonic, and energy threshold."
    echo "Usage: $0 <wandb_entity> <wandb_project> <harmonic> <energy_threshold> [accelerator]"
  exit 1
fi

WANDB_ENTITY="$1"
WANDB_PROJECT="$2"
HARMONIC="$3"
ENERGY_THRESHOLD="$4"
ACCELERATOR_FLAG=()

if [ "$#" -ge 5 ]; then
    ACCELERATOR="$5"
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

# Convert harmonic to string suitable for filename (replace . with _)
HARMONIC_STR=$(echo "$HARMONIC" | tr '.' '_')
THRESHOLD_STR=$(echo "$ENERGY_THRESHOLD" | tr '.' '_')
EXP_NAME="pretraining_lag_llama_filtered_harmonic_${HARMONIC_STR}_threshold_${THRESHOLD_STR}"
FILENAME="experiments/seeds/${EXP_NAME}"
CONFIGPATH="configs/lag_llama.json"

echo $EXP_NAME

NUM_SEEDS=1

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

# Train with FITS-then-CPS filtering
for SEED in "${SEEDS[@]}"
do
    EXPERIMENT_NAME="${EXP_NAME}_seed_${SEED}"

    python run.py \
    -e $EXP_NAME -d "datasets" --seed $SEED \
    -r "experiments/results" \
    --batch_size 512 -m 500 -n 64 \
    --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" --wandb_tags "lagllama" "filtered" "harmonic_${HARMONIC_STR}" "threshold_${THRESHOLD_STR}" "faster" \
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
done