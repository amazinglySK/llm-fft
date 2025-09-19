# PLEASE FOLLOW THE BELOW INSTRUCTIONS FIRST

# 1. Install the requirements. It is recommend to use a new Anaconda environment with Python 3.10.8. Execute the below command (remove the #)
# !pip install -r requirements.txt

# 2. Please download https://drive.google.com/file/d/1JrDWMZyoPsc6d1wAAjgm3PosbGus-jCE/view?usp=sharing and use the below command to download the non-monash datasets (remove the #)
# tar -xvzf nonmonash_datasets.tar.gz -C datasets

# 3. Edit the Weights and Biases arguments on line 59 of this script

#!/usr/bin/env bash
set -euo pipefail

# === USAGE ===
# ./pretrain.sh <wandb_entity> <wandb_project>
#
# Example:
# ./pretrain.sh myusername lag-llama-pretrain

if [ "$#" -lt 2 ]; then
  echo "❌ Error: You must provide Weights & Biases entity and project name."
  echo "Usage: $0 <wandb_entity> <wandb_project>"
  exit 1
fi

WANDB_ENTITY="$1"
WANDB_PROJECT="$2"
WANDB_TAGS="lagllama"

mkdir -p experiments
mkdir -p experiments/seeds
mkdir -p experiments/results

EXP_NAME="pretraining_lag_llama"
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

# Train
for SEED in "${SEEDS[@]}"
do
    EXPERIMENT_NAME="${EXP_NAME}_seed_${SEED}"

    python run.py \
    -e $EXP_NAME -d "datasets" --seed $SEED \
    -r "experiments/results" \
    --batch_size 512 -m 1000 -n 128 \
    --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" --wandb_tags "$WANDB_TAGS" \
    --all_datasets "australian_electricity_demand" "electricity_hourly" "london_smart_meters_without_missing" "solar_10_minutes" "wind_farms_without_missing" "pedestrian_counts" "uber_tlc_hourly" "traffic" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "exchange_rate" "cpu_limit_minute" "cpu_usage_minute" "function_delay_minute" "instances_minute" "memory_limit_minute" "memory_usage_minute" "platform_delay_minute" "requests_minute" "ett_h1" "ett_h2" "ett_m1" "ett_m2" "beijing_pm25" "AirQualityUCI" "beijing_multisite" "weather" \
    --test_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "platform_delay_minute" "requests_minute" "beijing_pm25" \
    --num_workers 2 --args_from_dict_path $CONFIGPATH --search_batch_size \
    --lr 0.0001
done
