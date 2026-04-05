#!/usr/bin/env bash
# PLEASE FOLLOW THE BELOW INSTRUCTIONS FIRST
#
# 1. Install the requirements. It is recommend to use a new Anaconda environment with Python 3.10.8.
#    !pip install -r requirements.txt
#
# 2. Download datasets:
#    tar -xvzf nonmonash_datasets.tar.gz -C datasets
#
# 3. Edit the Weights and Biases arguments passed on the command line (see USAGE below).

set -euo pipefail

# === USAGE ===
# Run 5-seed frequency-cutoff experiments for the best model architecture found
# in the optim-grid search.  Tests h_order values [0.5, 0.75] (more aggressive
# filtering than the grid-search baseline of h_order=1.0).
#
# ./scripts/run_best_model_freq_cutoff.sh <wandb_entity> <wandb_project> <n_layer> <n_head> [accelerator]
#
# Example (best model identified by the notebook – n_layer=5, n_head=5):
#   ./scripts/run_best_model_freq_cutoff.sh myusername lag-llama-freq-cutoff 5 5
#
# To force CPU execution:
#   ./scripts/run_best_model_freq_cutoff.sh myusername lag-llama-freq-cutoff 5 5 cpu

if [ "$#" -lt 4 ]; then
  echo "❌  Error: You must provide Weights & Biases entity, project name, n_layer, and n_head."
  echo "Usage: $0 <wandb_entity> <wandb_project> <n_layer> <n_head> [accelerator]"
  exit 1
fi

WANDB_ENTITY="$1"
WANDB_PROJECT="$2"
N_LAYER="$3"
N_HEAD="$4"
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

# ── Filter settings ────────────────────────────────────────────────────────────
# Lower h_order = more aggressive high-frequency cutoff.
# Grid search used h_order=1.0; we test below that threshold here.
H_ORDERS=(0.5 0.75)
ENERGY_THRESHOLD=0.9

# ── Reproducibility / efficiency settings ─────────────────────────────────────
NUM_SEEDS=5
PRECOMPUTE_MAX_WINDOWS_PER_DATASET="${PRECOMPUTE_MAX_WINDOWS_PER_DATASET:-2048}"
PRECOMPUTE_MEMORY_CAP_MB="${PRECOMPUTE_MEMORY_CAP_MB:-1024}"

# ── Paths ──────────────────────────────────────────────────────────────────────
CONFIG_DIR="configs/freq_cutoff_configs"
CONFIG_FILE="${CONFIG_DIR}/freq_cutoff_l${N_LAYER}_h${N_HEAD}.json"

mkdir -p experiments
mkdir -p experiments/seeds
mkdir -p experiments/results

# ── Generate the architecture config once ─────────────────────────────────────
echo "======================================"
echo "Generating config for n_layer=${N_LAYER}, n_head=${N_HEAD}..."
echo "======================================"
python scripts/generate_freq_cutoff_config.py \
    --n_layer "$N_LAYER" \
    --n_head  "$N_HEAD"  \
    --base_config configs/lag_llama.json \
    --output_dir  "$CONFIG_DIR"

echo ""
echo "======================================"
echo "Frequency-cutoff experiment"
echo "  Architecture : n_layer=${N_LAYER}, n_head=${N_HEAD}"
echo "  h_order sweep: ${H_ORDERS[*]}"
echo "  energy thresh: ${ENERGY_THRESHOLD}"
echo "  Seeds per run: ${NUM_SEEDS}"
echo "  Total runs   : $(( ${#H_ORDERS[@]} * NUM_SEEDS ))"
echo "======================================"
echo ""

# ── Loop over h_order values ───────────────────────────────────────────────────
for HARMONIC in "${H_ORDERS[@]}"; do

    HARMONIC_STR=$(echo "$HARMONIC" | tr '.' '_')
    THRESHOLD_STR=$(echo "$ENERGY_THRESHOLD" | tr '.' '_')
    EXP_NAME="freq_cutoff_l${N_LAYER}_h${N_HEAD}_harmonic_${HARMONIC_STR}_threshold_${THRESHOLD_STR}"
    SEED_FILE="experiments/seeds/${EXP_NAME}"

    echo "======================================"
    echo "h_order = ${HARMONIC}  (experiment: ${EXP_NAME})"
    echo "======================================"

    # Create or reuse seed file so reruns are reproducible
    if [ -f "$SEED_FILE" ]; then
        echo "${SEED_FILE} already exists – reusing seeds."
        SEEDS=()
        while read -r LINE; do
            SEEDS+=("$LINE")
        done < "$SEED_FILE"
        echo "Found ${#SEEDS[@]} seeds."
    else
        echo "${SEED_FILE} created. Writing ${NUM_SEEDS} seeds."
        touch "$SEED_FILE"
        for (( i = 0; i < NUM_SEEDS; i++ )); do
            SEED=$((RANDOM + 1))
            echo "$SEED" >> "$SEED_FILE"
        done
        SEEDS=()
        while read -r LINE; do
            SEEDS+=("$LINE")
        done < "$SEED_FILE"
    fi

    SEED_COUNT=0
    for SEED in "${SEEDS[@]}"; do
        SEED_COUNT=$((SEED_COUNT + 1))
        echo ""
        echo "  Seed ${SEED_COUNT}/${NUM_SEEDS}: ${EXP_NAME}_seed_${SEED}"

        python run.py \
            -e "$EXP_NAME" -d "datasets" --seed "$SEED" \
            -r "experiments/results" \
            --batch_size 512 -m 500 -n 64 \
            --wandb_entity "$WANDB_ENTITY" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_tags "lagllama" "freq_cutoff" "extra_analysis" \
                         "l${N_LAYER}" "h${N_HEAD}" \
                         "harmonic_${HARMONIC_STR}" \
                         "threshold_${THRESHOLD_STR}" \
            --all_datasets \
                "electricity_hourly" "solar_10_minutes" "wind_farms_without_missing" \
                "uber_tlc_hourly" "kdd_cup_2018_without_missing" "saugeenday" \
                "sunspot_without_missing" "cpu_limit_minute" "function_delay_minute" \
                "instances_minute" "memory_usage_minute" "requests_minute" \
                "ett_h1" "ett_m1" "AirQualityUCI" "weather" "pedestrian_counts" \
                "exchange_rate" "ett_m2" \
            --test_datasets \
                "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "requests_minute" \
            --num_workers 4 \
            --args_from_dict_path "$CONFIG_FILE" \
            --search_batch_size \
            --lr 0.0001 \
            --fits_then_cps \
            --filter_h_order "$HARMONIC" \
            --filter_energy_threshold "$ENERGY_THRESHOLD" \
            --early_stopping_patience 30 \
            --num_validation_windows 10 \
            --evaluate_train_split \
            --precompute_train_filtered_windows \
            --precompute_max_windows_per_dataset "$PRECOMPUTE_MAX_WINDOWS_PER_DATASET" \
            --precompute_memory_cap_mb "$PRECOMPUTE_MEMORY_CAP_MB" \
            --precompute_seed "$SEED" \
            "${ACCELERATOR_FLAG[@]}"

        echo "  Completed seed ${SEED_COUNT}/${NUM_SEEDS}"
    done

    echo ""
    echo "======================================"
    echo "Finished all ${NUM_SEEDS} runs for h_order=${HARMONIC}"
    echo "======================================"
    echo ""
done

echo "======================================"
echo "✅  All frequency-cutoff experiments completed!"
echo "  Architecture : n_layer=${N_LAYER}, n_head=${N_HEAD}"
echo "  h_order values tested: ${H_ORDERS[*]}"
echo "  Total runs   : $(( ${#H_ORDERS[@]} * NUM_SEEDS ))"
echo "======================================"
