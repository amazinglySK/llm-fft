#!/usr/bin/env bash
set -euo pipefail

# Run minimal training/evaluation passes that exercise the per-batch filtering
# pipeline so that missing frequency mappings are surfaced immediately. The
# script defaults to a small representative subset of datasets, but you can
# pass your own dataset list either as CLI arguments or via the DATASETS
# environment variable (space-delimited).

DATASET_PATH=${DATASET_PATH:-"datasets"}
RESULTS_DIR=${RESULTS_DIR:-"experiments/freq_map_smoke"}
SEED=${SEED:-42}
GPU=${GPU:-0}
WANDB_PROJECT=${WANDB_PROJECT:-"freq-map-smoke"}
ENERGY_THRESHOLD=${ENERGY_THRESHOLD:-0.9}

# Representative coverage across hourly, minute, and daily cadences.
DEFAULT_DATASETS=(
  electricity_hourly
  solar_10_minutes
  AirQualityUCI
  traffic
)

if [[ $# -gt 0 ]]; then
  mapfile -t DATASETS < <(printf '%s\n' "$@")
elif [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206  # Split DATASETS on whitespace intentionally.
  DATASETS=(${DATASETS})
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

mkdir -p "${RESULTS_DIR}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

for DATASET in "${DATASETS[@]}"; do
  echo "==================================================================="
  echo "[freq-map-smoke] Verifying frequency map wiring for ${DATASET}"
  echo "==================================================================="

  EXP_NAME="freq_map_smoke_${DATASET}"

  CMD=(
    python run.py
    -e "${EXP_NAME}"
    -d "${DATASET_PATH}"
    --seed "${SEED}"
    -r "${RESULTS_DIR}"
    --single_dataset "${DATASET}"
    --use_dataset_prediction_length
    --batch_size 64
    -m 1
    -n 1
    --limit_val_batches 1
    --num_validation_windows 2
    --max_epochs 1
    --num_parallel_samples 1
    --num_samples 1
    --num_workers 0
    --gpu "${GPU}"
    --wandb_mode offline
    --wandb_project "${WANDB_PROJECT}"
    --wandb_tags freq-map-smoke filtering freq-check
    --fits_then_cps
    --filter_energy_threshold "${ENERGY_THRESHOLD}"
    --filter_h_order 2
  )

  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    CMD+=(--wandb_entity "${WANDB_ENTITY}")
  fi

  echo "Running: ${CMD[*]}"
  "${CMD[@]}"

  echo "Finished ${DATASET}" && echo
done

echo "All requested frequency smoke checks completed."
