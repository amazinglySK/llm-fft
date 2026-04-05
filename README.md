# LLM-FFT Project: Lag-Llama Pretraining with FITS Filtering

This repository contains experiments with Lag-Llama foundation model pretraining using FITS-based frequency filtering techniques.

## Quick Start

### Prerequisites
- Python 3.10+ (recommended: 3.10.8)
- CUDA-compatible GPU (optional but recommended for training)
- 16GB+ RAM
- ~50GB disk space for datasets

### 1. Environment Setup

First, run the automated setup script to create a virtual environment and download datasets:

```bash
# From the repository root
chmod +x setup.sh
./setup.sh
```

**What this script does:**
- Creates a virtual environment named `llmfft`
- Installs required dependencies from `lag-llama/requirements.txt`
- Downloads and extracts the non-Monash datasets (~2GB compressed)
- Sets up the `lag-llama/datasets/` directory structure

**Manual activation later:**
```bash
source llmfft/bin/activate
```

### 2. Configure Weights & Biases (Required)

Before running pretraining, you need to set up Weights & Biases for experiment tracking:

1. **Sign up** at [wandb.ai](https://wandb.ai) if you don't have an account
2. **Get your API key** from [wandb.ai/settings](https://wandb.ai/settings)
3. **Login locally:**
   ```bash
   source llmfft/bin/activate
   wandb login
   # Paste your API key when prompted
   ```

## Pretraining Options

### Option 1: Clean Replication (Paper Results)

Replicate the exact pretraining setup from the Lag-Llama paper:

```bash
source llmfft/bin/activate
cd lag-llama

# Run standard pretraining
./scripts/pretrain.sh YOUR_WANDB_USERNAME YOUR_PROJECT_NAME
```

**Example:**
```bash
./scripts/pretrain.sh john_doe lag-llama-reproduction
```

**Training details:**
- **Datasets:** 28 time series datasets (australian_electricity_demand, electricity_hourly, etc.)
- **Test datasets:** 7 held-out datasets for evaluation
- **Batch size:** 512
- **Max epochs:** 1000 
- **Learning rate:** 1e-4
- **Architecture:** 8 layers, 9 heads, 16 embedding per head
- **Context length:** 32 (as per config)

### Option 2: FITS-Filtered Pretraining

Train Lag-Llama using the FITS frequency filtering pipeline:

```bash
source llmfft/bin/activate
cd lag-llama

# Create custom pretrain script for FITS filtering
cp scripts/pretrain.sh scripts/pretrain_fits.sh
```

Edit `scripts/pretrain_fits.sh` and add the FITS filtering flag to the `python run.py` command:

```bash
# In scripts/pretrain_fits.sh, modify the python run.py line:
python run.py \
-e $EXP_NAME -d "datasets" --seed $SEED \
-r "experiments/results" \
--batch_size 512 -m 1000 -n 128 \
--wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" --wandb_tags "$WANDB_TAGS" \
--all_datasets "australian_electricity_demand" "electricity_hourly" "london_smart_meters_without_missing" "solar_10_minutes" "wind_farms_without_missing" "pedestrian_counts" "uber_tlc_hourly" "traffic" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "exchange_rate" "cpu_limit_minute" "cpu_usage_minute" "function_delay_minute" "instances_minute" "memory_limit_minute" "memory_usage_minute" "platform_delay_minute" "requests_minute" "ett_h1" "ett_h2" "ett_m1" "ett_m2" "beijing_pm25" "AirQualityUCI" "beijing_multisite" "weather" \
--test_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "platform_delay_minute" "requests_minute" "beijing_pm25" \
--num_workers 2 --args_from_dict_path $CONFIGPATH --search_batch_size \
--lr 0.0001 \
--fits_then_cps \
--filter_base_period 24 \
--filter_h_order 2 \
--filter_energy_threshold 0.9
```

Then run:
```bash
./scripts/pretrain_fits.sh YOUR_WANDB_USERNAME YOUR_PROJECT_NAME
```

## Filtering Options Available

The following filtering methods are now supported via command line:

### Basic Filters
- `--lpf`: Simple low-pass filter (cutoff controlled by `--filter_dropout_rate`)
- `--butterworth`: Butterworth low-pass filter (order controlled by `--filter_butter_order`)

### Advanced FITS-based Filters
- `--fits_filter`: FITS-style filtering based on sequence length and base period
- `--cumulative_power_filter`: Filtering based on cumulative power spectrum analysis  
- `--fits_then_cps`: **Recommended** - FITS → energy ratio → CPS pipeline

### Filter Parameters
- `--filter_dropout_rate`: Cutoff ratio (default: 0.2)
- `--filter_butter_order`: Butterworth filter order (default: 4)
- `--filter_base_period`: Base period for FITS (default: auto-infer from frequency)
- `--filter_h_order`: Harmonic order for FITS (default: 2)  
- `--filter_energy_threshold`: Energy threshold for CPS (default: 0.9)

## Monitoring Training

### Weights & Biases Dashboard
- Navigate to your W&B project: `https://wandb.ai/YOUR_USERNAME/YOUR_PROJECT_NAME`
- Monitor training loss, validation metrics, and system resources
- Compare runs between clean and FITS-filtered pretraining

### Local Monitoring
```bash
# Check GPU usage
nvidia-smi

# Monitor training logs
tail -f experiments/results/pretraining_lag_llama/*/wandb/latest-run/logs/debug.log
```

## Expected Training Time

**Hardware requirements vary significantly:**

| Setup | Time | Hardware |
|-------|------|----------|
| Clean pretraining | 2-4 days | 8x V100 (32GB) |
| FITS pretraining | 2-5 days | 8x V100 (32GB) |
| Single GPU | 2-3 weeks | 1x RTX 4090 |
| CPU only | Not recommended | Very slow |

**Memory requirements:**
- Minimum: 16GB GPU memory
- Recommended: 32GB GPU memory
- Batch size scales with available memory

## Output Files and Checkpoints

After training completes, you'll find:

```
experiments/
├── results/
│   └── pretraining_lag_llama/
│       └── SEED/
│           ├── checkpoints/
│           │   └── best-model.ckpt      # Best checkpoint
│           ├── wandb/                   # W&B logs
│           └── test_results/            # Evaluation results
└── seeds/
    └── pretraining_lag_llama           # Random seeds used
```

**Key files:**
- `best-model.ckpt`: Use this for downstream fine-tuning or inference
- `test_results/`: Contains evaluation metrics on held-out test datasets

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size in scripts/pretrain.sh
   --batch_size 256  # Instead of 512
   ```

2. **Dataset Download Fails**
   ```bash
   # Manual download
   cd lag-llama/datasets
   wget "https://drive.google.com/uc?id=1JrDWMZyoPsc6d1wAAjgm3PosbGus-jCE" -O nonmonash_datasets.tar.gz
   tar -xzf nonmonash_datasets.tar.gz
   ```

3. **CUDA Version Mismatch**
   ```bash
   # Install PyTorch for your CUDA version
   pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
   ```

4. **W&B Authentication**
   ```bash
   # Re-authenticate
   wandb logout
   wandb login
   ```

### Performance Tips

- **Multi-GPU:** The pretrain script automatically uses all available GPUs
- **Memory optimization:** Use `--search_batch_size` flag to auto-find optimal batch size
- **Resume training:** Scripts automatically resume from checkpoints if they exist

## Grid Search: Identifying the Best Architecture

### Running the Systematic Grid Search

```bash
cd lag-llama
./scripts/pretrain_systematic_grid.sh <wandb_entity> <wandb_project>
```

This sweeps `n_layer` (3–8) × `n_head` (3–9) with 5 seeds each and logs all results to W&B under the `optim-grid` project.

### Analysing Results

Open `analysis/optim_grid_summary.ipynb` and run all cells.  
The last section **"Best Model in the 0.5M – 1M Parameter Range"** automatically:

1. Filters models whose parameter count falls between 500 k and 1 M.
2. Ranks them by average test CRPS (lower is better).
3. Prints the winning `n_layer` / `n_head` pair and the exact command to re-run it.

## Frequency-Cutoff Experiments for the Best Architecture

Once you have identified the best model from the grid search, run frequency-cutoff experiments to explore how different levels of high-frequency filtering affect its performance.

### What the scripts do

| Script | Purpose |
|--------|---------|
| `scripts/generate_freq_cutoff_config.py` | Writes a single JSON config with the chosen `n_layer` / `n_head` |
| `scripts/run_best_model_freq_cutoff.sh` | Runs 5 seeds × 2 `h_order` values (0.5 and 0.75) for the target architecture |

The `h_order` parameter controls the frequency cutoff:  
`cut_freq = (seq_len // base_period + 1) × h_order + 10`

| `h_order` | Effect |
|-----------|--------|
| 2.0 | Original default – least filtering |
| 1.0 | Grid-search baseline – moderate filtering |
| 0.75 | More aggressive cutoff |
| 0.5 | Most aggressive cutoff (fewest high-freq components kept) |

### Usage

```bash
cd lag-llama

# Example: best model found was n_layer=5, n_head=5
./scripts/run_best_model_freq_cutoff.sh <wandb_entity> <wandb_project> 5 5

# Force CPU execution (slow, for testing only)
./scripts/run_best_model_freq_cutoff.sh <wandb_entity> <wandb_project> 5 5 cpu
```

This produces **10 runs** (2 h\_order values × 5 seeds), all logged to W&B with tags  
`lagllama`, `freq_cutoff`, `l<n_layer>`, `h<n_head>`, `harmonic_<value>`.

### Override precompute limits (optional)

```bash
PRECOMPUTE_MAX_WINDOWS_PER_DATASET=1024 PRECOMPUTE_MEMORY_CAP_MB=512 \
  ./scripts/run_best_model_freq_cutoff.sh myusername my-project 5 5
```

## Comparing Results

### Metrics to Track

1. **Training Loss:** Should decrease steadily
2. **Validation CRPS:** Lower is better (primary metric)
3. **Test CRPS:** Final performance on held-out datasets

### Expected Performance

| Method | Test CRPS (Weather) | Test CRPS (ETTm2) | Notes |
|--------|-------------------|-------------------|--------|
| Original Lag-Llama | ~0.45 | ~0.38 | Paper baseline |
| FITS-filtered | TBD | TBD | Experimental |

## Next Steps

After pretraining completes:

1. **Evaluate zero-shot performance** using the [Colab Demo](https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?usp=sharing)
2. **Fine-tune on specific datasets** using `scripts/finetune.sh`
3. **Compare filtering methods** by running multiple experiments

## References

- **Lag-Llama Paper:** [Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2310.08278)
- **FITS Paper:** [FITS: Modeling Time Series with 10k Parameters](https://arxiv.org/abs/2307.03756)
- **Original Repository:** [time-series-foundation-models/lag-llama](https://github.com/time-series-foundation-models/lag-llama)

---

**Questions?** Open an issue or check the troubleshooting section above.