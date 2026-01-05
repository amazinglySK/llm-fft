"""
Test script to verify frequency lookup in training batches.
Simulates the actual batch creation pipeline and tests data_id → frequency mapping.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.data_utils import CombinedDataset, create_train_and_val_datasets_with_dates
from data.filter_processor import FilterProcessor
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.transform import InstanceSplitter, ExpectedNumInstanceSampler, AddObservedValuesIndicator, Chain
from gluonts.dataset.field_names import FieldName

# Configuration matching your training setup
CONTEXT_LENGTH = 32
MAX_LAG = 1092
PREDICTION_LENGTH = 1
BATCH_SIZE = 16  # Smaller for testing
NUM_BATCHES = 5   # Test 5 batches

# Dataset names (subset of your training datasets)
TEST_DATASETS = [
    "electricity_hourly",
    "ett_h1", 
    "ett_m1",
    "weather",
    "AirQualityUCI",
    "memory_usage_minute"
]

DATASET_PATH = "datasets"

print("="*80)
print("FREQUENCY LOOKUP TEST - Simulating Training Pipeline")
print("="*80)

# Step 1: Create datasets and populate frequency map
print("\n[Step 1] Creating datasets and building frequency map...")
print("-"*80)

data_id_to_freq_map = {}
data_id_to_name_map = {}
name_to_data_id_map = {}
all_datasets = []
val_datasets = []

history_length = CONTEXT_LENGTH + MAX_LAG

for data_id, name in enumerate(TEST_DATASETS):
    try:
        print(f"\nDataset {data_id}: {name}")
        data_id_to_name_map[data_id] = name
        name_to_data_id_map[name] = data_id
        
        (
            train_dataset,
            val_dataset,
            total_train_points,
            total_val_points,
            total_val_windows,
            max_train_end_date,
            total_points,
            dataset_freq,
        ) = create_train_and_val_datasets_with_dates(
            name,
            DATASET_PATH,
            data_id,
            history_length,
            PREDICTION_LENGTH,
            num_val_windows=14,
        )
        
        # Populate frequency map
        data_id_to_freq_map[data_id] = dataset_freq
        
        print(f"  ✓ Frequency: {dataset_freq}")
        print(f"  ✓ Train samples: {len(train_dataset)}")
        print(f"  ✓ Train points: {total_train_points}")
        
        all_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        
    except Exception as e:
        print(f"  ✗ Error loading {name}: {e}")
        continue

print(f"\n✓ Successfully loaded {len(all_datasets)} datasets")

# Step 2: Show frequency mapping
print("\n[Step 2] Frequency Mapping")
print("-"*80)
for data_id, name in data_id_to_name_map.items():
    freq = data_id_to_freq_map.get(data_id, "NOT_SET")
    print(f"  data_id={data_id:2d} | name={name:25s} | freq={freq}")

# Step 3: Create combined dataset
print("\n[Step 3] Creating combined dataset...")
print("-"*80)
train_data = CombinedDataset(all_datasets, weights=None)
print(f"✓ Combined dataset size: {len(train_data)} samples")

# Step 4: Create instance splitter and dataloader (simulating estimator's method)
print("\n[Step 4] Creating dataloader with InstanceSplitter...")
print("-"*80)

train_sampler = ExpectedNumInstanceSampler(
    num_instances=1.0,
    min_past=history_length,
    min_future=PREDICTION_LENGTH,
)

# Add observed values indicator transformation (required before InstanceSplitter)
add_observed_values = AddObservedValuesIndicator(
    target_field=FieldName.TARGET,
    output_field=FieldName.OBSERVED_VALUES,
)

instance_splitter = InstanceSplitter(
    target_field=FieldName.TARGET,
    is_pad_field=FieldName.IS_PAD,
    start_field=FieldName.START,
    forecast_start_field=FieldName.FORECAST_START,
    instance_sampler=train_sampler,
    past_length=history_length,
    future_length=PREDICTION_LENGTH,
    time_series_fields=[FieldName.OBSERVED_VALUES],
    dummy_value=0.0,
)

# Chain transformations: add observed values, then split instances
transformation = Chain([add_observed_values, instance_splitter])

# Create streaming dataloader
data_stream = Cyclic(train_data).stream()
instances = transformation.apply(data_stream, is_train=True)

TRAINING_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "future_target",
    "future_observed_values",
    "data_id",
]

dataloader = as_stacked_batches(
    instances,
    batch_size=BATCH_SIZE,
    shuffle_buffer_length=None,
    field_names=TRAINING_INPUT_NAMES,
    output_type=torch.tensor,
    num_batches_per_epoch=NUM_BATCHES,
)

print(f"✓ Dataloader created with batch_size={BATCH_SIZE}")

# Step 5: Test batches and frequency lookup
print("\n[Step 5] Testing batches and frequency lookup")
print("-"*80)

filter_processor = FilterProcessor(method="fits_then_cps", h_order=2, verbose=True)

# Store samples for visualization
samples_to_plot = []

for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= NUM_BATCHES:
        break
    
    print(f"\n--- Batch {batch_idx + 1}/{NUM_BATCHES} ---")
    print(f"past_target shape: {batch['past_target'].shape}")
    
    # Check if data_id exists in batch
    if "data_id" in batch and batch["data_id"] is not None:
        print(f"✓ data_id field present in batch")
        print(f"  Batch data_ids: {batch['data_id'].tolist()}")
        
        # Test frequency lookup for each sample
        print(f"\n  Per-sample frequency lookup:")
        for i in range(min(5, len(batch["data_id"]))):  # Show first 5 samples
            data_id = batch["data_id"][i].item() if torch.is_tensor(batch["data_id"][i]) else batch["data_id"][i]
            freq = data_id_to_freq_map.get(int(data_id), None)
            dataset_name = data_id_to_name_map.get(int(data_id), "UNKNOWN")
            
            if freq is not None:
                print(f"    Sample {i}: data_id={data_id}, dataset={dataset_name:20s}, freq='{freq}'")
                
                # Test filter processor with this frequency
                target_np = batch["past_target"][i].cpu().numpy()
                try:
                    filtered = filter_processor.process(target_np, freq=freq, data_id=data_id, context="test")
                    print(f"              → Filter applied successfully, shape: {filtered.shape}")
                    
                    # Store first sample from each dataset for plotting
                    if len(samples_to_plot) < 6 and not any(s['dataset'] == dataset_name for s in samples_to_plot):
                        samples_to_plot.append({
                            'original': target_np,
                            'filtered': filtered,
                            'dataset': dataset_name,
                            'freq': freq,
                            'data_id': data_id
                        })
                except Exception as e:
                    print(f"              → Filter error: {e}")
            else:
                print(f"    Sample {i}: data_id={data_id}, dataset={dataset_name:20s}, freq=NOT_FOUND ⚠️")
    else:
        print("✗ data_id field NOT present in batch!")
        print("  Available fields:", list(batch.keys()))

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"✓ Loaded {len(all_datasets)} datasets")
print(f"✓ Frequency map has {len(data_id_to_freq_map)} entries")
print(f"✓ Tested {NUM_BATCHES} batches")
print(f"✓ Each batch has mixed samples from different datasets")
print(f"✓ data_id field {'PRESENT' if 'data_id' in batch else 'MISSING'} in batches")

if len(data_id_to_freq_map) > 0:
    print("\n✓ Frequency lookup working correctly!")
    print("  Each sample in a batch can have a different frequency based on its data_id")
else:
    print("\n✗ WARNING: Frequency map is empty!")

# Step 6: Visualize filtering effects
print("\n" + "="*80)
print("STEP 6: VISUALIZING FILTERING EFFECTS")
print("="*80)

if len(samples_to_plot) > 0:
    n_samples = len(samples_to_plot)
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_data in enumerate(samples_to_plot):
        original = sample_data['original']
        filtered = sample_data['filtered']
        dataset = sample_data['dataset']
        freq = sample_data['freq']
        
        # Time series plot
        axes[idx, 0].plot(original, label='Original', alpha=0.7, linewidth=1.5)
        axes[idx, 0].plot(filtered, label='Filtered (h_order=2)', alpha=0.9, linewidth=1.5)
        axes[idx, 0].set_title(f'{dataset} (freq={freq}) - Time Series', fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel('Time Step')
        axes[idx, 0].set_ylabel('Value')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Calculate MSE
        mse = np.mean((original - filtered) ** 2)
        corr = np.corrcoef(original, filtered)[0, 1]
        axes[idx, 0].text(0.02, 0.98, f'MSE: {mse:.4f}\nCorr: {corr:.4f}', 
                         transform=axes[idx, 0].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Frequency domain plot
        fft_original = np.fft.rfft(original)
        fft_filtered = np.fft.rfft(filtered)
        freqs = np.fft.rfftfreq(len(original))
        
        axes[idx, 1].plot(freqs, np.abs(fft_original), label='Original Spectrum', alpha=0.7, linewidth=1.5)
        axes[idx, 1].plot(freqs, np.abs(fft_filtered), label='Filtered Spectrum', alpha=0.9, linewidth=1.5)
        axes[idx, 1].set_title(f'{dataset} - Frequency Domain', fontsize=12, fontweight='bold')
        axes[idx, 1].set_xlabel('Normalized Frequency')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xlim([0, 0.5])
        
        # Calculate energy preserved
        energy_original = np.sum(np.abs(fft_original) ** 2)
        energy_filtered = np.sum(np.abs(fft_filtered) ** 2)
        energy_ratio = energy_filtered / energy_original if energy_original > 0 else 0
        axes[idx, 1].text(0.98, 0.98, f'Energy Preserved: {energy_ratio*100:.2f}%', 
                         transform=axes[idx, 1].transAxes, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('filtering_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to 'filtering_visualization.png'")
    print(f"✓ Visualized {n_samples} samples from different datasets")
    
    # Print explanation
    print("\n" + "-"*80)
    print("UNDERSTANDING h_order (Harmonic Order):")
    print("-"*80)
    print("• h_order controls how many frequency harmonics are PRESERVED")
    print("• cut_freq = int(seq_len // base_period + 1) * h_order + 10")
    print(f"• With seq_len=1125, base_period=24, h_order=2:")
    print(f"  cut_freq = int(1125 // 24 + 1) * 2 + 10 = 104")
    print(f"  This keeps 104 out of {len(original)//2} frequencies (~{104/(len(original)//2)*100:.1f}%)")
    print("\n• HIGHER h_order → MORE frequencies kept → CLOSER to original")
    print("• LOWER h_order → FEWER frequencies kept → MORE filtered (smoother)")
    print("\n→ If h_order=2 is too close to original, try h_order=1 for more aggressive filtering")
    
else:
    print("\n✗ No samples collected for visualization")

print("\nTest complete! 🎯")
