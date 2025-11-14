# Summary of Changes to lag-llama/data/data_utils.py

## New Filtering Functions Added

### 1. FITS-based Filter
```python
def fits_based_filter(x, seq_len=None, base_period=24, h_order=2, dim=0):
```
- **Purpose**: Implements the same low-pass filtering mechanism used in FITS
- **Cutoff Calculation**: `cut_freq = int(seq_len // base_period + 1) * h_order + 10`
- **Default Parameters**: 
  - `base_period=24` (for hourly data)
  - `h_order=2` (harmonic order multiplier)
- **Method**: Zeros out frequencies above the calculated cutoff in the FFT domain

### 2. Cumulative Power Spectrum Filter
```python
def cumulative_power_spectrum_filter(x, energy_threshold=0.9, dim=0):
```
- **Purpose**: Preserves frequencies that contain the highest cumulative power
- **Method**: 
  - Calculates power spectrum (magnitude squared of FFT)
  - Sorts frequencies by power (descending)
  - Preserves frequencies until reaching the energy threshold (default 90%)
- **Advantage**: Adaptively preserves the most important frequencies for each signal

### 3. Enhanced freq_dropout Function
```python
def freq_dropout(x, y=None, dropout_rate=0.2, dim=0, keep_dominant=True):
```
- **Improvements**: 
  - Added `keep_dominant` parameter to preserve top 10% of frequencies
  - Similar to FITS approach of preserving important frequencies
  - Prevents dropout of dominant frequency components

## Updated Function Signature

The main data loading function now includes new parameters:

```python
def create_train_and_val_datasets_with_dates(
    # ... existing parameters ...
    fits_filter = False,
    cumulative_power_filter = False,
    base_period = 24,
    h_order = 2,
    energy_threshold = 0.9,
):
```

## New Parameters

- `fits_filter`: Enable FITS-style filtering
- `cumulative_power_filter`: Enable cumulative power spectrum filtering
- `base_period`: Base period for FITS filter (default: 24 for hourly data)
- `h_order`: Harmonic order multiplier for FITS filter (default: 2)
- `energy_threshold`: Energy preservation threshold for power spectrum filter (default: 0.9)

## Updated Filtering Logic

The filtering logic in both training and validation data sections now supports all filter types:

```python
# Apply filtering based on the specified method
if lpf:
    target = low_pass_filter(target, cutoff_ratio=dropout_rate) 
elif butterworth:
    target = butterworth_lowpass_filter(target, butter_cutoff, butter_fs, butter_order)
elif fits_filter:
    target = fits_based_filter(target, seq_len=len(target), base_period=base_period, h_order=h_order)
elif cumulative_power_filter:
    target = cumulative_power_spectrum_filter(target, energy_threshold=energy_threshold)
```

## Usage Examples

### FITS-style Filtering
```python
train_data, val_data, ... = create_train_and_val_datasets_with_dates(
    name="ett_h1",
    dataset_path="./datasets",
    data_id=0,
    history_length=336,
    fits_filter=True,
    base_period=24,  # hourly data
    h_order=2
)
```

### Cumulative Power Spectrum Filtering
```python
train_data, val_data, ... = create_train_and_val_datasets_with_dates(
    name="ett_h1",
    dataset_path="./datasets", 
    data_id=0,
    history_length=336,
    cumulative_power_filter=True,
    energy_threshold=0.95  # preserve 95% of energy
)
```

## Key Differences from Original Functions

### freq_dropout vs FITS approach:
- **freq_dropout**: Randomly masks frequencies based on dropout rate
- **FITS**: Deterministically removes high frequencies above a calculated cutoff
- **New enhancement**: freq_dropout now optionally preserves dominant frequencies

### butterworth_lowpass_filter vs FITS approach:
- **Butterworth**: Uses traditional signal processing with smooth frequency response
- **FITS**: Uses sharp cutoff in frequency domain (brick-wall filter)
- **FITS** is simpler and more direct for time series forecasting

### New cumulative_power_filter:
- **Adaptive**: Cutoff varies per signal based on actual frequency content
- **Energy-preserving**: Maintains specified percentage of signal energy
- **Data-driven**: Automatically finds the most important frequencies

## Benefits

1. **FITS compatibility**: Can now apply the exact same filtering as FITS models
2. **Adaptive filtering**: Cumulative power spectrum adapts to each signal's characteristics
3. **Energy preservation**: Maintains most important frequency components
4. **Flexibility**: Multiple filtering options can be selected based on use case
5. **Improved performance**: Should provide better filtering for time series forecasting tasks