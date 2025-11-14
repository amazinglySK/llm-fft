# FITS-Style Filtering Demo - Graph Explanations

This README explains each graph in the `filter_demo.ipynb` notebook in simple terms. The notebook demonstrates different ways to filter time series data to remove noise while keeping important patterns.

## What is Filtering?

Think of filtering like cleaning up a noisy recording. Imagine you're listening to music on a radio with lots of static - filtering helps remove the static while keeping the music clear. In time series data, filtering removes high-frequency noise while preserving the main trends and patterns.

## The Test Signal

Our test signal is like a synthetic example that combines:
- **Slow trend** (low frequency) - like the overall music melody
- **Medium patterns** (medium frequency) - like the rhythm section
- **Fast oscillations** (high frequency) - like detailed musical notes
- **Random noise** - like radio static

## Graph Explanations

### 1. Original Test Signal
**What it shows:** The raw data with all frequency components mixed together.
**What to look for:** You can see smooth curves mixed with jagged noise.
**Why it matters:** This is what your data looks like before cleaning.

### 2. FITS Filter Results

#### Time Domain Comparison
**What it shows:** Original signal (blue) vs FITS filtered signal (red).
**What FITS does:** Uses a mathematical formula to decide which frequencies to keep based on your data's natural cycles (like daily patterns).
**What to look for:** The red line is smoother but still follows the main patterns of the blue line.
**Key insight:** FITS is systematic - it always cuts at the same frequency level.

#### Frequency Domain Comparison  
**What it shows:** How much of each frequency is present in the original vs filtered signal.
**The vertical line:** Shows where FITS decided to cut off frequencies.
**What to look for:** Everything to the right of the line gets removed (set to zero).
**Why it matters:** Shows exactly which parts of your data got filtered out.

### 3. Power Spectrum Filter Results

#### Different Energy Thresholds (90%, 80%)
**What it shows:** How the filter behaves when you ask it to keep different amounts of the signal's "energy."
**90% threshold:** Keeps frequencies that contain 90% of the signal's total energy.
**80% threshold:** More aggressive - only keeps frequencies with 80% of energy, removes more noise.
**What to look for:** Lower thresholds = smoother results but potentially more information loss.

#### Frequency Domain Views
**What it shows:** Which frequencies got kept vs removed for each threshold.
**Key insight:** This filter is smart - it automatically finds the most important frequencies instead of just cutting at a fixed point.

### 4. Side-by-Side Comparison

#### Time Domain Comparison
**What it shows:** All three signals on one plot so you can compare directly.
**Original (blue):** Raw noisy data
**FITS (red):** Systematic filtering based on cycles
**Power (green):** Smart filtering based on energy content
**What to look for:** Which filter preserves the patterns you care about best.

#### Frequency Domain (Linear and Log Scale)
**Linear scale:** Shows the actual magnitude of each frequency.
**Log scale:** Makes it easier to see small frequency components.
**Transfer functions:** Shows exactly how much each frequency gets reduced (0 = completely removed, 1 = fully preserved).

#### Difference Plots
**What it shows:** What each filter removed from the original signal.
**Why it matters:** Helps you understand what information you're losing with each filter.

#### Statistics Comparison
**What it shows:** Numbers comparing mean, standard deviation, and range for each filter.
**Why it matters:** Quantifies how much each filter changes your data.

### 5. Frequency Domain Analysis Deep Dive

#### Magnitude and Power Spectral Density
**Magnitude:** How strong each frequency is.
**Power Spectral Density (PSD):** Shows how the signal's power is distributed across different frequencies (normalized).
**What to look for:** FITS creates sharp cutoffs, while Power filter preserves the strongest frequencies regardless of their position. PSD makes it easier to see which frequencies contain the most energy.

#### Power Spectrum (Log Scale)
**What it shows:** The energy content at each frequency on a logarithmic scale.
**Why log scale:** Makes it easier to see both strong and weak frequency components.

#### Filter Transfer Functions
**What it shows:** For each frequency, how much gets through the filter (0-100%).
**FITS transfer:** Sharp cutoff - either 100% or 0%.
**Power transfer:** Gradual - keeps important frequencies, removes unimportant ones.

#### Top 10 Frequency Components Table
**What it shows:** The strongest frequency components in your data and whether each filter keeps them.
**✓ = kept, ✗ = removed**
**Why it matters:** Shows which important patterns each filter preserves.

### 6. Different Signal Types

#### Seasonal Pattern
**What it is:** Data with regular daily and weekly cycles (like website traffic).
**Filter comparison:** Shows how each filter handles predictable patterns.

#### Trending Data
**What it is:** Data with a long-term upward/downward trend plus noise.
**Filter comparison:** Shows how filters preserve trends while removing noise.

#### Noisy Signal
**What it is:** A clean pattern buried in heavy noise (like sensor readings).
**Filter comparison:** Shows which filter best recovers the hidden pattern.

#### Effectiveness Analysis
**Standard Deviation:** Measures how much the signal varies (lower = smoother).
**Smoothness metric:** Measures how jagged the signal is (lower = less jagged).
**Percentage changes:** Shows how much each filter reduces noise.

## Key Takeaways

### When to Use FITS Filter:
- When you want **consistent, predictable** filtering across different datasets
- When your data has **known cycles** (hourly, daily, weekly patterns)
- When you need **reproducible results** for scientific analysis
- Formula: Cutoff = (data_length ÷ cycle_length + 1) × 2 + 10

### When to Use Power Spectrum Filter:
- When your data characteristics **vary significantly** between datasets
- When you want to **automatically adapt** to signal content
- When you want to **preserve the most important information** regardless of frequency
- Parameter: Energy threshold (90% = keep frequencies containing 90% of total energy)

### When to Use Enhanced Frequency Dropout:
- When you want **randomized filtering** that preserves dominant patterns
- For **data augmentation** during machine learning training
- When you want **some randomness** but still keep important frequencies

### When to Use Butterworth Filter:
- When you need **traditional signal processing** with smooth frequency response
- When you have **specific cutoff frequency requirements**
- When working with **well-understood frequency ranges**

### When to Use Simple Low-Pass Filter:
- When you want **basic noise removal** with simple percentage-based cutoff
- For **quick prototyping** and initial data exploration
- When you want **predictable filtering** with minimal parameters

## Practical Usage

In your lag-llama project, these filters help:
1. **Reduce noise** in time series data before training
2. **Preserve important patterns** that the model needs to learn
3. **Improve model performance** by providing cleaner input data
4. **Handle different data frequencies** automatically (hourly, daily, etc.)

The automatic base_period inference means the filters adapt to your data's natural cycles without manual configuration!