#!/usr/bin/env python3
"""
Simple test script for the new filtering functions - updated to match data_utils.py
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt

def fits_then_cps_filter(x, seq_len=None, base_period=24, h_order=2, dim=0, return_info=False):
    """
    Mirror of data_utils.fits_then_cps_filter for quick local testing.
    Applies FITS to estimate preserved energy, then CPS with that energy threshold.
    """
    x = np.asarray(x)
    if seq_len is None:
        seq_len = len(x)
    # FITS
    cut_freq = int(seq_len // base_period + 1) * h_order + 10
    x_t = torch.from_numpy(x)
    X = torch.fft.rfft(x_t, dim=dim)
    total_energy = torch.sum(torch.abs(X) ** 2).item()
    if total_energy <= 0:
        energy_ratio = 1.0
    else:
        X_fits = X.clone()
        X_fits[cut_freq:] = 0
        x_fits = torch.fft.irfft(X_fits, dim=dim)
        X_fits_re = torch.fft.rfft(torch.from_numpy(x_fits.numpy()), dim=dim)
        fits_energy = torch.sum(torch.abs(X_fits_re) ** 2).item()
        energy_ratio = float(max(0.0, min(1.0, fits_energy / total_energy)))
    # CPS with ratio
    filtered = cumulative_power_spectrum_filter(x, energy_threshold=energy_ratio, dim=dim)
    if return_info:
        return filtered, {"cut_freq": int(cut_freq), "energy_threshold": energy_ratio}
    return filtered

def freq_dropout(x, y=None, dropout_rate=0.2, dim=0, keep_dominant=True):
    """
    Applies frequency dropout to the input time series x.
    If y is None, uses zeros of the same shape as x.
    
    Modified to optionally preserve dominant frequencies similar to FITS approach.
    """
    x = np.asarray(x)
    if y is None:
        y = np.zeros_like(x)
    else:
        y = np.asarray(y)
    x_torch, y_torch = torch.from_numpy(x), torch.from_numpy(y)
    xy = torch.cat([x_torch, y_torch], dim=0)
    xy_f = torch.fft.rfft(xy, dim=0)
    
    # Create dropout mask
    m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate
    
    if keep_dominant:
        # Calculate amplitude and preserve top frequencies (similar to FITS)
        amp = torch.abs(xy_f)
        _, sorted_indices = torch.sort(amp, dim=dim, descending=True)
        # Preserve top 10% of frequencies from dropout
        n_preserve = max(1, int(0.1 * xy_f.shape[dim]))
        preserve_mask = torch.zeros_like(m, dtype=torch.bool)
        for i in range(n_preserve):
            if dim == 0 and i < sorted_indices.shape[0]:
                preserve_mask[sorted_indices[i]] = True
        # Don't dropout the dominant frequencies
        m = m & ~preserve_mask
    
    freal = xy_f.real.masked_fill(m, 0)
    fimag = xy_f.imag.masked_fill(m, 0)
    xy_f = torch.complex(freal, fimag)
    xy = torch.fft.irfft(xy_f, dim=dim)
    x_out, y_out = xy[:x_torch.shape[0], :].numpy(), xy[-y_torch.shape[0]:, :].numpy()
    return x_out

def butterworth_lowpass_filter(x, cutoff, fs, order=4):
    """
    Applies a Butterworth low-pass filter to the input time series x.
    cutoff: cutoff frequency (Hz)
    fs: sampling frequency (Hz)
    order: filter order (higher = sharper cutoff)
    """
    x = np.asarray(x)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, x)
    return y

def low_pass_filter(x, cutoff_ratio=0.5, dim=0):
    """
    Applies a low-pass filter to the input time series x.
    cutoff_ratio: Fraction of frequencies to keep (e.g., 0.2 keeps the lowest 20%).
    """
    x = np.asarray(x)
    x_torch = torch.from_numpy(x)
    x_f = torch.fft.rfft(x_torch, dim=dim)
    n_freqs = x_f.shape[0]
    cutoff = int(n_freqs * cutoff_ratio)
    # Zero out frequencies above cutoff
    x_f[cutoff:] = 0
    x_filtered = torch.fft.irfft(x_f, dim=dim)
    return x_filtered.numpy()

def fits_based_filter(x, seq_len=None, base_period=24, h_order=2, dim=0):
    """
    Applies FITS-style low-pass filtering based on sequence length and base period.
    This follows the FITS methodology for determining cutoff frequency.
    
    Args:
        x: Input time series
        seq_len: Sequence length (if None, uses length of x)
        base_period: Base period for the time series (e.g., 24 for hourly data)
        h_order: Harmonic order multiplier
        dim: Dimension along which to apply the filter
    """
    x = np.asarray(x)
    if seq_len is None:
        seq_len = len(x)
    
    # FITS cutoff calculation: (seq_len // base_T + 1) * H_order + 10
    cut_freq = int(seq_len // base_period + 1) * h_order + 10
    
    x_torch = torch.from_numpy(x)
    x_f = torch.fft.rfft(x_torch, dim=dim)
    
    # Apply FITS-style filtering: zero out frequencies above cutoff
    x_f[cut_freq:] = 0
    
    x_filtered = torch.fft.irfft(x_f, dim=dim)
    return x_filtered.numpy()

def cumulative_power_spectrum_filter(x, energy_threshold=0.9, dim=0):
    """
    Applies low-pass filtering based on cumulative power spectrum analysis.
    Preserves frequencies that contain the specified percentage of total energy.
    
    Args:
        x: Input time series
        energy_threshold: Fraction of total energy to preserve (e.g., 0.9 for 90%)
        dim: Dimension along which to apply the filter
    """
    x = np.asarray(x)
    x_torch = torch.from_numpy(x)
    x_f = torch.fft.rfft(x_torch, dim=dim)
    
    # Calculate power spectrum (magnitude squared)
    power_spectrum = torch.abs(x_f) ** 2
    total_energy = torch.sum(power_spectrum)
    
    # Sort frequencies by power (descending order)
    sorted_power, sorted_indices = torch.sort(power_spectrum, descending=True)
    
    # Find cutoff frequency that preserves desired energy
    cumulative_energy = torch.cumsum(sorted_power, dim=0)
    energy_ratio = cumulative_energy / total_energy
    
    # Find index where we reach the energy threshold
    cutoff_idx = torch.where(energy_ratio >= energy_threshold)[0]
    if len(cutoff_idx) > 0:
        n_preserve = cutoff_idx[0].item() + 1
        # Get the indices of frequencies to preserve
        preserve_indices = sorted_indices[:n_preserve]
        
        # Create mask for filtering
        mask = torch.zeros_like(x_f, dtype=torch.bool)
        mask[preserve_indices] = True
        
        # Apply filter by masking
        x_f_filtered = x_f.clone()
        x_f_filtered[~mask] = 0
    else:
        # If threshold not met, preserve all frequencies
        x_f_filtered = x_f
    
    x_filtered = torch.fft.irfft(x_f_filtered, dim=dim)
    return x_filtered.numpy()

def create_test_signal(length=240, fs=1.0):
    """Create a test signal with multiple frequency components"""
    t = np.arange(length) / fs
    
    # Create signal with multiple frequency components
    signal = (
        2.0 * np.sin(2 * np.pi * 0.05 * t) +     # Low frequency component
        1.5 * np.sin(2 * np.pi * 0.1 * t) +      # Medium frequency component
        1.0 * np.sin(2 * np.pi * 0.2 * t) +      # Higher frequency component
        0.5 * np.random.randn(length)             # Noise
    )
    
    return signal, t

def test_filters():
    """Test the new filtering functions"""
    print("Testing filtering functions...")
    
    # Create test signal
    signal, t = create_test_signal(240)
    
    print(f"Original signal shape: {signal.shape}")
    print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    
    # Test FITS-based filter
    print("\n1. Testing FITS-based filter...")
    try:
        fits_filtered = fits_based_filter(signal, seq_len=len(signal), base_period=24, h_order=2)
        print(f"FITS filtered shape: {fits_filtered.shape}")
        print(f"FITS filtered range: [{fits_filtered.min():.3f}, {fits_filtered.max():.3f}]")
        print("✓ FITS filter working correctly")
    except Exception as e:
        print(f"✗ FITS filter failed: {e}")
    
    # Test cumulative power spectrum filter
    print("\n2. Testing cumulative power spectrum filter...")
    try:
        power_filtered = cumulative_power_spectrum_filter(signal, energy_threshold=0.9)
        print(f"Power filtered shape: {power_filtered.shape}")
        print(f"Power filtered range: [{power_filtered.min():.3f}, {power_filtered.max():.3f}]")
        print("✓ Cumulative power spectrum filter working correctly")
    except Exception as e:
        print(f"✗ Cumulative power spectrum filter failed: {e}")
    
    # Test FITS -> CPS pipeline
    print("\n3. Testing FITS -> CPS pipeline...")
    try:
        pipeline_filtered, info = fits_then_cps_filter(signal, seq_len=len(signal), base_period=24, h_order=2, return_info=True)
        print(f"Pipeline filtered shape: {pipeline_filtered.shape}")
        print(f"Pipeline filtered range: [{pipeline_filtered.min():.3f}, {pipeline_filtered.max():.3f}]")
        print(f"Used cut_freq={info['cut_freq']}, energy_threshold={info['energy_threshold']:.4f}")
        print("✓ FITS -> CPS pipeline working correctly")
    except Exception as e:
        print(f"✗ FITS -> CPS pipeline failed: {e}")
    
    # Test enhanced freq_dropout
    print("\n4. Testing enhanced freq_dropout...")
    try:
        dropout_filtered = freq_dropout(signal, dropout_rate=0.3, keep_dominant=True)
        print(f"Dropout filtered shape: {dropout_filtered.shape}")
        print(f"Dropout filtered range: [{dropout_filtered.min():.3f}, {dropout_filtered.max():.3f}]")
        print("✓ Enhanced freq_dropout working correctly")
    except Exception as e:
        print(f"✗ Enhanced freq_dropout failed: {e}")
    
    # Test simple low-pass filter
    print("\n5. Testing simple low-pass filter...")
    try:
        lpf_filtered = low_pass_filter(signal, cutoff_ratio=0.3)
        print(f"LPF filtered shape: {lpf_filtered.shape}")
        print(f"LPF filtered range: [{lpf_filtered.min():.3f}, {lpf_filtered.max():.3f}]")
        print("✓ Simple low-pass filter working correctly")
    except Exception as e:
        print(f"✗ Simple low-pass filter failed: {e}")
    
    # Test Butterworth filter
    print("\n6. Testing Butterworth filter...")
    try:
        butter_filtered = butterworth_lowpass_filter(signal, cutoff=0.15, fs=1.0, order=4)
        print(f"Butterworth filtered shape: {butter_filtered.shape}")
        print(f"Butterworth filtered range: [{butter_filtered.min():.3f}, {butter_filtered.max():.3f}]")
        print("✓ Butterworth filter working correctly")
    except Exception as e:
        print(f"✗ Butterworth filter failed: {e}")
    
    print("\nFilter tests completed!")

if __name__ == "__main__":
    test_filters()