"""
FilterProcessor: Abstraction for time series filtering methods.
Separates filtering logic from data loading to maintain clean architecture.
"""

import torch
import numpy as np
from scipy.signal import butter, filtfilt
from pandas.tseries.frequencies import to_offset
from typing import Dict, Any, Optional, Tuple, Union


class FilterProcessor:
    """
    Encapsulates all time series filtering methods with a clean interface.
    Supports multiple filtering strategies: LPF, Butterworth, FITS, CPS, and FITS+CPS pipeline.
    """
    
    def __init__(
        self,
        method: str = "none",
        dropout_rate: float = 0.2,
        butter_cutoff: float = 0.1,
        butter_fs: float = 1.0,
        butter_order: int = 4,
        base_period: Optional[int] = None,
        h_order: int = 2,
        energy_threshold: float = 0.9,
        verbose: bool = True,
    ):
        """
        Initialize FilterProcessor with specified filtering method and parameters.
        
        Args:
            method: Filtering method ("none", "lpf", "butterworth", "fits", "cps", "fits_then_cps")
            dropout_rate: Cutoff ratio for LPF filter
            butter_cutoff: Cutoff frequency for Butterworth filter
            butter_fs: Sampling frequency for Butterworth filter
            butter_order: Order of Butterworth filter
            base_period: Base period for FITS filter (auto-inferred if None)
            h_order: Harmonic order for FITS filter
            energy_threshold: Energy threshold for CPS filter
            verbose: Whether to print filtering information
        """
        self.method = method.lower()
        self.dropout_rate = dropout_rate
        self.butter_cutoff = butter_cutoff
        self.butter_fs = butter_fs
        self.butter_order = butter_order
        self.base_period = base_period
        self.h_order = h_order
        self.energy_threshold = energy_threshold
        self.verbose = verbose
        
        # Validate method
        valid_methods = {"none", "lpf", "butterworth", "fits", "cps", "fits_then_cps"}
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
    
    def infer_base_period_from_frequency(self, freq_str: str) -> int:
        """
        Auto-infer base_period from frequency string using pandas offset parsing.
        
        Args:
            freq_str: Frequency string (e.g., '1H', '15min', 'D', 'M')
            
        Returns:
            Appropriate base_period value for the frequency
        """
        try:
            freq_offset = to_offset(freq_str)
            freq_type = type(freq_offset).__name__
            
            # For minute-based frequencies
            if hasattr(freq_offset, 'n') and 'Minute' in freq_type:
                minutes = freq_offset.n
                return 24 * 60 // minutes  # Intervals per day
            
            # For hour-based frequencies  
            elif hasattr(freq_offset, 'n') and 'Hour' in freq_type:
                hours = freq_offset.n
                return 24 // hours  # Intervals per day
            
            # For day-based frequencies
            elif 'Day' in freq_type or freq_type == 'Day':
                return 7  # 7 days per week
            
            # For week-based frequencies
            elif 'Week' in freq_type:
                return 52  # 52 weeks per year
            
            # For month-based frequencies
            elif 'Month' in freq_type:
                return 12  # 12 months per year
            
            # For quarter-based frequencies
            elif 'Quarter' in freq_type:
                return 4  # 4 quarters per year
            
            # For year-based frequencies
            elif 'Year' in freq_type:
                return 1  # 1 year cycle
            
            # Default fallback
            else:
                if self.verbose:
                    print(f"Unknown frequency type '{freq_type}' for '{freq_str}', using base_period=24")
                return 24
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not parse frequency '{freq_str}': {e}")
                print("Using default base_period=24 (hourly pattern)")
            return 24  # Default to hourly patterns
    
    def process(
        self, 
        target: Union[list, np.ndarray], 
        freq: Optional[str] = None,
        context: str = "unknown"
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Apply the configured filtering method to the target time series.
        
        Args:
            target: Input time series data
            freq: Frequency string for auto-inferring base_period
            context: Context string for logging (e.g., "train", "val")
            
        Returns:
            Filtered time series (numpy array) or tuple (filtered, info_dict) for some methods
        """
        if self.method == "none":
            return np.asarray(target)
        
        # Convert to numpy array
        target_np = np.asarray(target)
        
        # Auto-infer base_period if needed and not provided
        if self.base_period is None and freq is not None and self.method in ["fits", "fits_then_cps"]:
            inferred_period = self.infer_base_period_from_frequency(freq)
            if self.verbose:
                print(f"Auto-inferred base_period={inferred_period} from frequency '{freq}' for {context}")
            # Use inferred period for this call only (don't modify self.base_period)
            effective_base_period = inferred_period
        else:
            effective_base_period = self.base_period or 24  # Default fallback
        
        # Apply the specified filtering method
        if self.method == "lpf":
            return self._low_pass_filter(target_np)
            
        elif self.method == "butterworth":
            return self._butterworth_filter(target_np)
            
        elif self.method == "fits":
            return self._fits_filter(target_np, effective_base_period)
            
        elif self.method == "cps":
            return self._cps_filter(target_np)
            
        elif self.method == "fits_then_cps":
            filtered, info = self._fits_then_cps_filter(target_np, effective_base_period)
            if self.verbose:
                print(f"[{self.method}][{context}] cut_freq={info['cut_freq']}, energy_threshold={info['energy_threshold']:.4f}")
            return filtered
            
        else:
            # Fallback (should not reach here due to validation in __init__)
            return target_np
    
    def _low_pass_filter(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        """Apply simple low-pass filter."""
        x_torch = torch.from_numpy(x)
        x_f = torch.fft.rfft(x_torch, dim=dim)
        n_freqs = x_f.shape[0]
        cutoff = int(n_freqs * self.dropout_rate)
        x_f[cutoff:] = 0
        x_filtered = torch.fft.irfft(x_f, dim=dim)
        return x_filtered.numpy()
    
    def _butterworth_filter(self, x: np.ndarray) -> np.ndarray:
        """Apply Butterworth low-pass filter."""
        nyq = 0.5 * self.butter_fs
        normal_cutoff = self.butter_cutoff / nyq
        b, a = butter(self.butter_order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, x)
    
    def _fits_filter(self, x: np.ndarray, base_period: int, dim: int = 0) -> np.ndarray:
        """Apply FITS-style filtering."""
        seq_len = len(x)
        cut_freq = int(seq_len // base_period + 1) * self.h_order + 10
        
        x_torch = torch.from_numpy(x)
        x_f = torch.fft.rfft(x_torch, dim=dim)
        x_f[cut_freq:] = 0
        x_filtered = torch.fft.irfft(x_f, dim=dim)
        return x_filtered.numpy()
    
    def _cps_filter(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        """Apply cumulative power spectrum filter."""
        x_torch = torch.from_numpy(x)
        x_f = torch.fft.rfft(x_torch, dim=dim)
        
        # Calculate power spectrum
        power_spectrum = torch.abs(x_f) ** 2
        total_energy = torch.sum(power_spectrum)
        
        # Sort by power and find cutoff
        sorted_power, sorted_indices = torch.sort(power_spectrum, descending=True)
        cumulative_energy = torch.cumsum(sorted_power, dim=0)
        energy_ratio = cumulative_energy / total_energy
        
        # Find threshold crossing
        cutoff_idx = torch.where(energy_ratio >= self.energy_threshold)[0]
        if len(cutoff_idx) > 0:
            n_preserve = cutoff_idx[0].item() + 1
            preserve_indices = sorted_indices[:n_preserve]
            
            # Create and apply mask
            mask = torch.zeros_like(x_f, dtype=torch.bool)
            mask[preserve_indices] = True
            x_f_filtered = x_f.clone()
            x_f_filtered[~mask] = 0
        else:
            x_f_filtered = x_f
        
        x_filtered = torch.fft.irfft(x_f_filtered, dim=dim)
        return x_filtered.numpy()
    
    def _fits_then_cps_filter(
        self, 
        x: np.ndarray, 
        base_period: int, 
        dim: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply FITS → CPS pipeline filter."""
        seq_len = len(x)
        cut_freq = int(seq_len // base_period + 1) * self.h_order + 10
        
        x_t = torch.from_numpy(x)
        X = torch.fft.rfft(x_t, dim=dim)
        total_energy = torch.sum(torch.abs(X) ** 2).item()
        
        if total_energy <= 0:
            energy_ratio = 1.0
        else:
            # Apply FITS filter
            X_fits = X.clone()
            X_fits[cut_freq:] = 0
            x_fits = torch.fft.irfft(X_fits, dim=dim)
            
            # Compute energy ratio
            X_fits_re = torch.fft.rfft(x_fits, dim=dim)
            fits_energy = torch.sum(torch.abs(X_fits_re) ** 2).item()
            energy_ratio = float(max(0.0, min(1.0, fits_energy / total_energy)))
        
        # Apply CPS with computed threshold
        # Create temporary CPS processor with the computed threshold
        temp_processor = FilterProcessor(
            method="cps", 
            energy_threshold=energy_ratio,
            verbose=False
        )
        filtered = temp_processor._cps_filter(x, dim=dim)
        
        info = {"cut_freq": int(cut_freq), "energy_threshold": energy_ratio}
        return filtered, info


def create_filter_processor_from_args(args) -> FilterProcessor:
    """
    Convenience function to create FilterProcessor from command line arguments.
    
    Args:
        args: Namespace object with filtering arguments
        
    Returns:
        Configured FilterProcessor instance
    """
    # Determine method from boolean flags
    method = "none"
    if getattr(args, 'lpf', False):
        method = "lpf"
    elif getattr(args, 'butterworth', False):
        method = "butterworth" 
    elif getattr(args, 'fits_filter', False):
        method = "fits"
    elif getattr(args, 'cumulative_power_filter', False):
        method = "cps"
    elif getattr(args, 'fits_then_cps', False):
        method = "fits_then_cps"
    
    return FilterProcessor(
        method=method,
        dropout_rate=getattr(args, 'filter_dropout_rate', 0.2),
        butter_cutoff=getattr(args, 'butter_cutoff', 0.1),
        butter_fs=getattr(args, 'butter_fs', 1.0),
        butter_order=getattr(args, 'filter_butter_order', 4),
        base_period=getattr(args, 'filter_base_period', None),
        h_order=getattr(args, 'filter_h_order', 2),
        energy_threshold=getattr(args, 'filter_energy_threshold', 0.9),
        verbose=True,
    )