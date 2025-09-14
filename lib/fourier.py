import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class Fourier:
    def __init__(self, df, sample_rate=1.0):
        self.df = df
        self.values = df.squeeze().to_numpy()
        self._len = len(self.values)
        self.sample_rate = sample_rate
        self.raw_fft = self._calc_transform()

        self.fft = np.abs(self.raw_fft)[: self._len // 2]
        self._threshold = np.median(self.fft) * 2
        self.peaks, _ = find_peaks(self.fft, threshold=self._threshold)

    def _calc_transform(self):
        return np.fft.fft(self.values)

    def _reconstruct(self, val=None):
        if val is None:
            val = self.raw_fft

        inverse = np.fft.ifft(val)
        return inverse.real

    def calc_metrics(self, new_signal):
        loss = self.values - new_signal
        nrmse = np.sqrt(np.mean(loss**2)) / np.std(self.values)
        energy_loss = np.sum(loss**2) / np.sum(self.values**2) * 100

        # SNR calculation
        power_signal = np.mean(new_signal**2)
        power_noise = np.mean(loss**2)
        epsilon = 1e-10
        snr = 10 * np.log10(power_signal / (power_noise + epsilon))

        return {"energy_loss": energy_loss, "nrmse": nrmse, "snr": f"{snr:.2f}dB"}

    def filter_freqs(self, threshold_freq):
        modified_fft = self.raw_fft.copy()
        modified_fft[threshold_freq : (len(modified_fft) - threshold_freq)] = 0
        filtered_signal = self._reconstruct(modified_fft)
        metrics = self.calc_metrics(filtered_signal)
        return (filtered_signal, metrics)

    def plot(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.fft)
        plt.axhline(y=float(self._threshold), color="b", linestyle="--")
        plt.text(
            len(self.fft),
            float(self._threshold) * 1.5,
            "Threshold",
            color="b",
            ha="center",
        )

        for peak in self.peaks:
            plt.axvline(peak, color="r", linestyle="--", alpha=0.7)
        plt.xlabel("Frequencies")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.grid(True)
