import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class Fourier:
    def __init__(self, df):
        self.df = df
        self.values = df.squeeze().to_numpy()
        self._len = len(self.values)
        self.fft = np.abs(self._calc_transform())[: self._len // 2]

        self._threshold = np.median(self.fft) * 2
        self.peaks, _ = find_peaks(self.fft, threshold=self._threshold)
        print(self.peaks)

    def _calc_transform(self):
        return np.fft.fft(self.values)

    def plot_fft(self, data_name):
        plt.figure(figsize=(15, 5))
        plt.plot(self.fft)
        plt.title(f"Fourier transform output for {data_name}")
        plt.axhline(y=float(self._threshold), color="b", linestyle="--")
        plt.text(
            len(self.fft),
            float(self._threshold) + 10000,
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
