from .fourier import Fourier
import matplotlib.pyplot as plt
import pandas as pd


class TimeSeries:
    def __init__(self, series, ylabel, series_name):
        self.series = series
        self.shape = self.series.shape
        self.fft = Fourier(self.series)
        self.name = series_name
        self.x = "Time"
        self.y = ylabel

    def plot(self, df=None):
        plt.figure(figsize=(15, 5))
        plt.plot(self.series.index, self.series.values, label="original")
        if df is not None:
            plt.plot(df.index, df.values, label="filtered", alpha=0.7)
        plt.title(self.name)
        plt.xlabel(self.x)
        plt.ylabel(self.y)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

    def filter(self, threshold):
        filtered_signal, metrics = self.fft.filter_freqs(threshold)
        df = pd.DataFrame({"values": filtered_signal}, index=self.series.index)
        return df, metrics

    def head(self):
        self.series.head()

    def __str__(self):
        return str(self.series)
