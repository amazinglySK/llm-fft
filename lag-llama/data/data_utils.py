# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import random
import warnings
import json
import os
from pathlib import Path


# NOTE: Shashwat's changes
import torch
import numpy as np
from scipy.signal import butter, filtfilt

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import InstanceSampler
from pandas.tseries.frequencies import to_offset

from data.read_new_dataset import (
    get_ett_dataset,
    create_train_dataset_without_last_k_timesteps,
    TrainDatasets,
    MetaData,
)
from data.filter_processor import FilterProcessor


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)


class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return sum([len(ds) for ds in self._datasets])


class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    """End index of the history"""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        if window_size <= 0:
            return np.array([], dtype=int)
        indices = np.random.randint(window_size, size=1)
        return indices + a


def _count_timesteps(
    left: pd.Timestamp, right: pd.Timestamp, delta: pd.DateOffset
) -> int:
    """
    Count how many timesteps there are between left and right, according to the given timesteps delta.
    If the number if not integer, round down.
    """
    # This is due to GluonTS replacing Timestamp by Period for version 0.10.0.
    # Original code was tested on version 0.9.4
    if type(left) == pd.Period:
        left = left.to_timestamp()
    if type(right) == pd.Period:
        right = right.to_timestamp()
    assert (
        right >= left
    ), f"Case where left ({left}) is after right ({right}) is not implemented in _count_timesteps()."
    try:
        return (right - left) // delta
    except TypeError:
        # For MonthEnd offsets, the division does not work, so we count months one by one.
        for i in range(10000):
            if left + (i + 1) * delta > right:
                return i
        else:
            raise RuntimeError(
                f"Too large difference between both timestamps ({left} and {right}) for _count_timesteps()."
            )


from pathlib import Path
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset


def create_train_dataset_last_k_percentage(raw_train_dataset, freq, k=100):
    # Get training data
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        number_of_values = int(len(s_train["target"]) * k / 100)
        train_start_index = len(s_train["target"]) - number_of_values
        s_train["target"] = s_train["target"][train_start_index:]
        train_data.append(s_train)

    train_data = ListDataset(train_data, freq=freq)

    return train_data

def freq_dropout(x, y=None, dropout_rate=0.2, dim=0, keep_dominant=True):
    """
    Applies frequency dropout to the input time series x.
    If y is None, uses zeros of the same shape as x.
    
    Modified to optionally preserve dominant frequencies similar to FITS approach.
    NOTE: This function is preserved for backward compatibility but new code should use FilterProcessor.
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

def create_train_and_val_datasets_with_dates(
    name,
    dataset_path,
    data_id,
    history_length,
    prediction_length=None,
    num_val_windows=None,
    val_start_date=None,
    train_start_date=None,
    freq=None,
    last_k_percentage=None,
    # Legacy filtering parameters (deprecated - use filter_processor instead)
    lpf = False,
    butterworth = False,
    fits_filter = False,
    cumulative_power_filter = False,
    butter_cutoff = 0.1,
    butter_fs = 1.0,
    butter_order = 4,
    dropout_rate = 0.2,
    base_period = None,  # Will be auto-inferred from frequency if None
    h_order = 2,
    energy_threshold = 0.9,
    fits_then_cps: bool = False,
    # New clean interface
    filter_processor: FilterProcessor = None,
):
    """
    Create train and validation datasets with optional time series filtering.
    
    Train Start date is assumed to be the start of the series if not provided.
    Freq is inferred from the data if not given.
    We can use ListDataset to just group multiple time series.
    
    Filtering:
        Use either the new filter_processor parameter (recommended) or the legacy boolean flags.
        
        New approach:
        - filter_processor: FilterProcessor instance with configured filtering method
        
        Legacy approach (deprecated):
        - lpf/butterworth/fits_filter/cumulative_power_filter/fits_then_cps: boolean flags
        - Various filter parameters (dropout_rate, butter_*, base_period, etc.)
    """
    
    # Create FilterProcessor from legacy parameters if no processor provided
    if filter_processor is None:
        # Determine method from legacy boolean flags
        if fits_then_cps:
            method = "fits_then_cps"
        elif fits_filter:
            method = "fits"
        elif cumulative_power_filter:
            method = "cps"
        elif butterworth:
            method = "butterworth"
        elif lpf:
            method = "lpf"
        else:
            method = "none"
        
        # Create processor with legacy parameters
        filter_processor = FilterProcessor(
            method=method,
            dropout_rate=dropout_rate,
            butter_cutoff=butter_cutoff,
            butter_fs=butter_fs,
            butter_order=butter_order,
            base_period=base_period,
            h_order=h_order,
            energy_threshold=energy_threshold,
            verbose=True,
        )
    
    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = os.path.join(dataset_path, "ett_datasets")
        raw_dataset = get_ett_dataset(name, path)
    elif name in (
        "cpu_limit_minute",
        "cpu_usage_minute",
        "function_delay_minute",
        "instances_minute",
        "memory_limit_minute",
        "memory_usage_minute",
        "platform_delay_minute",
        "requests_minute",
    ):
        path = os.path.join(dataset_path, "huawei/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) != str]
        test_data = [x for x in data["test"] if type(x["target"][0]) != str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        raw_dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = os.path.join(dataset_path, "air_quality/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(
            full_dataset, freq=metadata.freq, k=24
        )
        raw_dataset = TrainDatasets(
            metadata=metadata, train=train_ds, test=full_dataset
        )
    else:
        raw_dataset = get_dataset(name, path=Path(dataset_path))

    if prediction_length is None:
        prediction_length = raw_dataset.metadata.prediction_length
    if freq is None:
        freq = raw_dataset.metadata.freq
    timestep_delta = pd.tseries.frequencies.to_offset(freq)
    
    # The FilterProcessor will handle base_period inference internally
    
    raw_train_dataset = raw_dataset.train

    if not num_val_windows and not val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")
    if num_val_windows and val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")

    max_train_end_date = None

    # Get training data
    total_train_points = 0
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"] if not train_start_date else train_start_date,
                val_start_date,
                timestep_delta,
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        # Compute train_start_index based on last_k_percentage
        if last_k_percentage:
            number_of_values = int(len(s_train["target"]) * last_k_percentage / 100)
            train_start_index = train_end_index - number_of_values
        else:
            train_start_index = 0

        # Extract target data (no filtering applied here - will be done per batch)
        target = series["target"][train_start_index:train_end_index]
        
        s_train["target"] = target 
        s_train["item_id"] = i
        s_train["data_id"] = data_id
        train_data.append(s_train)
        total_train_points += len(s_train["target"])

        # Calculate the end date
        end_date = s_train["start"] + to_offset(freq) * (len(s_train["target"]) - 1)
        if max_train_end_date is None or end_date > max_train_end_date:
            max_train_end_date = end_date

    train_data = ListDataset(train_data, freq=freq)

    # Get validation data
    total_val_points = 0
    total_val_windows = 0
    val_data = []
    for i, series in enumerate(raw_train_dataset):
        s_val = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"], val_start_date, timestep_delta
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        val_start_index = train_end_index - prediction_length - history_length
        target = series["target"][val_start_index:]
        
        s_val["target"] = target 
        s_val["item_id"] = i
        s_val["data_id"] = data_id
        val_data.append(s_val)
        total_val_points += len(s_val["target"])
        total_val_windows += len(s_val["target"]) - prediction_length - history_length
    val_data = ListDataset(val_data, freq=freq)

    total_points = (
        total_train_points
        + total_val_points
        - (len(raw_train_dataset) * (prediction_length + history_length))
    )

    return (
        train_data,
        val_data,
        total_train_points,
        total_val_points,
        total_val_windows,
        max_train_end_date,
        total_points,
        freq,
    )


def create_test_dataset(name, dataset_path, history_length, freq=None, data_id=None):
    """
    For now, only window per series is used.
    make_evaluation_predictions automatically only predicts for the last "prediction_length" timesteps
    NOTE / TODO: For datasets where the test set has more series (possibly due to more timestamps), \
    we should check if we only use the last N series where N = series per single timestamp, or if we should do something else.
    """

    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = os.path.join(dataset_path, "ett_datasets")
        dataset = get_ett_dataset(name, path)
    elif name in (
        "cpu_limit_minute",
        "cpu_usage_minute",
        "function_delay_minute",
        "instances_minute",
        "memory_limit_minute",
        "memory_usage_minute",
        "platform_delay_minute",
        "requests_minute",
    ):
        path = os.path.join(dataset_path, "huawei/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) != str]
        test_data = [x for x in data["test"] if type(x["target"][0]) != str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = os.path.join(dataset_path, "air_quality/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(
            full_dataset, freq=metadata.freq, k=24
        )
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    else:
        dataset = get_dataset(name, path=Path(dataset_path))

    if freq is None:
        freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    data = []
    total_points = 0
    for i, series in enumerate(dataset.test):
        offset = len(series["target"]) - (history_length + prediction_length)
        if offset > 0:
            target = series["target"][-(history_length + prediction_length) :]
            data.append(
                {
                    "target": target,
                    "start": series["start"] + offset,
                    "item_id": i,
                    "data_id": data_id,
                }
            )
        else:
            series_copy = copy.deepcopy(series)
            series_copy["item_id"] = i
            series_copy["data_id"] = data_id
            data.append(series_copy)
        total_points += len(data[-1]["target"])
    return ListDataset(data, freq=freq), prediction_length, total_points, freq
