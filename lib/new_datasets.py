from datetime import timedelta
import pandas as pd
from typing import List, Union
from .data_loader import convert_tsf_to_dataframe
from .TimeSeries import TimeSeries


class Dataset:
    """
    Base class for datasets.
    Modified to support lists of TimeSeries objects for train and test,
    allowing for handling multiple time series within a single dataset.
    """

    def __init__(
        self,
        train: Union[TimeSeries, List[TimeSeries]],
        test: Union[TimeSeries, List[TimeSeries]],
        title: str,
    ):
        # Ensure train and test are always lists for consistent handling
        self.train = [train] if isinstance(train, TimeSeries) else train
        self.test = [test] if isinstance(test, TimeSeries) else test
        self.title = title

    def _process_df(self, df):
        """
        Placeholder for DataFrame processing that might be needed before splitting.
        This method should be overridden by subclasses if specific preprocessing is required.
        """
        return pd.DataFrame(df)

    def __str__(self):
        """
        Provides a string representation of the dataset, showing details for all
        train and test time series within the dataset.
        """
        st = f"Dataset Title: {self.title}\n\n"

        st += "--- TRAIN SERIES ---\n"
        if not self.train:
            st += "No training series available.\n\n"
        else:
            for i, ts in enumerate(self.train):
                st += f"Series {i+1}: {ts.name}\n"
                st += f"{ts.head()}\n"
                st += f"Shape: {ts.shape()}\n\n"

        st += "--- TEST SERIES ---\n"
        if not self.test:
            st += "No test series available.\n\n"
        else:
            for i, ts in enumerate(self.test):
                st += f"Series {i+1}: {ts.name}\n"
                st += f"{ts.head()}\n"
                st += f"Shape: {ts.shape()}\n\n"
        return st


class AirQuality(Dataset):
    """
    AirQuality dataset, loaded from a CSV file.
    This class handles a single time series as per the original requirement.
    """

    def __init__(self):
        # Using pd.read_csv directly for CSV files as in the original code
        air_quality_csv = pd.read_csv(
            "./datasets/AirQualityUCI.csv",
            delimiter=";",
            decimal=",",
            usecols=lambda col: col != "",  # Remove empty columns
            header=0,
            engine="python",
        )
        # Select relevant rows and columns and create a copy to avoid SettingWithCopyWarning
        df = air_quality_csv.iloc[:9357, :15].copy()

        # Combine Date and Time columns into a single datetime string
        df["datetime_str"] = df["Date"] + " " + df["Time"]
        # Convert to datetime objects
        df["datetime"] = pd.to_datetime(df["datetime_str"], format="%d/%m/%Y %H.%M.%S")

        # Move 'datetime' to the first column
        df.insert(0, "datetime", df.pop("datetime"))
        # Drop the original date and time columns
        df.drop(columns=["datetime_str", "Date", "Time"], inplace=True)

        # Process the DataFrame (handle -200 values and set datetime index)
        # Ensure to pass a copy to _process_df if it modifies the DataFrame in place
        processed_df = self._process_df(df.copy())

        # The original intention was to use 'CO(GT)' for the TimeSeries.
        # Ensure we select only the 'CO(GT)' column as the value column.
        co_column_name = "CO(GT)"
        if co_column_name not in processed_df.columns:
            # Fallback if the column name isn't exactly 'CO(GT)' or is not available
            # This relies on 'CO(GT)' being the first numeric column after datetime
            numeric_cols = processed_df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                co_column_name = numeric_cols[0]
            else:
                raise ValueError(
                    "Could not find a suitable numeric column for CO concentration."
                )

        # Split into train and test TimeSeries objects
        train_split = int(processed_df.shape[0] * 0.8)
        train = TimeSeries(
            processed_df.iloc[
                :train_split, [processed_df.columns.get_loc(co_column_name)]
            ],  # Select only the CO column
            co_column_name,
            "AirQuality (TRAIN)",
        )
        test = TimeSeries(
            processed_df.iloc[
                train_split:, [processed_df.columns.get_loc(co_column_name)]
            ],  # Select only the CO column
            co_column_name,
            "AirQuality (TEST)",
        )

        # Initialize the base Dataset class with the single train/test TimeSeries
        super().__init__(train, test, "True hourly averaged concentration CO in mg/m^3")

    def _process_df(self, df):
        """
        Processes the AirQuality DataFrame by setting the datetime index
        and replacing -200 values with the mean of the valid values.
        """
        out = df.copy()  # Work on a copy of the DataFrame

        # Ensure 'datetime' is the index
        if "datetime" in out.columns:
            out.set_index("datetime", inplace=True)
        else:
            raise ValueError(
                "DataFrame must contain a 'datetime' column to be set as index."
            )

        # Replace -200 values with the mean for all numeric columns
        for col in out.select_dtypes(include="number").columns:
            # Calculate mean only from valid values (not -200)
            valid_values = out.loc[out[col] != -200, col]
            if not valid_values.empty:
                mean_val = valid_values.mean()
                out[col] = out[col].replace(-200, mean_val)
            else:
                # If all values are -200 or no valid values, set to 0 or handle as appropriate
                out[col] = out[col].replace(
                    -200, 0
                )  # Or NaN, depending on desired behavior
        return out


class TSFDataset(Dataset):
    """
    Abstract base class for datasets loaded from .tsf files.
    This class handles the common logic for reading .tsf files,
    iterating through multiple time series within the file, and
    creating separate TimeSeries objects for each, splitting them
    into train and test sets.
    """

    def __init__(
        self,
        file_path: str,
        time_delta_unit: str,
        title: str,
        value_column_name: str = "values",
    ):
        # Load the entire dataset from the .tsf file
        all_df, frequency, forecast_horizon, _, _ = convert_tsf_to_dataframe(
            file_path, value_column_name=value_column_name
        )

        train_series_list: List[TimeSeries] = []
        test_series_list: List[TimeSeries] = []

        # Iterate over each unique series identified by 'series_name'
        unique_series_names = all_df["series_name"].unique()
        for series_name in unique_series_names:
            # Get the row corresponding to the current series
            series_data = all_df.loc[all_df["series_name"] == series_name].iloc[0]
            start_ts = series_data["start_timestamp"]
            raw_values = series_data[value_column_name]

            # Ensure raw_values is a DataFrame for TimeSeries constructor
            if isinstance(raw_values, pd.Series):
                # If 'values' column contains a pandas Series (e.g., from a list of lists)
                series_values_df = pd.DataFrame(
                    {value_column_name: raw_values.tolist()}
                )
            elif isinstance(raw_values, pd.DataFrame):
                # If 'values' column already contains a DataFrame
                series_values_df = raw_values
            else:
                # If 'values' column contains a list or numpy array, convert to DataFrame
                series_values_df = pd.DataFrame({value_column_name: raw_values})

            # Create the datetime index based on the specified time unit
            indices = []
            for i in range(series_values_df.shape[0]):
                if time_delta_unit == "days":
                    indices.append(start_ts + timedelta(days=i))
                elif time_delta_unit == "weeks":
                    indices.append(start_ts + timedelta(weeks=i))
                elif time_delta_unit == "months":
                    indices.append(start_ts + pd.DateOffset(months=i))
                elif time_delta_unit == "hours":
                    indices.append(start_ts + timedelta(hours=i))
                else:
                    raise ValueError(f"Unsupported time_delta_unit: {time_delta_unit}")

            # Convert indices to DatetimeIndex
            datetime_indices = pd.to_datetime(indices)

            # Set the created datetime index on the series DataFrame
            series_values_df.set_index(datetime_indices, inplace=True)

            # Apply specific processing for Pedestrian dataset: hourly to daily averaging
            if "pedestrian_counts_dataset" in file_path and time_delta_unit == "hours":
                # Floor datetime index to day to group by day
                hourly_bins = series_values_df.index.floor("D")
                # Group by day and calculate the mean, then round for daily average
                series_values_df = series_values_df.groupby(hourly_bins).mean().round(3)

            # Determine the split point for train and test sets (80/20 split)
            train_split = int(series_values_df.shape[0] * 0.8)

            # Create TimeSeries objects for train and test splits
            train_ts = TimeSeries(
                series_values_df.iloc[:train_split, :],
                value_column_name,
                series_name,
            )
            test_ts = TimeSeries(
                series_values_df.iloc[train_split:, :],
                value_column_name,
                series_name,
            )

            # Add the created TimeSeries objects to their respective lists
            train_series_list.append(train_ts)
            test_series_list.append(test_ts)

        # Initialize the base Dataset class with the lists of TimeSeries objects
        super().__init__(train_series_list, test_series_list, title)


class USBirths(TSFDataset):
    """
    Dataset for daily number of births in the US, loaded from a .tsf file.
    """

    def __init__(self):
        super().__init__(
            "./datasets/us_births_dataset.tsf",
            "days",  # Data is daily
            "Daily number of births in US from 01/01/1969 to 31/12/1988",
        )


class Bitcoin(TSFDataset):
    """
    Dataset for the daily price of Bitcoin, loaded from a .tsf file.
    """

    def __init__(self):
        super().__init__(
            "./datasets/bitcoin_dataset_without_missing_values.tsf",
            "days",  # Data is daily
            "Daily Price of BTC",
        )


class COVID(TSFDataset):
    """
    Dataset for daily COVID deaths, loaded from a .tsf file.
    """

    def __init__(self):
        super().__init__(
            "./datasets/covid_deaths_dataset.tsf",
            "days",  # Data is daily
            "Daily COVID deaths from 22/01/2020 to 20/08/2020 (in an unknown region)",
        )


class EnergyConsumption(Dataset):
    """
    EnergyConsumption dataset, loaded from a CSV file.
    It processes a single series of appliance energy consumption.
    """

    def __init__(self):
        # Load the CSV file
        df = pd.read_csv(
            "./datasets/energydata_complete.csv", header=0, engine="python"
        )

        # Process the DataFrame
        processed_series = self._process_df(df)

        # Determine the split point for train and test sets
        train_split = int(processed_series.shape[0] * 0.8)

        # Create TimeSeries objects for train and test splits
        train = TimeSeries(
            processed_series.iloc[
                :train_split, :1
            ],  # Selects the first column (Appliances)
            processed_series.columns[
                0
            ],  # Get column name dynamically (e.g., 'Appliances')
            "Energy consumption (TRAIN)",
        )
        test = TimeSeries(
            processed_series.iloc[
                train_split:, :1
            ],  # Selects the first column (Appliances)
            processed_series.columns[0],  # Get column name dynamically
            "Energy consumption (TEST)",
        )

        # Initialize the base Dataset class with the single train/test TimeSeries
        super().__init__(train, test, "Energy consumption of appliances (in kWh)")

    def _process_df(self, df):
        """
        Processes the EnergyConsumption DataFrame by setting the 'date' column as index,
        converting it to datetime, and then resampling to hourly averages.
        """
        df_copy = df.copy()  # Work on a copy
        df_copy.set_index("date", inplace=True)
        df_copy.index = pd.to_datetime(df_copy.index)  # Ensure index is a DatetimeIndex

        # Resample to hourly averages. Assume 'Appliances' is the column of interest.
        # Use .resample('H').mean() for direct resampling to hourly frequency
        # This will group by hour and calculate the mean for each hour.
        # We assume 'Appliances' is the primary column for this dataset.
        numeric_cols = df_copy.select_dtypes(include="number").columns
        if "Appliances" in numeric_cols:
            df_processed = (
                df_copy["Appliances"].resample("H").mean().round(3).to_frame()
            )
        else:
            # Fallback if 'Appliances' column is not found or other numeric columns exist
            # This will resample all numeric columns.
            df_processed = df_copy[numeric_cols].resample("H").mean().round(3)
        return df_processed


class Pedestrian(TSFDataset):
    """
    Dataset for daily average pedestrian count, loaded from a .tsf file.
    Note: Raw data is hourly, but it is processed to daily averages within TSFDataset.
    """

    def __init__(self):
        super().__init__(
            "./datasets/pedestrian_counts_dataset.tsf",
            "hours",  # Raw data is at hourly resolution
            "Daily average pedestrian count caught by Sensor 1 in Melbourne",
        )

    # The _process_df logic (hourly to daily averaging) is now handled
    # directly within the TSFDataset's __init__ method for this specific file.


class Tourism(TSFDataset):
    """
    Dataset for total tourism numbers at country level, loaded from a .tsf file.
    """

    def __init__(self):
        super().__init__(
            "./datasets/tourism/tourism_monthly_dataset.tsf",
            "months",  # Data is monthly
            "Total tourism numbers at country level of aggregation",
        )


class Traffic(TSFDataset):
    """
    Dataset for weekly road occupancy rates, loaded from a .tsf file.
    """

    def __init__(self):
        super().__init__(
            "./datasets/traffic_weekly_dataset.tsf",
            "weeks",  # Data is weekly
            "Weekly road occupancy rates on San Francisco Bay area freeways (2015 - 2016)",
        )


class Web(TSFDataset):
    """
    Dataset for weekly web traffic, loaded from a .tsf file.
    """

    def __init__(self):
        super().__init__(
            "./datasets/kaggle_web_traffic_weekly_dataset.tsf",
            "weeks",  # Data is weekly
            "Weekly web traffic for a set of Wikipedia pages from 2015-07-01 to 2017-09-05",
        )


class AirPassengers(Dataset):
    """
    AirPassengers dataset, loaded from a CSV file.
    This class handles a single time series.
    """

    def __init__(self):
        # Load the CSV file
        df = pd.read_csv(
            "./datasets/AirPassengers.csv",
            header=0,
            engine="python",
        )
        # Convert 'Month' column to datetime and set as index
        df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
        df.set_index("Month", inplace=True)

        # Determine the split point for train and test sets
        train_split = int(df.shape[0] * 0.8)

        # The column for passengers is '#Passengers'.
        passenger_column_name = "#Passengers"

        # Create TimeSeries objects for train and test splits
        train = TimeSeries(
            df.iloc[
                :train_split, [df.columns.get_loc(passenger_column_name)]
            ],  # Selects the '#Passengers' column
            passenger_column_name,
            "AirPassengers (TRAIN)",
        )
        test = TimeSeries(
            df.iloc[
                train_split:, [df.columns.get_loc(passenger_column_name)]
            ],  # Selects the '#Passengers' column
            passenger_column_name,
            "AirPassengers (TEST)",
        )

        # Initialize the base Dataset class with the single train/test TimeSeries
        super().__init__(
            train, test, "Monthly totals of a US airline passengers from 1949 to 1960"
        )
