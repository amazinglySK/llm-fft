from datetime import timedelta
import pandas as pd
from .TimeSeries import TimeSeries
from .data_loader import convert_tsf_to_dataframe


class Dataset:

    def __init__(self, train: TimeSeries, test: TimeSeries, title: str):
        self.train = train
        self.test = test
        self.title = title

    def _process_df(self, df):
        return pd.DataFrame(df)

    def __str__(self):
        st = f"""TRAIN
        {self.train.head()}
        {self.train.shape}
        TEST
        {self.test.head()}
        {self.test.shape}
        """
        return st


class AirQuality(Dataset):

    def __init__(self):
        air_quality_csv = pd.read_csv(
            "./datasets/AirQualityUCI.csv",
            delimiter=";",
            decimal=",",
            usecols=lambda col: col != "",
            header=0,
            engine="python",
        )
        df = air_quality_csv.iloc[:9357, :15]
        df["datetime_str"] = df["Date"] + " " + df["Time"]
        df["datetime"] = pd.to_datetime(df["datetime_str"], format="%d/%m/%Y %H.%M.%S")
        df.insert(0, "datetime", df.pop("datetime"))
        df.drop(columns=["datetime_str", "Date", "Time"], inplace=True)

        # Just test with CO output
        train = TimeSeries(
            self._process_df(df.iloc[:7400, :2]),
            "CO conc. (mg/m^3)",
            "AirQuality (TRAIN)",
        )
        test = TimeSeries(
            self._process_df(df.iloc[7400:, :2]),
            "CO conc. (mg/m^3)",
            "AirQuality (TEST)",
        )
        super().__init__(train, test, "True hourly averaged concentration CO in mg/m^3")

    def _process_df(self, df):
        out = df
        out.set_index("datetime", inplace=True)
        for col in out.select_dtypes(include="number").columns:
            mean_val = out.loc[out[col] != -200, col].mean()
            out[col] = out[col].replace(-200, mean_val)
        return out


class USBirths(Dataset):

    def __init__(self):
        df, frequency, forecast_horizon, _, _ = convert_tsf_to_dataframe(
            "./datasets/us_births_dataset.tsf", value_column_name="values"
        )
        series = self._process_df(df)
        train_split = int(series.shape[0] * 0.8)
        train = TimeSeries(
            series.iloc[:train_split, :], "No. of births", "USBirths (TRAIN)"
        )
        test = TimeSeries(
            series.iloc[train_split:, :], "No. of births", "USBirths (TEST)"
        )
        super().__init__(
            train,
            test,
            "Daily number of births in US from 01/01/1969 to 31/12/1988",
        )

    def _process_df(self, df):
        df = df.loc[df["series_name"] == "T1"]
        start_ts = df["start_timestamp"].item()
        series = df["values"].item()
        series = pd.DataFrame({"values": series})
        indices = pd.to_datetime(
            [start_ts + timedelta(days=i) for i in range(series.shape[0])]
        )
        series.set_index(
            indices,
            inplace=True,
        )

        return series


class Bitcoin(Dataset):

    def __init__(self):
        (
            df,
            frequency,
            forecast_horizon,
            _,
            _,
        ) = convert_tsf_to_dataframe(
            "./datasets/bitcoin_dataset_without_missing_values.tsf",
            value_column_name="values",
        )

        series = self._process_df(df)
        train_split = int(series.shape[0] * 0.8)
        train = TimeSeries(
            series.iloc[:train_split, :], "Price of BTC", "Bitcoin (TRAIN)"
        )
        test = TimeSeries(
            series.iloc[train_split:, :], "Price of BTC", "Bitcoin (TEST)"
        )
        super().__init__(train, test, "Daily Price of BTC")

    def _process_df(self, df):
        df = df.loc[df["series_name"] == "price"]

        start_ts = df["start_timestamp"].item()
        series = df["values"].item()
        series = pd.DataFrame({"values": series})
        indices = pd.to_datetime(
            [start_ts + timedelta(days=i) for i in range(series.shape[0])]
        )
        series.set_index(
            indices,
            inplace=True,
        )

        return series


class COVID(Dataset):
    def __init__(self):
        df, frequency, forecast_horizon, _, _ = convert_tsf_to_dataframe(
            "./datasets/covid_deaths_dataset.tsf", value_column_name="values"
        )
        series = self._process_df(df)
        train_split = int(series.shape[0] * 0.8)
        train = TimeSeries(series.iloc[:train_split, :], "Deaths", "COVID (TRAIN)")
        test = TimeSeries(series.iloc[train_split:, :], "Deaths", "COVID (TEST)")
        super().__init__(
            train,
            test,
            "Daily COVID deaths from 22/01/2020 to 20/08/2020 (in an unknown region)",
        )

    def _process_df(self, df):
        df = df.loc[df["series_name"] == "T1"]
        start_ts = df["start_timestamp"].item()
        series = df["values"].item()
        series = pd.DataFrame({"values": series})
        indices = pd.to_datetime(
            [start_ts + timedelta(days=i) for i in range(series.shape[0])]
        )
        series.set_index(
            indices,
            inplace=True,
        )
        return series


class EnergyConsumption(Dataset):

    def __init__(self):
        df = pd.read_csv(
            "./datasets/energydata_complete.csv", header=0, engine="python"
        )

        series = self._process_df(df)
        train_split = int(series.shape[0] * 0.8)

        train = TimeSeries(
            series.iloc[:train_split, :1],
            "Energy consumed (in kWh)",
            "Energy consumption (TRAIN)",
        )
        test = TimeSeries(
            series.iloc[train_split:, :1],
            "Energy consumed (in kWh)",
            "Energy consumption (TEST)",
        )

        super().__init__(train, test, "Energy consumption of appliances (in kWh)")

    def _process_df(self, df):
        df.set_index("date", inplace=True)
        datetime_indices = pd.to_datetime(df.index)
        hourly_bins = datetime_indices.floor("H")
        df = df.groupby(hourly_bins, axis=0).mean().round(3)
        return df


class Pedestrian(Dataset):

    def __init__(self):
        df, frequency, forecast_horizon, _, _ = convert_tsf_to_dataframe(
            "./datasets/pedestrian_counts_dataset.tsf", value_column_name="values"
        )

        df = df.loc[df["series_name"] == "T1"]
        series = self._process_df(df)
        train_split = int(series.shape[0] * 0.8)
        train = TimeSeries(
            series.iloc[:train_split, :], "Pedestrain count", "Pedestrian (TRAIN)"
        )
        test = TimeSeries(
            series.iloc[train_split:, :], "Pedestrain count", "Pedestrian (TEST)"
        )

        super().__init__(
            train,
            test,
            "Daily average pedestrian count caught by Sensor 1 in Melbourne",
        )

    def _process_df(self, df):
        start_ts = df["start_timestamp"].item()
        series = df["values"].item()
        series = pd.DataFrame({"values": series})
        indices = pd.to_datetime(
            [start_ts + timedelta(hours=i) for i in range(series.shape[0])]
        )
        hourly_bins = indices.floor("D")
        series.set_index(
            indices,
            inplace=True,
        )

        series = series.groupby(hourly_bins, axis=0).mean().round(3)
        return series


class Tourism(Dataset):

    def __init__(self):
        (
            df,
            frequency,
            forecast_horizon,
            _,
            _,
        ) = convert_tsf_to_dataframe(
            "./datasets/tourism/tourism_monthly_dataset.tsf", value_column_name="values"
        )

        self.prediction_length = forecast_horizon
        self.frequency = frequency

        series = self._process_df(df)

        train_split = int(series.shape[0] * 0.8)
        train = TimeSeries(
            series.iloc[:train_split, :], "No. of tourists", "Tourism (TRAIN)"
        )
        test = TimeSeries(
            series.iloc[train_split:, :], "No. of tourists", "Tourism (TEST)"
        )
        super().__init__(
            train, test, "Total tourism numbers at country level of aggregation"
        )

    def _process_df(self, df):
        df = df.loc[df["series_name"] == "T1"]
        start_ts = df["start_timestamp"].item()
        series = df["values"].item()
        series = pd.DataFrame({"values": series})
        series.set_index(
            pd.to_datetime(
                [start_ts + pd.DateOffset(months=i) for i in range(series.shape[0])]
            ),
            inplace=True,
        )
        return series


class Traffic(Dataset):

    def __init__(self):
        (
            df,
            frequency,
            forecast_horizon,
            _,
            _,
        ) = convert_tsf_to_dataframe(
            "./datasets/traffic_weekly_dataset.tsf", value_column_name="values"
        )
        # Just get the first row
        series = self._process_df(df)
        train_split = int(series.shape[0] * 0.8)
        train = TimeSeries(
            series.iloc[:train_split, :], "Road occupancy", "Traffic (TRAIN)"
        )
        test = TimeSeries(
            series.iloc[train_split:, :], "Road occupancy", "Traffic (TEST)"
        )
        title = "Weekly road occupancy rates on San Francisco Bay area freeways (2015 - 2016)"

        super().__init__(train, test, title)

    def _process_df(self, df):
        df = df.loc[df["series_name"] == "T1"]
        start_ts = df["start_timestamp"].item()
        series = df["values"].item()
        series = pd.DataFrame(
            {"values": series},
            index=pd.to_datetime(
                [start_ts + timedelta(weeks=i) for i in range(series.shape[0])]
            ),
        )
        return series


class Web(Dataset):

    def __init__(self):
        (
            df,
            frequency,
            forecast_horizon,
            _,
            _,
        ) = convert_tsf_to_dataframe(
            "./datasets/kaggle_web_traffic_weekly_dataset.tsf",
            value_column_name="values",
        )

        df = df.loc[df["series_name"] == "T1"]
        series = self._process_df(df)
        train_split = int(series.shape[0] * 0.8)
        train = TimeSeries(series.iloc[:train_split, :], "Clicks", "Web (TRAIN)")
        test = TimeSeries(series.iloc[train_split:, :], "Clicks", "Web (TEST)")
        title = "Weekly web traffic for a set of Wikipedia pages from 2015-07-01 to 2017-09-05"
        super().__init__(train, test, title)

    def _process_df(self, df):
        start_ts = df["start_timestamp"].item()
        series = df["values"].item()
        series = pd.DataFrame({"values": series})
        indices = pd.to_datetime(
            [start_ts + timedelta(weeks=i) for i in range(series.shape[0])]
        )
        series.set_index(
            indices,
            inplace=True,
        )
        return series


class AirPassengers(Dataset):

    def __init__(self):
        df = pd.read_csv(
            "./datasets/AirPassengers.csv",
            header=0,
            engine="python",
        )
        df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
        df.set_index("Month", inplace=True)

        train_split = int(df.shape[0] * 0.8)
        train = TimeSeries(
            df.iloc[:train_split, :], "No. of passengers", "AirPassengers (TRAIN)"
        )
        test = TimeSeries(
            df.iloc[train_split:, :], "No. of passengers", "AirPassengers (TEST)"
        )

        super().__init__(
            train, test, "Monthly totals of a US airline passengers from 1949 to 1960"
        )
