from itertools import islice
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
import pandas as pd
import numpy as np

from lag_llama.gluon.estimator import LagLlamaEstimator

import lib.new_datasets as ds

import sys
from types import ModuleType


def create_dummy_module(module_path):
    """
    Create a dummy module hierarchy for the given path.
    Returns the leaf module.
    """
    parts = module_path.split(".")
    current = ""
    parent = None

    for part in parts:
        current = current + "." + part if current else part
        if current not in sys.modules:
            module = ModuleType(current)
            sys.modules[current] = module
            if parent:
                setattr(sys.modules[parent], part, module)
        parent = current

    return sys.modules[module_path]


# Create the dummy gluonts module hierarchy
gluonts_module = create_dummy_module("gluonts.torch.modules.loss")


class DistributionLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


gluonts_module.DistributionLoss = DistributionLoss


from torch.serialization import add_safe_globals


add_safe_globals([StudentTOutput, NegativeLogLikelihood])


class LlagLlamaExperiment:

    def __init__(self):
        self.prediction_length = 0
        self.context_length = 0

    def _get_lag_llama_predictions(
        self,
        dataset,
        prediction_length,
        device,
        context_length=32,
        use_rope_scaling=False,
        num_samples=100,
    ):
        ckpt = torch.load(
            "./lag-llama/lag-llama.ckpt", map_location=device, weights_only=False
        )  # Uses GPU since in this Colab we use a GPU.
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(
                1.0,
                (context_length + prediction_length) / estimator_args["context_length"],
            ),
        }

        estimator = LagLlamaEstimator(
            ckpt_path="./lag-llama/lag-llama.ckpt",
            prediction_length=prediction_length,
            context_length=context_length,  # Lag-Llama was trained with a context length of 32, but can work with any context length
            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
            batch_size=1,
            num_parallel_samples=100,
            device=device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset, predictor=predictor, num_samples=num_samples
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)

        return forecasts, tss

    def _convert_to_float(self, df):
        for col in df.columns:
            if (
                df[col].dtype != "object"
                and pd.api.types.is_string_dtype(df[col]) == False
            ):
                df[col] = df[col].astype("float32")

    def __call__(
        self,
        dataset,
        threshold,
        prediction_length=None,
        context_length=None,
        num_samples=100,
        device=torch.device("cpu"),
    ):
        dfs = dataset.train

        if not prediction_length:
            self.prediction_length = dataset.prediction_length
        else:
            self.prediction_length = prediction_length

        if not context_length:
            self.context_length = self.prediction_length * 6
        else:
            self.context_length = context_length

        frames = []
        for df in dfs[:9]:
            frame = df.series
            frame["id"] = df.name
            frames.append(frame)

        final_df = pd.concat(frames)
        print(final_df.head())
        self._convert_to_float(final_df)

        dataset = PandasDataset.from_long_dataframe(
            final_df, target="values", item_id="id"
        )
        print(self.prediction_length)
        print("Beginning prediction on raw time series")
        forecasts, tss = self._get_lag_llama_predictions(
            dataset,
            self.prediction_length,
            device,
            num_samples=num_samples,
            context_length=self.context_length,
        )
        print("Finished prediction on raw time series")

        frames = []
        print("Filtering signals")
        for df in dfs[:9]:
            filtered, _ = df.filter(threshold)
            filtered["id"] = df.name + " filtered"
            frames.append(filtered)
        filtered_final_df = pd.concat(frames)
        self._convert_to_float(filtered_final_df)

        filtered_dataset = PandasDataset.from_long_dataframe(
            filtered_final_df, target="values", item_id="id"
        )

        print("Beginning prediction on filtered time series")
        filtered_forecasts, _ = self._get_lag_llama_predictions(
            filtered_dataset,
            self.prediction_length,
            device,
            num_samples=num_samples,
            context_length=self.context_length,
        )
        print("Finished prediction on filtered time series")

        return forecasts, filtered_forecasts, tss

    def evaluate(self, forecasts, filtered_forecasts, tss):
        evaluator = Evaluator(num_workers=0)
        agg_metrics, _ = evaluator(iter(list(tss)), iter(list(forecasts)))
        filtered_agg_metrics, _ = evaluator(
            iter(list(tss)), iter(list(filtered_forecasts))
        )

        return agg_metrics, filtered_agg_metrics

    def plot_results(self, forecasts, tss, filename, agg_metrics=None):
        plt.figure(figsize=(20, 20))
        date_formater = mdates.DateFormatter("%b %Y")
        plt.rcParams.update({"font.size": 12})
        for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
            ax = plt.subplot(3, 3, idx + 1)

            plt.subplots_adjust(top=0.9, bottom=0.12)

            plt.plot(
                ts[-self.context_length :].to_timestamp(),
                label="target",
            )
            forecast.plot(color="g")
            plt.xticks(rotation=45)
            ax.xaxis.set_major_formatter(date_formater)
            ax.set_title(forecast.item_id)

        plt.gcf().tight_layout()
        plt.legend()
        plt.savefig(filename)
        plt.show()

    def plot_together(self, raw_forecast, filtered_forecast, filename):
        plt.figure(figsize=(20, 20))
        date_formater = mdates.DateFormatter("%b %Y")
        plt.rcParams.update({"font.size": 12})
        for idx, (raw, filtered) in islice(
            enumerate(zip(raw_forecast, filtered_forecast)), 9
        ):
            ax = plt.subplot(3, 3, idx + 1)

            raw.plot(color="g")
            filtered.plot(color="r")

            plt.xticks(rotation=45)
            ax.xaxis.set_major_formatter(date_formater)
            ax.set_title(f"{raw.item_id} + {filtered.item_id}")

        plt.gcf().tight_layout()
        plt.legend()
        plt.savefig(filename)
        plt.show()


if __name__ == "__main__":
    exp = LlagLlamaExperiment()
    tourism = ds.Tourism()
    forecasts, filtered_forecasts, tss = exp(
        tourism, 40, prediction_length=6, context_length=12 * 6
    )
    raw_metrics, filtered_metrics = exp.evaluate(forecasts, filtered_forecasts, tss)
    exp.plot_results(forecasts, tss, "./results/experiments/tourism_raw_forecast.png")
    exp.plot_results(
        filtered_forecasts, tss, "./results/experiments/tourism_filtered_forecasts.png"
    )
    exp.plot_together(
        forecasts,
        filtered_forecasts,
        "./results/experiments/tourism_filtered_raw_comparison.png",
    )
    print("CRPS (raw):", raw_metrics["mean_wQuantileLoss"])
    print("RMSE (raw):", np.sqrt(raw_metrics["MSE"]))
    print("CRPS (filtered):", filtered_metrics["mean_wQuantileLoss"])
    print("RMSE (filtered):", np.sqrt(filtered_metrics["MSE"]))
