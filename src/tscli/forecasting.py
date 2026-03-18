from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from darts import TimeSeries

from tscli.data import LoadedSeries


SUPPORTED_MODELS = (
    "naive-drift",
    "naive-seasonal",
    "moving-average",
    "linear-trend",
)


@dataclass
class ForecastResult:
    model_name: str
    forecast_frame: pd.DataFrame


def build_series(dataset: LoadedSeries) -> TimeSeries:
    frame = dataset.frame[[dataset.time_col, dataset.target_col]].dropna().copy()

    if dataset.time_col == "__index__":
        frame[dataset.time_col] = pd.RangeIndex(start=0, stop=len(frame), step=1)
        return TimeSeries.from_dataframe(
            frame,
            time_col=dataset.time_col,
            value_cols=dataset.target_col,
        )

    return TimeSeries.from_dataframe(
        frame,
        time_col=dataset.time_col,
        value_cols=dataset.target_col,
        fill_missing_dates=False,
    )


def _future_index(dataset: LoadedSeries, horizon: int) -> pd.Index:
    frame = dataset.frame[[dataset.time_col, dataset.target_col]].dropna().copy()

    if dataset.time_col == "__index__":
        start = int(frame[dataset.time_col].iloc[-1]) + 1
        return pd.RangeIndex(start=start, stop=start + horizon, step=1)

    inferred = pd.infer_freq(frame[dataset.time_col])
    if inferred:
        offset = pd.tseries.frequencies.to_offset(inferred)
        start = frame[dataset.time_col].iloc[-1] + offset
        return pd.date_range(start=start, periods=horizon, freq=offset)

    if len(frame) >= 2:
        delta = frame[dataset.time_col].iloc[-1] - frame[dataset.time_col].iloc[-2]
    else:
        delta = pd.Timedelta(days=1)
    start = frame[dataset.time_col].iloc[-1] + delta
    return pd.Index([start + delta * step for step in range(horizon)])


def generate_forecast(
    dataset: LoadedSeries,
    model_name: str,
    horizon: int,
    seasonal_period: int = 12,
) -> ForecastResult:
    if model_name not in SUPPORTED_MODELS:
        allowed = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {allowed}.")

    series = build_series(dataset)
    history = series.values(copy=False).reshape(-1)
    if len(history) == 0:
        raise ValueError("The target series is empty after dropping missing values.")

    if model_name == "naive-drift":
        if len(history) == 1:
            forecast_values = np.repeat(history[-1], horizon)
        else:
            slope = (history[-1] - history[0]) / (len(history) - 1)
            forecast_values = np.array([history[-1] + slope * step for step in range(1, horizon + 1)])
    elif model_name == "naive-seasonal":
        if len(history) < seasonal_period:
            raise ValueError(
                f"Naive seasonal needs at least {seasonal_period} observations, but found {len(history)}."
            )
        pattern = history[-seasonal_period:]
        forecast_values = np.array([pattern[step % seasonal_period] for step in range(horizon)])
    elif model_name == "moving-average":
        window = min(seasonal_period, len(history))
        average = float(np.mean(history[-window:]))
        forecast_values = np.repeat(average, horizon)
    else:
        x = np.arange(len(history), dtype=float)
        slope, intercept = np.polyfit(x, history.astype(float), 1)
        future_x = np.arange(len(history), len(history) + horizon, dtype=float)
        forecast_values = intercept + slope * future_x

    forecast_frame = pd.DataFrame(
        {
            dataset.time_col: _future_index(dataset, horizon),
            dataset.target_col: forecast_values,
        }
    )

    return ForecastResult(model_name=model_name, forecast_frame=forecast_frame)
