from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tscli.data import LoadedSeries
from tscli.model_catalog import SUPPORTED_MODELS


@dataclass(frozen=True)
class ModelSpec:
    description: str
    family: str


MODEL_SPECS = {
    "naive-last": ModelSpec(SUPPORTED_MODELS["naive-last"], "built-in"),
    "naive-drift": ModelSpec(SUPPORTED_MODELS["naive-drift"], "built-in"),
    "naive-seasonal": ModelSpec(SUPPORTED_MODELS["naive-seasonal"], "built-in"),
    "moving-average": ModelSpec(SUPPORTED_MODELS["moving-average"], "built-in"),
    "weighted-moving-average": ModelSpec(SUPPORTED_MODELS["weighted-moving-average"], "built-in"),
    "exp-smoothing": ModelSpec(SUPPORTED_MODELS["exp-smoothing"], "built-in"),
    "seasonal-average": ModelSpec(SUPPORTED_MODELS["seasonal-average"], "built-in"),
    "seasonal-median": ModelSpec(SUPPORTED_MODELS["seasonal-median"], "built-in"),
    "linear-trend": ModelSpec(SUPPORTED_MODELS["linear-trend"], "built-in"),
    "quadratic-trend": ModelSpec(SUPPORTED_MODELS["quadratic-trend"], "built-in"),
    "arima": ModelSpec(SUPPORTED_MODELS["arima"], "darts-classical"),
    "theta": ModelSpec(SUPPORTED_MODELS["theta"], "darts-classical"),
    "exponential-smoothing": ModelSpec(SUPPORTED_MODELS["exponential-smoothing"], "darts-classical"),
    "auto-arima": ModelSpec(SUPPORTED_MODELS["auto-arima"], "darts-classical"),
    "sarima": ModelSpec(SUPPORTED_MODELS["sarima"], "darts-classical"),
}


@dataclass
class ForecastResult:
    model_name: str
    forecast_frame: pd.DataFrame


@dataclass
class EvaluationResult:
    model_name: str
    mae: float
    rmse: float
    mape: float


@dataclass
class BenchmarkResult:
    scores: list[EvaluationResult]
    actual_frame: pd.DataFrame
    forecasts: dict[str, pd.DataFrame]
    best_model: str
    skipped_models: dict[str, str]


def build_series(dataset: LoadedSeries) -> Any:
    from darts import TimeSeries

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


def _future_index(frame: pd.DataFrame, time_col: str, horizon: int) -> pd.Index:
    if time_col == "__index__":
        start = int(frame[time_col].iloc[-1]) + 1
        return pd.RangeIndex(start=start, stop=start + horizon, step=1)

    inferred = pd.infer_freq(frame[time_col])
    if inferred:
        offset = pd.tseries.frequencies.to_offset(inferred)
        start = frame[time_col].iloc[-1] + offset
        return pd.date_range(start=start, periods=horizon, freq=offset)

    if len(frame) >= 2:
        delta = frame[time_col].iloc[-1] - frame[time_col].iloc[-2]
    else:
        delta = pd.Timedelta(days=1)
    start = frame[time_col].iloc[-1] + delta
    return pd.Index([start + delta * step for step in range(horizon)])


def _coerce_output_frame(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    export = frame.copy()
    if time_col != "__index__" and pd.api.types.is_datetime64_any_dtype(export[time_col]):
        export[time_col] = export[time_col].dt.strftime("%Y-%m-%d")
    return export


def _dataset_from_frame(dataset: LoadedSeries, frame: pd.DataFrame) -> LoadedSeries:
    return LoadedSeries(
        source=dataset.source,
        frame=frame.reset_index(drop=True),
        time_col=dataset.time_col,
        target_col=dataset.target_col,
        report=dataset.report,
    )


def _load_optional_model(model_name: str) -> Any:
    loaders = {
        "arima": ("darts.models.forecasting.arima", "ARIMA"),
        "theta": ("darts.models.forecasting.theta", "Theta"),
        "exponential-smoothing": ("darts.models.forecasting.exponential_smoothing", "ExponentialSmoothing"),
        "auto-arima": ("darts.models.forecasting.sf_auto_arima", "AutoARIMA"),
        "sarima": ("darts.models.forecasting.arima", "ARIMA"),
    }
    module_name, class_name = loaders[model_name]
    try:
        module = import_module(module_name)
        return getattr(module, class_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        extra_hint = "Install tscli with the right optional extra."
        if model_name in {"arima", "theta", "exponential-smoothing", "sarima"}:
            extra_hint = "Install tscli with the 'classical' extra: pip install -e .[classical]"
        elif model_name == "auto-arima":
            extra_hint = (
                "Install tscli with the 'autoarima' extra, or 'full' for everything: "
                "pip install -e .[autoarima]"
            )
        raise ValueError(
            f"Model '{model_name}' is unavailable in this environment. {extra_hint}. "
            f"Original error: {exc}"
        ) from exc


def _heuristic_forecast_values(
    history: np.ndarray,
    model_name: str,
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:
    if len(history) == 0:
        raise ValueError("The target series is empty after dropping missing values.")

    if model_name == "naive-last":
        return np.repeat(history[-1], horizon)

    if model_name == "naive-drift":
        if len(history) == 1:
            return np.repeat(history[-1], horizon)
        slope = (history[-1] - history[0]) / (len(history) - 1)
        return np.array([history[-1] + slope * step for step in range(1, horizon + 1)])

    if model_name == "naive-seasonal":
        if len(history) < seasonal_period:
            raise ValueError(
                f"Naive seasonal needs at least {seasonal_period} observations, but found {len(history)}."
            )
        pattern = history[-seasonal_period:]
        return np.array([pattern[step % seasonal_period] for step in range(horizon)])

    if model_name == "moving-average":
        window = min(seasonal_period, len(history))
        average = float(np.mean(history[-window:]))
        return np.repeat(average, horizon)

    if model_name == "weighted-moving-average":
        window = min(seasonal_period, len(history))
        weights = np.arange(1, window + 1, dtype=float)
        average = float(np.average(history[-window:], weights=weights))
        return np.repeat(average, horizon)

    if model_name == "exp-smoothing":
        span = max(2, min(seasonal_period, len(history)))
        level = float(pd.Series(history).ewm(span=span, adjust=False).mean().iloc[-1])
        return np.repeat(level, horizon)

    if model_name == "seasonal-average":
        if len(history) < seasonal_period:
            raise ValueError(
                f"Seasonal average needs at least {seasonal_period} observations, but found {len(history)}."
            )
        values = []
        for step in range(horizon):
            position = step % seasonal_period
            seasonal_slice = history[position::seasonal_period]
            values.append(float(np.mean(seasonal_slice)))
        return np.array(values)

    if model_name == "seasonal-median":
        if len(history) < seasonal_period:
            raise ValueError(
                f"Seasonal median needs at least {seasonal_period} observations, but found {len(history)}."
            )
        values = []
        for step in range(horizon):
            position = step % seasonal_period
            seasonal_slice = history[position::seasonal_period]
            values.append(float(np.median(seasonal_slice)))
        return np.array(values)

    if model_name == "quadratic-trend":
        if len(history) < 3:
            raise ValueError("Quadratic trend needs at least 3 observations.")
        x = np.arange(len(history), dtype=float)
        a, b, c = np.polyfit(x, history.astype(float), 2)
        future_x = np.arange(len(history), len(history) + horizon, dtype=float)
        return a * np.square(future_x) + b * future_x + c

    x = np.arange(len(history), dtype=float)
    slope, intercept = np.polyfit(x, history.astype(float), 1)
    future_x = np.arange(len(history), len(history) + horizon, dtype=float)
    return intercept + slope * future_x


def _darts_classical_forecast(
    dataset: LoadedSeries,
    model_name: str,
    horizon: int,
    seasonal_period: int,
) -> pd.DataFrame:
    series = build_series(dataset)
    model_class = _load_optional_model(model_name)

    if model_name == "theta":
        model = model_class()
    elif model_name == "exponential-smoothing":
        kwargs: dict[str, Any] = {}
        if len(series) >= seasonal_period * 2:
            kwargs["seasonal_periods"] = seasonal_period
        model = model_class(**kwargs)
    elif model_name == "auto-arima":
        kwargs = {}
        if len(series) >= seasonal_period * 2:
            kwargs["season_length"] = seasonal_period
        model = model_class(**kwargs)
    elif model_name == "sarima":
        kwargs = {"p": 1, "d": 1, "q": 1}
        if len(series) >= seasonal_period * 2:
            kwargs["seasonal_order"] = (1, 1, 1, seasonal_period)
        model = model_class(**kwargs)
    else:
        model = model_class(p=1, d=1, q=1)

    try:
        model.fit(series)
        forecast = model.predict(horizon)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ValueError(f"Model '{model_name}' failed to fit or predict. Original error: {exc}") from exc

    try:
        if hasattr(forecast, "pd_dataframe"):
            forecast_frame = forecast.pd_dataframe().reset_index()
        elif hasattr(forecast, "to_dataframe"):
            forecast_frame = forecast.to_dataframe().reset_index()
        elif hasattr(forecast, "pd_series"):
            forecast_frame = forecast.pd_series().to_frame(name=dataset.target_col).reset_index()
        else:
            raise AttributeError("No supported dataframe conversion method was found on the DARTS TimeSeries result.")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ValueError(
            f"Model '{model_name}' produced a forecast, but tscli could not convert it to a table. "
            f"Original error: {exc}"
        ) from exc

    forecast_frame.columns = [dataset.time_col, dataset.target_col]
    return forecast_frame


def _statsmodels_fallback_forecast(
    dataset: LoadedSeries,
    model_name: str,
    horizon: int,
    seasonal_period: int,
) -> pd.DataFrame:
    from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    frame = dataset.frame[[dataset.time_col, dataset.target_col]].dropna().copy()
    history = frame[dataset.target_col].astype(float).to_numpy()

    try:
        if model_name == "arima":
            fitted = StatsmodelsARIMA(history, order=(1, 1, 1)).fit()
        else:
            seasonal_order = (1, 1, 1, seasonal_period)
            if len(history) < max(24, seasonal_period * 2 + 6):
                seasonal_order = (0, 1, 1, seasonal_period)
            fitted = SARIMAX(
                history,
                order=(1, 1, 1),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        forecast_values = np.asarray(fitted.forecast(steps=horizon), dtype=float)
    except Exception as exc:
        raise ValueError(
            f"Model '{model_name}' is unavailable through DARTS and the fallback implementation also failed. "
            f"Original error: {exc}"
        ) from exc

    return pd.DataFrame(
        {
            dataset.time_col: _future_index(frame, dataset.time_col, horizon),
            dataset.target_col: forecast_values,
        }
    )


def generate_forecast(
    dataset: LoadedSeries,
    model_name: str,
    horizon: int,
    seasonal_period: int = 12,
) -> ForecastResult:
    if model_name not in SUPPORTED_MODELS:
        allowed = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {allowed}.")

    frame = dataset.frame[[dataset.time_col, dataset.target_col]].dropna().copy()
    family = MODEL_SPECS[model_name].family

    if family == "built-in":
        series = build_series(dataset)
        history = series.values(copy=False).reshape(-1)
        forecast_values = _heuristic_forecast_values(history, model_name, horizon, seasonal_period)
        forecast_frame = pd.DataFrame(
            {
                dataset.time_col: _future_index(frame, dataset.time_col, horizon),
                dataset.target_col: forecast_values,
            }
        )
    else:
        try:
            forecast_frame = _darts_classical_forecast(dataset, model_name, horizon, seasonal_period)
        except ValueError as exc:
            if model_name in {"arima", "sarima"} and "unavailable in this environment" in str(exc):
                forecast_frame = _statsmodels_fallback_forecast(dataset, model_name, horizon, seasonal_period)
            else:
                raise

    return ForecastResult(model_name=model_name, forecast_frame=forecast_frame)


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> EvaluationResult:
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    non_zero_mask = actual != 0
    if non_zero_mask.any():
        mape = float(np.mean(np.abs(errors[non_zero_mask] / actual[non_zero_mask])) * 100)
    else:
        mape = float("nan")

    return EvaluationResult(model_name=model_name, mae=mae, rmse=rmse, mape=mape)


def benchmark_models(
    dataset: LoadedSeries,
    model_names: list[str],
    horizon: int,
    seasonal_period: int = 12,
) -> BenchmarkResult:
    clean_frame = dataset.frame[[dataset.time_col, dataset.target_col]].dropna().reset_index(drop=True)
    if len(clean_frame) <= horizon:
        raise ValueError(
            f"Need more than {horizon} observations to benchmark models, but found {len(clean_frame)}."
        )

    train_frame = clean_frame.iloc[:-horizon].copy()
    actual_frame = clean_frame.iloc[-horizon:].copy().reset_index(drop=True)
    train_dataset = _dataset_from_frame(dataset, train_frame)

    scores: list[EvaluationResult] = []
    forecasts: dict[str, pd.DataFrame] = {}
    skipped_models: dict[str, str] = {}
    from tqdm import tqdm

    for model_name in tqdm(model_names, desc="Benchmarking models", unit="model"):
        try:
            result = generate_forecast(
                train_dataset,
                model_name=model_name,
                horizon=horizon,
                seasonal_period=seasonal_period,
            )
            predicted = result.forecast_frame[dataset.target_col].to_numpy(dtype=float)
            actual = actual_frame[dataset.target_col].to_numpy(dtype=float)
            scores.append(evaluate_forecast(actual, predicted, model_name))
            forecasts[model_name] = result.forecast_frame
        except Exception as exc:
            skipped_models[model_name] = str(exc)

    if not scores:
        raise ValueError("No models could be benchmarked successfully with the current data and settings.")

    scores.sort(key=lambda item: (item.rmse, item.mae, item.model_name))
    return BenchmarkResult(
        scores=scores,
        actual_frame=actual_frame,
        forecasts=forecasts,
        best_model=scores[0].model_name,
        skipped_models=skipped_models,
    )


def export_frame(frame: pd.DataFrame, output_path: Path, time_col: str) -> None:
    export = _coerce_output_frame(frame, time_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False)


def export_scores(scores: list[EvaluationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_frame = pd.DataFrame(
        [
            {
                "model": score.model_name,
                "mae": score.mae,
                "rmse": score.rmse,
                "mape": score.mape,
            }
            for score in scores
        ]
    )
    score_frame.to_csv(output_path, index=False)


def export_forecast_plot(
    history_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    time_col: str,
    target_col: str,
    output_path: Path,
    model_name: str,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_frame[time_col], history_frame[target_col], label="history", linewidth=2)
    ax.plot(forecast_frame[time_col], forecast_frame[target_col], label=f"forecast ({model_name})", linewidth=2)
    ax.set_title(f"Forecast - {model_name}")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def export_benchmark_plot(
    actual_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    time_col: str,
    target_col: str,
    output_path: Path,
    model_name: str,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_frame[time_col], actual_frame[target_col], label="actual", linewidth=2)
    ax.plot(forecast_frame[time_col], forecast_frame[target_col], label=f"predicted ({model_name})", linewidth=2)
    ax.set_title(f"Benchmark Holdout - {model_name}")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
