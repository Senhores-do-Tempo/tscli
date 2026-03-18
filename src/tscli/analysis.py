from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tscli.data import LoadedSeries


@dataclass
class SeriesSummary:
    row_count: int
    start: str
    end: str
    missing_target: int
    mean: float
    median: float
    minimum: float
    maximum: float
    std_dev: float
    inferred_frequency: str
    trend_direction: str


def summarize_series(dataset: LoadedSeries) -> SeriesSummary:
    frame = dataset.frame.copy()
    target = frame[dataset.target_col]

    inferred_frequency = "not available"
    if dataset.time_col != "__index__":
        inferred = pd.infer_freq(frame[dataset.time_col])
        if inferred:
            inferred_frequency = inferred

    clean_target = target.dropna()
    if clean_target.empty:
        raise ValueError("The target series is empty after dropping missing values.")

    trend_delta = clean_target.iloc[-1] - clean_target.iloc[0]
    if trend_delta > 0:
        trend_direction = "upward"
    elif trend_delta < 0:
        trend_direction = "downward"
    else:
        trend_direction = "flat"

    return SeriesSummary(
        row_count=len(frame),
        start=str(frame[dataset.time_col].iloc[0]),
        end=str(frame[dataset.time_col].iloc[-1]),
        missing_target=int(target.isna().sum()),
        mean=float(clean_target.mean()),
        median=float(clean_target.median()),
        minimum=float(clean_target.min()),
        maximum=float(clean_target.max()),
        std_dev=float(clean_target.std(ddof=0)),
        inferred_frequency=inferred_frequency,
        trend_direction=trend_direction,
    )


def recent_observations(dataset: LoadedSeries, rows: int = 5) -> pd.DataFrame:
    return dataset.frame[[dataset.time_col, dataset.target_col]].tail(rows).copy()
