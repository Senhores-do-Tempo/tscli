from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tscli.preprocessing import (
    PreprocessingReport,
    clean_numeric_column,
    finalize_time_series,
    normalize_columns,
    parse_time_column,
)


@dataclass
class LoadedSeries:
    source: Path
    frame: pd.DataFrame
    time_col: str
    target_col: str
    report: PreprocessingReport


def load_csv(csv_path: Path, time_col: str | None, target_col: str) -> LoadedSeries:
    frame = pd.read_csv(csv_path)
    report = PreprocessingReport()
    frame = normalize_columns(frame, report)
    if target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the CSV.")

    resolved_time_col = time_col
    if resolved_time_col is None:
        for candidate in ("date", "datetime", "timestamp", "ds", "time"):
            if candidate in frame.columns:
                resolved_time_col = candidate
                break

    if resolved_time_col is not None:
        if resolved_time_col not in frame.columns:
            raise ValueError(f"Time column '{resolved_time_col}' was not found in the CSV.")
        frame = parse_time_column(frame, resolved_time_col, report)
        if frame[resolved_time_col].isna().any():
            raise ValueError(
                f"Time column '{resolved_time_col}' contains values that could not be parsed as datetime."
            )
    else:
        resolved_time_col = "__index__"
        frame[resolved_time_col] = pd.RangeIndex(start=0, stop=len(frame), step=1)
        report.add_fix("Created a synthetic integer time index because no time column was provided.")

    frame = clean_numeric_column(frame, target_col, report)
    if frame[target_col].isna().all():
        raise ValueError(f"Target column '{target_col}' does not contain numeric values.")
    frame = finalize_time_series(frame, resolved_time_col, target_col, report)

    return LoadedSeries(
        source=csv_path,
        frame=frame,
        time_col=resolved_time_col,
        target_col=target_col,
        report=report,
    )
