from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PreprocessingReport:
    issues: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)

    def add_issue(self, message: str) -> None:
        self.issues.append(message)

    def add_fix(self, message: str) -> None:
        self.fixes.append(message)


def normalize_columns(frame: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
    renamed = {column: str(column).strip() for column in frame.columns}
    if renamed != {column: column for column in frame.columns}:
        report.add_fix("Trimmed whitespace from column names.")
    return frame.rename(columns=renamed)


def clean_numeric_column(frame: pd.DataFrame, column: str, report: PreprocessingReport) -> pd.DataFrame:
    original = frame[column].copy()
    if pd.api.types.is_numeric_dtype(original):
        return frame

    cleaned = (
        original.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    recovered = int(numeric.notna().sum() - pd.to_numeric(original, errors="coerce").notna().sum())
    if recovered > 0:
        report.add_fix(f"Recovered {recovered} numeric values from formatted text in '{column}'.")
    frame[column] = numeric
    return frame


def parse_time_column(frame: pd.DataFrame, column: str, report: PreprocessingReport) -> pd.DataFrame:
    series = frame[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        return frame

    stripped = series.astype(str).str.strip()
    parsed = pd.to_datetime(stripped, errors="coerce")
    if parsed.notna().all():
        frame[column] = parsed
        report.add_fix(f"Parsed '{column}' as datetimes.")
        return frame

    shorthand = stripped.str.extract(r"^(?P<year>\d+)-(?P<month>\d{1,2})$")
    if not shorthand.isna().any(axis=None):
        years = shorthand["year"].astype(int) + 2000
        months = shorthand["month"].astype(int)
        if months.between(1, 12).all():
            normalized = pd.to_datetime(
                {
                    "year": years,
                    "month": months,
                    "day": 1,
                },
                errors="coerce",
            )
            if normalized.notna().all():
                frame[column] = normalized
                report.add_fix(
                    f"Converted shorthand year-month values in '{column}' to first-of-month datetimes."
                )
                return frame

    frame[column] = parsed
    invalid_count = int(parsed.isna().sum())
    if invalid_count:
        report.add_issue(f"'{column}' still has {invalid_count} unparseable datetime values.")
    return frame


def finalize_time_series(
    frame: pd.DataFrame,
    time_col: str,
    target_col: str,
    report: PreprocessingReport,
) -> pd.DataFrame:
    if frame[time_col].duplicated().any():
        duplicates = int(frame[time_col].duplicated().sum())
        report.add_issue(f"Found {duplicates} duplicate timestamps in '{time_col}'.")
        frame = frame.groupby(time_col, as_index=False)[target_col].mean()
        report.add_fix(f"Aggregated duplicate timestamps in '{time_col}' using the mean of '{target_col}'.")

    if not frame[time_col].is_monotonic_increasing:
        frame = frame.sort_values(time_col).reset_index(drop=True)
        report.add_fix(f"Sorted rows by '{time_col}'.")

    if frame[target_col].isna().any():
        missing = int(frame[target_col].isna().sum())
        report.add_issue(f"Found {missing} missing values in '{target_col}'.")
        frame[target_col] = frame[target_col].interpolate(method="linear", limit_direction="both")
        remaining = int(frame[target_col].isna().sum())
        if remaining == 0:
            report.add_fix(f"Filled missing values in '{target_col}' with linear interpolation.")
        else:
            report.add_issue(f"{remaining} missing values in '{target_col}' remain after interpolation.")

    return frame
