from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from tscli.analysis import recent_observations, summarize_series
from tscli.data import load_csv
from tscli.forecasting import SUPPORTED_MODELS, generate_forecast

app = typer.Typer(help="Time series analysis and forecasting CLI powered by DARTS.")
console = Console()


def _print_preprocessing_report(dataset) -> None:
    if not dataset.report.issues and not dataset.report.fixes:
        return

    table = Table(title="Preprocessing Report")
    table.add_column("Type", style="cyan")
    table.add_column("Details", style="green")
    for message in dataset.report.fixes:
        table.add_row("Fix", message)
    for message in dataset.report.issues:
        table.add_row("Issue", message)
    console.print(table)


def _print_dataset_overview(csv_path: Path, time_col: str | None, target_col: str) -> None:
    dataset = load_csv(csv_path, time_col=time_col, target_col=target_col)
    summary = summarize_series(dataset)
    _print_preprocessing_report(dataset)

    overview = Table(title="Dataset Overview")
    overview.add_column("Metric", style="cyan")
    overview.add_column("Value", style="green")
    overview.add_row("Source", str(dataset.source))
    overview.add_row("Rows", str(summary.row_count))
    overview.add_row("Time column", dataset.time_col)
    overview.add_row("Target column", dataset.target_col)
    overview.add_row("Start", summary.start)
    overview.add_row("End", summary.end)
    overview.add_row("Frequency", summary.inferred_frequency)
    overview.add_row("Missing target values", str(summary.missing_target))
    console.print(overview)


def _print_analysis(csv_path: Path, time_col: str | None, target_col: str) -> None:
    dataset = load_csv(csv_path, time_col=time_col, target_col=target_col)
    summary = summarize_series(dataset)
    recent = recent_observations(dataset)
    _print_preprocessing_report(dataset)

    stats = Table(title="Series Analysis")
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green")
    stats.add_row("Mean", f"{summary.mean:.4f}")
    stats.add_row("Median", f"{summary.median:.4f}")
    stats.add_row("Minimum", f"{summary.minimum:.4f}")
    stats.add_row("Maximum", f"{summary.maximum:.4f}")
    stats.add_row("Std. dev.", f"{summary.std_dev:.4f}")
    stats.add_row("Trend", summary.trend_direction)
    console.print(stats)

    tail = Table(title="Recent Observations")
    tail.add_column(dataset.time_col, style="cyan")
    tail.add_column(dataset.target_col, style="green")
    for _, row in recent.iterrows():
        tail.add_row(str(row[dataset.time_col]), f"{float(row[dataset.target_col]):.4f}")
    console.print(tail)


def _print_forecast(
    csv_path: Path,
    time_col: str | None,
    target_col: str,
    model: str,
    horizon: int,
    seasonal_period: int,
) -> None:
    dataset = load_csv(csv_path, time_col=time_col, target_col=target_col)
    _print_preprocessing_report(dataset)
    result = generate_forecast(
        dataset,
        model_name=model,
        horizon=horizon,
        seasonal_period=seasonal_period,
    )

    forecast_table = Table(title=f"Forecast ({result.model_name})")
    forecast_table.add_column(dataset.time_col, style="cyan")
    forecast_table.add_column(dataset.target_col, style="green")
    for _, row in result.forecast_frame.iterrows():
        forecast_table.add_row(str(row[dataset.time_col]), f"{float(row[dataset.target_col]):.4f}")
    console.print(forecast_table)


def _write_cleaned_csv(csv_path: Path, time_col: str | None, target_col: str, output_path: Path) -> None:
    dataset = load_csv(csv_path, time_col=time_col, target_col=target_col)
    export_frame = dataset.frame.copy()
    if dataset.time_col != "__index__":
        export_frame[dataset.time_col] = export_frame[dataset.time_col].dt.strftime("%Y-%m-%d")
    export_frame.to_csv(output_path, index=False)
    _print_preprocessing_report(dataset)
    console.print(f"Saved cleaned dataset to [bold]{output_path}[/bold].")


@app.command()
def inspect(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input CSV file."),
    target_col: str = typer.Option(..., "--target-col", help="Numeric column to analyze and forecast."),
    time_col: str | None = typer.Option(None, "--time-col", help="Datetime column in the CSV."),
) -> None:
    """Inspect the structure of a CSV time series dataset."""
    _print_dataset_overview(csv_path, time_col=time_col, target_col=target_col)


@app.command()
def analyze(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input CSV file."),
    target_col: str = typer.Option(..., "--target-col", help="Numeric column to analyze."),
    time_col: str | None = typer.Option(None, "--time-col", help="Datetime column in the CSV."),
) -> None:
    """Run descriptive analysis on the selected time series."""
    _print_analysis(csv_path, time_col=time_col, target_col=target_col)


@app.command()
def forecast(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input CSV file."),
    target_col: str = typer.Option(..., "--target-col", help="Numeric column to forecast."),
    time_col: str | None = typer.Option(None, "--time-col", help="Datetime column in the CSV."),
    model: str = typer.Option("naive-drift", "--model", help="Forecasting model to use."),
    horizon: int = typer.Option(12, "--horizon", min=1, help="Number of future steps to forecast."),
    seasonal_period: int = typer.Option(
        12,
        "--seasonal-period",
        min=1,
        help="Season length used by the naive seasonal model.",
    ),
) -> None:
    """Generate a forecast using a DARTS model."""
    _print_forecast(
        csv_path,
        time_col=time_col,
        target_col=target_col,
        model=model,
        horizon=horizon,
        seasonal_period=seasonal_period,
    )


@app.command()
def models() -> None:
    """List supported forecasting models."""
    table = Table(title="Supported DARTS Models")
    table.add_column("Model", style="cyan")
    for model_name in sorted(SUPPORTED_MODELS):
        table.add_row(model_name)
    console.print(table)


@app.command()
def clean(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input CSV file."),
    target_col: str = typer.Option(..., "--target-col", help="Numeric column to clean."),
    time_col: str | None = typer.Option(None, "--time-col", help="Datetime column in the CSV."),
    output_path: Path = typer.Option(
        ...,
        "--output",
        help="Where to save the cleaned CSV.",
    ),
) -> None:
    """Clean formatting issues and save a normalized CSV."""
    _write_cleaned_csv(csv_path, time_col=time_col, target_col=target_col, output_path=output_path)


@app.command()
def interactive(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input CSV file."),
    target_col: str = typer.Option(..., "--target-col", help="Numeric column to analyze and forecast."),
    time_col: str | None = typer.Option(None, "--time-col", help="Datetime column in the CSV."),
) -> None:
    """Launch an interactive terminal workflow for a single CSV."""
    dataset_loaded = load_csv(csv_path, time_col=time_col, target_col=target_col)
    console.print(f"Loaded [bold]{dataset_loaded.source}[/bold] with target column [bold]{target_col}[/bold].")

    while True:
        console.print(
            "\nChoose an action:\n"
            "1. Inspect dataset\n"
            "2. Analyze series\n"
            "3. Forecast series\n"
            "4. List models\n"
            "5. Exit\n"
        )

        choice = Prompt.ask("Selection", choices=["1", "2", "3", "4", "5"], default="1")
        if choice == "1":
            _print_dataset_overview(csv_path, time_col=time_col, target_col=target_col)
        elif choice == "2":
            _print_analysis(csv_path, time_col=time_col, target_col=target_col)
        elif choice == "3":
            model = Prompt.ask("Model", choices=sorted(SUPPORTED_MODELS), default="naive-drift")
            horizon = int(Prompt.ask("Forecast horizon", default="12"))
            seasonal_period = int(Prompt.ask("Seasonal period", default="12"))
            _print_forecast(
                csv_path,
                time_col=time_col,
                target_col=target_col,
                model=model,
                horizon=horizon,
                seasonal_period=seasonal_period,
            )
        elif choice == "4":
            models()
        else:
            console.print("Exiting tscli.")
            break


def main() -> None:
    app()


if __name__ == "__main__":
    main()
