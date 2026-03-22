from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from tscli.analysis import recent_observations, summarize_series
from tscli.data import load_csv
from tscli.model_catalog import SUPPORTED_MODELS

app = typer.Typer(help="Time series analysis and forecasting CLI powered by DARTS.")
console = Console()


def _exit_with_message(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


def _parse_model_names(models: str | None) -> list[str]:
    if models is None or models.strip().lower() == "all":
        return sorted(SUPPORTED_MODELS)

    parsed = [name.strip() for name in models.split(",") if name.strip()]
    invalid = [name for name in parsed if name not in SUPPORTED_MODELS]
    if invalid:
        allowed = ", ".join(sorted(SUPPORTED_MODELS))
        invalid_display = ", ".join(invalid)
        raise ValueError(f"Unsupported models: {invalid_display}. Choose from: {allowed}.")
    return parsed


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
    output_path: Path | None,
    plot_output: Path | None,
) -> None:
    from tscli.forecasting import export_forecast_plot, export_frame, generate_forecast

    try:
        dataset = load_csv(csv_path, time_col=time_col, target_col=target_col)
    except ValueError as exc:
        _exit_with_message(str(exc))
    _print_preprocessing_report(dataset)
    try:
        result = generate_forecast(
            dataset,
            model_name=model,
            horizon=horizon,
            seasonal_period=seasonal_period,
        )
    except ValueError as exc:
        message = str(exc)
        if model == "exponential-smoothing" and "unavailable in this environment" in message:
            message += " Try '--model exp-smoothing' for the built-in lightweight fallback."
        _exit_with_message(message)

    forecast_table = Table(title=f"Forecast ({result.model_name})")
    forecast_table.add_column(dataset.time_col, style="cyan")
    forecast_table.add_column(dataset.target_col, style="green")
    for _, row in result.forecast_frame.iterrows():
        forecast_table.add_row(str(row[dataset.time_col]), f"{float(row[dataset.target_col]):.4f}")
    console.print(forecast_table)

    if output_path is not None:
        export_frame(result.forecast_frame, output_path, dataset.time_col)
        console.print(f"Saved forecast to [bold]{output_path}[/bold].")

    if plot_output is not None:
        history_frame = dataset.frame[[dataset.time_col, dataset.target_col]].dropna().copy()
        export_forecast_plot(
            history_frame=history_frame,
            forecast_frame=result.forecast_frame,
            time_col=dataset.time_col,
            target_col=dataset.target_col,
            output_path=plot_output,
            model_name=result.model_name,
        )
        console.print(f"Saved forecast plot to [bold]{plot_output}[/bold].")


def _print_benchmark(
    csv_path: Path,
    time_col: str | None,
    target_col: str,
    models: str | None,
    horizon: int,
    seasonal_period: int,
    scores_output: Path | None,
    forecast_output: Path | None,
    plot_output: Path | None,
) -> None:
    from tscli.forecasting import benchmark_models, export_benchmark_plot, export_frame, export_scores

    try:
        dataset = load_csv(csv_path, time_col=time_col, target_col=target_col)
    except ValueError as exc:
        _exit_with_message(str(exc))
    _print_preprocessing_report(dataset)
    try:
        model_names = _parse_model_names(models)
        result = benchmark_models(
            dataset,
            model_names=model_names,
            horizon=horizon,
            seasonal_period=seasonal_period,
        )
    except ValueError as exc:
        _exit_with_message(str(exc))

    score_table = Table(title=f"Benchmark Results (holdout={horizon})")
    score_table.add_column("Model", style="cyan")
    score_table.add_column("MAE", style="green")
    score_table.add_column("RMSE", style="green")
    score_table.add_column("MAPE %", style="green")
    for score in result.scores:
        mape_display = "n/a" if pd.isna(score.mape) else f"{score.mape:.2f}"
        score_table.add_row(score.model_name, f"{score.mae:.4f}", f"{score.rmse:.4f}", mape_display)
    console.print(score_table)
    console.print(f"Best model: [bold]{result.best_model}[/bold]")

    if result.skipped_models:
        skipped_table = Table(title="Skipped Models")
        skipped_table.add_column("Model", style="cyan")
        skipped_table.add_column("Reason", style="green")
        for model_name, reason in sorted(result.skipped_models.items()):
            skipped_table.add_row(model_name, reason)
        console.print(skipped_table)

    if scores_output is not None:
        export_scores(result.scores, scores_output)
        console.print(f"Saved benchmark scores to [bold]{scores_output}[/bold].")

    if forecast_output is not None:
        export_frame(result.forecasts[result.best_model], forecast_output, dataset.time_col)
        console.print(f"Saved best-model forecast to [bold]{forecast_output}[/bold].")

    if plot_output is not None:
        export_benchmark_plot(
            actual_frame=result.actual_frame,
            forecast_frame=result.forecasts[result.best_model],
            time_col=dataset.time_col,
            target_col=dataset.target_col,
            output_path=plot_output,
            model_name=result.best_model,
        )
        console.print(f"Saved benchmark plot to [bold]{plot_output}[/bold].")


def _write_cleaned_csv(csv_path: Path, time_col: str | None, target_col: str, output_path: Path) -> None:
    from tscli.forecasting import export_frame

    try:
        dataset = load_csv(csv_path, time_col=time_col, target_col=target_col)
    except ValueError as exc:
        _exit_with_message(str(exc))
    export_frame(dataset.frame, output_path, dataset.time_col)
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
        help="Season length used by seasonal models.",
    ),
    output_path: Path | None = typer.Option(None, "--output", help="Optional path to save the forecast CSV."),
    plot_output: Path | None = typer.Option(None, "--plot-output", help="Optional path to save a forecast plot."),
) -> None:
    """Generate a forecast using a supported model."""
    _print_forecast(
        csv_path,
        time_col=time_col,
        target_col=target_col,
        model=model,
        horizon=horizon,
        seasonal_period=seasonal_period,
        output_path=output_path,
        plot_output=plot_output,
    )


@app.command()
def benchmark(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input CSV file."),
    target_col: str = typer.Option(..., "--target-col", help="Numeric column to evaluate."),
    time_col: str | None = typer.Option(None, "--time-col", help="Datetime column in the CSV."),
    models: str | None = typer.Option(
        "all",
        "--models",
        help="Comma-separated model names to compare, or 'all'.",
    ),
    horizon: int = typer.Option(12, "--horizon", min=1, help="Holdout length for evaluation."),
    seasonal_period: int = typer.Option(
        12,
        "--seasonal-period",
        min=1,
        help="Season length used by seasonal models.",
    ),
    scores_output: Path | None = typer.Option(
        None,
        "--scores-output",
        help="Optional path to save benchmark scores as CSV.",
    ),
    forecast_output: Path | None = typer.Option(
        None,
        "--forecast-output",
        help="Optional path to save the best model forecast as CSV.",
    ),
    plot_output: Path | None = typer.Option(
        None,
        "--plot-output",
        help="Optional path to save a plot of the best benchmark model on the holdout window.",
    ),
) -> None:
    """Compare multiple models on a holdout window and report the best one."""
    _print_benchmark(
        csv_path,
        time_col=time_col,
        target_col=target_col,
        models=models,
        horizon=horizon,
        seasonal_period=seasonal_period,
        scores_output=scores_output,
        forecast_output=forecast_output,
        plot_output=plot_output,
    )


@app.command()
def models() -> None:
    """List supported forecasting models."""
    table = Table(title="Supported Models")
    table.add_column("Model", style="cyan")
    table.add_column("Description", style="green")
    for model_name, description in sorted(SUPPORTED_MODELS.items()):
        table.add_row(model_name, description)
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
    try:
        dataset_loaded = load_csv(csv_path, time_col=time_col, target_col=target_col)
    except ValueError as exc:
        _exit_with_message(str(exc))
    console.print(f"Loaded [bold]{dataset_loaded.source}[/bold] with target column [bold]{target_col}[/bold].")

    while True:
        console.print(
            "\nChoose an action:\n"
            "1. Inspect dataset\n"
            "2. Analyze series\n"
            "3. Forecast series\n"
            "4. Benchmark models\n"
            "5. List models\n"
            "6. Exit\n"
        )

        choice = Prompt.ask("Selection", choices=["1", "2", "3", "4", "5", "6"], default="1")
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
                output_path=None,
                plot_output=None,
            )
        elif choice == "4":
            horizon = int(Prompt.ask("Benchmark holdout horizon", default="12"))
            seasonal_period = int(Prompt.ask("Seasonal period", default="12"))
            _print_benchmark(
                csv_path,
                time_col=time_col,
                target_col=target_col,
                models="all",
                horizon=horizon,
                seasonal_period=seasonal_period,
                scores_output=None,
                forecast_output=None,
                plot_output=None,
            )
        elif choice == "5":
            models()
        else:
            console.print("Exiting tscli.")
            break


def main() -> None:
    app()


if __name__ == "__main__":
    main()
