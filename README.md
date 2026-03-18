# tscli

`tscli` is a command-line tool for time series analysis and forecasting built around [DARTS](https://unit8co.github.io/darts/).

It helps a user:

- load a CSV file
- clean common formatting issues
- inspect the dataset
- run quick descriptive analysis
- generate forecasts with DARTS models
- use an interactive terminal menu to choose actions

## Features

- CSV-based workflow
- automatic datetime parsing
- preprocessing for shorthand month values like `1-01`
- target column and optional time column selection
- descriptive statistics and missing-value checks
- DARTS-powered forecasting with configurable horizon
- interactive mode for non-technical users

## Install

```bash
pip install -e .
```

## Usage

### Show available commands

```bash
tscli --help
```

### Inspect a CSV

```bash
tscli inspect data.csv --time-col date --target-col sales
```

### Analyze a time series

```bash
tscli analyze data.csv --time-col date --target-col sales
```

### Clean a messy CSV before forecasting

```bash
tscli clean examples/sales.csv --time-col Month --target-col Sales --output cleaned_sales.csv
```

### Forecast with DARTS

```bash
tscli forecast data.csv --time-col date --target-col sales --horizon 12 --model naive-drift
```

### Start the interactive menu

```bash
tscli interactive data.csv --time-col date --target-col sales
```

## Forecasting models

The first version exposes a few practical forecasting options over DARTS time-series objects:

- `naive-drift`
- `naive-seasonal`
- `moving-average`
- `linear-trend`

## Notes

- The CSV should include a target column and optionally a time column.
- If no time column is provided, the CLI builds a simple range index.
- If DARTS cannot infer a frequency automatically, the CLI still forecasts using the ordered observations.
- The bundled `examples/sales.csv` shows the shorthand monthly sales format that preprocessing can repair.
