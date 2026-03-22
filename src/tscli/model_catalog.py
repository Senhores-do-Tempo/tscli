from __future__ import annotations


SUPPORTED_MODELS = {
    "arima": "DARTS ARIMA model for classical forecasting.",
    "auto-arima": "DARTS AutoARIMA model with automatic order selection.",
    "exp-smoothing": "Forecasts with an exponentially weighted moving average level.",
    "exponential-smoothing": "DARTS ExponentialSmoothing model for level, trend, and seasonality.",
    "linear-trend": "Fits a straight trend line across the series.",
    "moving-average": "Forecasts with the mean of the latest seasonal window.",
    "naive-drift": "Extends the line from the first to the last observation.",
    "naive-last": "Repeats the last observed value.",
    "naive-seasonal": "Repeats the last seasonal pattern.",
    "quadratic-trend": "Fits a quadratic trend curve across the series.",
    "sarima": "DARTS ARIMA model configured with seasonal ARIMA defaults.",
    "seasonal-average": "Forecasts each seasonal position with the average of past matching positions.",
    "seasonal-median": "Forecasts each seasonal position with the median of past matching positions.",
    "theta": "DARTS Theta model for classical univariate forecasting.",
    "weighted-moving-average": "Forecasts with a linearly weighted average of the latest seasonal window.",
}
