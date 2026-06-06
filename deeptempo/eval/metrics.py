"""Evaluation metrics for time-series forecasting.

Standard stuff: MSE, MAE, RMSE. Nothing fancy.
"""

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.

    NOTE: sMAPE is controversial. It's included because it's commonly
    reported, but it has known issues (undefined when y_true + y_pred = 0,
    biased toward over-forecasting). Don't use it as your only metric.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    mask = denominator > 1e-8
    return 100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
