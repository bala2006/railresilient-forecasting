"""Severe-delay and connection-feasibility utilities."""

from __future__ import annotations

import numpy as np


def exceedance_probability(
    quantile_predictions: np.ndarray, quantiles: list[float], threshold: float
) -> np.ndarray:
    """Approximate P(Y > threshold) by piecewise-linear quantile-CDF interpolation."""
    q = np.asarray(quantiles, dtype=np.float64)
    prediction = np.asarray(quantile_predictions, dtype=np.float64)
    flat = prediction.reshape(-1, prediction.shape[-1])
    probability = np.empty(flat.shape[0], dtype=np.float64)
    for index, row in enumerate(flat):
        if threshold < row[0]:
            cdf = q[0] * max(0.0, 1.0 - (row[0] - threshold) / max(abs(row[0]), 1.0))
        elif threshold >= row[-1]:
            tail_scale = max(row[-1] - row[-2], 1.0)
            cdf = q[-1] + (1.0 - q[-1]) * min((threshold - row[-1]) / tail_scale, 1.0)
        else:
            upper = int(np.searchsorted(row, threshold, side="right"))
            lower = upper - 1
            width = max(row[upper] - row[lower], 1e-6)
            fraction = (threshold - row[lower]) / width
            cdf = q[lower] + fraction * (q[upper] - q[lower])
        probability[index] = np.clip(1.0 - cdf, 1e-6, 1.0 - 1e-6)
    return probability.reshape(prediction.shape[:-1]).astype(np.float32)


def connection_miss_probability(
    arrival_delay_quantiles: np.ndarray,
    quantiles: list[float],
    transfer_buffer_seconds: float | np.ndarray,
) -> np.ndarray:
    """Derived connection-feasibility risk; not an observed passenger outcome."""
    buffers = np.asarray(transfer_buffer_seconds)
    if buffers.ndim == 0:
        return exceedance_probability(arrival_delay_quantiles, quantiles, float(buffers))
    output = np.empty(arrival_delay_quantiles.shape[:-1], dtype=np.float32)
    for index in np.ndindex(output.shape):
        output[index] = exceedance_probability(
            arrival_delay_quantiles[index][None, :], quantiles, float(buffers[index])
        )[0]
    return output
