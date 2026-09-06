"""Point, probabilistic, alert, routing, and systems metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from railresilient.risk import exceedance_probability


def _mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def interval_score(
    target: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> np.ndarray:
    width = upper - lower
    below = (target < lower) * (2.0 / alpha) * (lower - target)
    above = (target > upper) * (2.0 / alpha) * (target - upper)
    return width + below + above


def weighted_interval_score_values(
    target: np.ndarray, prediction: np.ndarray, quantiles: list[float]
) -> np.ndarray:
    """Return standard WIS per sample and horizon for three central intervals."""
    q_to_index = {round(value, 4): index for index, value in enumerate(quantiles)}
    median = prediction[:, :, q_to_index[0.5]]
    components = 0.5 * np.abs(target - median)
    intervals = (
        (0.5, 0.25, 0.75),
        (0.2, 0.1, 0.9),
        (0.1, 0.05, 0.95),
    )
    for alpha, lower_q, upper_q in intervals:
        components += (alpha / 2.0) * interval_score(
            target,
            prediction[:, :, q_to_index[lower_q]],
            prediction[:, :, q_to_index[upper_q]],
            alpha,
        )
    return components / (len(intervals) + 0.5)


def weighted_interval_score(
    target: np.ndarray, prediction: np.ndarray, quantiles: list[float]
) -> float:
    return _mean(weighted_interval_score_values(target, prediction, quantiles))


def select_alert_threshold(
    prediction: np.ndarray,
    quantiles: list[float],
    severe_threshold: float,
    alert_fraction: float,
) -> float:
    if not 0.0 < alert_fraction < 1.0:
        raise ValueError("Alert fraction must lie strictly between zero and one")
    probabilities = exceedance_probability(
        prediction, quantiles, severe_threshold
    ).ravel()
    return float(np.quantile(probabilities, 1.0 - alert_fraction, method="higher"))


def probabilistic_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    quantiles: list[float],
    persistence_mae: float,
    severe_threshold: float = 300.0,
    alert_threshold: float = 0.5,
) -> dict[str, Any]:
    quantile_array = np.asarray(quantiles, dtype=np.float32)
    median_index = quantiles.index(0.5)
    median = prediction[:, :, median_index]
    error = median - target
    pinball_values = []
    for index, quantile in enumerate(quantiles):
        residual = target - prediction[:, :, index]
        pinball_values.append(
            np.maximum(quantile * residual, (quantile - 1.0) * residual)
        )
    pinball_stack = np.stack(pinball_values, axis=-1)
    pinball_by_q = [_mean(pinball_stack[:, :, index]) for index in range(len(quantiles))]
    coverage: dict[str, float] = {}
    interval_width: dict[str, float] = {}
    coverage_error: dict[str, float] = {}
    q_index = {round(value, 4): index for index, value in enumerate(quantiles)}
    for level, lower, upper in ((50, 0.25, 0.75), (80, 0.1, 0.9), (90, 0.05, 0.95)):
        inside = (target >= prediction[:, :, q_index[lower]]) & (
            target <= prediction[:, :, q_index[upper]]
        )
        observed = _mean(inside)
        coverage[str(level)] = observed
        interval_width[str(level)] = _mean(
            prediction[:, :, q_index[upper]] - prediction[:, :, q_index[lower]]
        )
        coverage_error[str(level)] = abs(observed - level / 100.0)

    severe_probability = exceedance_probability(prediction, quantiles, severe_threshold)
    severe_label = (target >= severe_threshold).astype(np.int32)
    flat_probability = severe_probability.ravel()
    flat_label = severe_label.ravel()
    binary_prediction = flat_probability >= alert_threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_label, binary_prediction, average="binary", zero_division="warn"
    )
    ap = (
        average_precision_score(flat_label, flat_probability)
        if np.unique(flat_label).size > 1
        else float("nan")
    )
    brier = _mean((flat_probability - flat_label) ** 2)
    clipped = np.clip(flat_probability, 1e-6, 1.0 - 1e-6)
    severe_log_loss = _mean(
        -(flat_label * np.log(clipped) + (1 - flat_label) * np.log(1 - clipped))
    )
    crps = 2.0 * np.trapezoid(pinball_stack, quantile_array, axis=-1)

    return {
        "mae": _mean(np.abs(error)),
        "rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        "mase": _mean(np.abs(error)) / max(persistence_mae, 1e-6),
        "mae_by_horizon": [
            _mean(np.abs(error[:, horizon])) for horizon in range(error.shape[1])
        ],
        "pinball": _mean(pinball_stack),
        "pinball_by_quantile": dict(
            zip(
                [str(value) for value in quantile_array],
                pinball_by_q,
                strict=True,
            )
        ),
        "crps_quantile_approx": _mean(crps),
        "wis": weighted_interval_score(target, prediction, quantiles),
        "coverage": coverage,
        "coverage_error": coverage_error,
        "mean_absolute_coverage_error": _mean(
            np.asarray(list(coverage_error.values()))
        ),
        "interval_width": interval_width,
        "severe_brier": brier,
        "severe_log_loss": severe_log_loss,
        "severe_precision": float(precision),
        "severe_recall": float(recall),
        "severe_f1": float(f1),
        "severe_pr_auc": float(ap),
        "alert_threshold": float(alert_threshold),
        "alert_rate": _mean(binary_prediction),
    }


def point_metrics(
    target: np.ndarray, point_prediction: np.ndarray, persistence_mae: float
) -> dict[str, Any]:
    error = point_prediction - target
    mae = _mean(np.abs(error))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        "mase": mae / max(persistence_mae, 1e-6),
        "mae_by_horizon": [
            _mean(np.abs(error[:, index])) for index in range(error.shape[1])
        ],
    }


def regime_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    regime: np.ndarray,
    quantiles: list[float],
) -> dict[str, dict[str, float]]:
    names = {0: "normal", 1: "shock", 2: "recovery"}
    median = prediction[:, :, quantiles.index(0.5)]
    output: dict[str, dict[str, float]] = {}
    for value, name in names.items():
        selected = regime == value
        if not np.any(selected):
            continue
        output[name] = {
            "samples": int(np.sum(selected)),
            "mae": _mean(np.abs(median[selected] - target[selected])),
            "wis": weighted_interval_score(
                target[selected], prediction[selected], quantiles
            ),
        }
    return output


def routing_metrics(
    assignments: np.ndarray, regime: np.ndarray, num_experts: int
) -> dict[str, Any]:
    utilization = np.bincount(assignments, minlength=num_experts).astype(np.float64)
    utilization /= max(utilization.sum(), 1.0)
    by_regime: dict[str, list[float]] = {}
    for value, name in {0: "normal", 1: "shock", 2: "recovery"}.items():
        selected = assignments[regime == value]
        counts = np.bincount(selected, minlength=num_experts).astype(np.float64)
        counts /= max(counts.sum(), 1.0)
        by_regime[name] = counts.tolist()
    entropy = -float(
        np.sum(utilization * np.log(np.clip(utilization, 1e-12, 1.0)))
    )
    return {
        "utilization": utilization.tolist(),
        "utilization_by_regime": by_regime,
        "entropy": entropy,
    }


def paired_day_bootstrap(
    day_ordinal: np.ndarray,
    baseline_losses: np.ndarray,
    candidate_losses: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    unique_days = np.unique(day_ordinal)
    if unique_days.size == 0 or replicates < 1:
        raise ValueError("Bootstrap requires at least one day and one replicate")
    baseline_by_day = np.array(
        [np.mean(baseline_losses[day_ordinal == day]) for day in unique_days]
    )
    candidate_by_day = np.array(
        [np.mean(candidate_losses[day_ordinal == day]) for day in unique_days]
    )
    difference = candidate_by_day - baseline_by_day
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.choice(difference, size=len(difference), replace=True)
        bootstrap[index] = np.mean(sample)
    return {
        "candidate_minus_baseline_mean": float(np.mean(difference)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "days": int(len(unique_days)),
        "replicates": int(replicates),
    }



def point_alert_metrics(
    target: np.ndarray,
    point_prediction: np.ndarray,
    severe_threshold: float,
    validation_point: np.ndarray,
    validation_target: np.ndarray,
    alert_fraction: float,
) -> dict[str, float]:
    """Evaluate a point-only alert rule at a validation-frozen alert budget."""
    threshold = float(
        np.quantile(validation_point.ravel(), 1.0 - alert_fraction, method="higher")
    )
    labels = (target >= severe_threshold).astype(np.int32).ravel()
    binary = (point_prediction.ravel() >= threshold).astype(np.int32)
    probability = binary.astype(np.float32)
    return {
        "threshold_seconds": threshold,
        "alert_rate": float(np.mean(binary)),
        "brier": _mean((probability - labels) ** 2),
        "precision": float(
            precision_recall_fscore_support(
                labels, binary, average="binary", zero_division="warn"
            )[0]
        ),
        "recall": float(
            precision_recall_fscore_support(
                labels, binary, average="binary", zero_division="warn"
            )[1]
        ),
    }
