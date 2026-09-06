"""Static and state-stratified causal online quantile calibration."""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CalibrationResult:
    predictions: np.ndarray
    updates: int
    fallback_predictions: int
    update_p50_ms: float
    update_p95_ms: float


class StaticQuantileCalibrator:
    def __init__(self, quantiles: list[float]) -> None:
        self.quantiles = np.asarray(quantiles, dtype=np.float32)
        self.corrections: np.ndarray | None = None

    def fit(self, predictions: np.ndarray, target: np.ndarray) -> None:
        horizons = target.shape[1]
        corrections = np.zeros((horizons, len(self.quantiles)), dtype=np.float32)
        for horizon in range(horizons):
            for q_index, quantile in enumerate(self.quantiles):
                residual = target[:, horizon] - predictions[:, horizon, q_index]
                corrections[horizon, q_index] = np.quantile(residual, quantile)
        self.corrections = corrections

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        if self.corrections is None:
            raise RuntimeError("Calibrator must be fitted before transform")
        adjusted = predictions + self.corrections[None, :, :]
        return np.sort(adjusted, axis=2).astype(np.float32)


class OnlineQuantileCalibrator:
    """Forward-only, per-horizon and observable-feed-state calibration.

    State keys are intentionally coarse and observable: 0 is a fresh/ordinary
    feed, 1 is a partially missing feed, and 2 is a no-fresh-observation feed.
    Residuals from catastrophic empty-context cases therefore cannot contaminate
    the ordinary-feed buffer.
    """

    def __init__(
        self,
        quantiles: list[float],
        window: int = 1000,
        min_samples: int = 100,
        update_stride: int = 1,
        num_states: int = 3,
    ) -> None:
        self.quantiles = np.asarray(quantiles, dtype=np.float32)
        self.window = window
        self.min_samples = min_samples
        self.update_stride = update_stride
        self.num_states = num_states

    def transform(
        self,
        predictions: np.ndarray,
        target: np.ndarray,
        origin_ns: np.ndarray,
        target_maturity_ns: np.ndarray,
        state_key: np.ndarray | None = None,
        initial_predictions: np.ndarray | None = None,
        initial_target: np.ndarray | None = None,
        initial_state_key: np.ndarray | None = None,
    ) -> CalibrationResult:
        if predictions.shape[:2] != target.shape:
            raise ValueError("Prediction and target shapes do not agree")
        if target_maturity_ns.shape != target.shape:
            raise ValueError("Target maturity timestamps must have shape [samples, horizons]")
        if np.any(target_maturity_ns <= origin_ns[:, None]):
            raise ValueError("Every target must mature strictly after its forecast origin")
        state = np.zeros(len(target), dtype=np.int64) if state_key is None else np.asarray(state_key, dtype=np.int64)
        if state.shape != (len(target),) or np.any((state < 0) | (state >= self.num_states)):
            raise ValueError("state_key must be a valid [samples] array")

        order = np.argsort(origin_ns, kind="stable")
        result = predictions.copy().astype(np.float32)
        horizons, quantile_count = predictions.shape[1], predictions.shape[2]
        buffers = [
            [
                [deque(maxlen=self.window) for _ in range(quantile_count)]
                for _ in range(self.num_states)
            ]
            for _ in range(horizons)
        ]
        if initial_predictions is not None and initial_target is not None:
            initial_state = (
                np.zeros(len(initial_target), dtype=np.int64)
                if initial_state_key is None
                else np.asarray(initial_state_key, dtype=np.int64)
            )
            for horizon in range(horizons):
                for sample in range(len(initial_target)):
                    state_index = int(initial_state[sample])
                    for q_index in range(quantile_count):
                        residual = initial_target[sample, horizon] - initial_predictions[sample, horizon, q_index]
                        buffers[horizon][state_index][q_index].append(float(residual))

        corrections = np.zeros((horizons, self.num_states, quantile_count), dtype=np.float32)
        correction_ready = np.zeros((horizons, self.num_states), dtype=bool)
        dirty_updates = np.full((horizons, self.num_states), self.update_stride, dtype=np.int64)
        update_times_ms: list[float] = []

        def refresh(horizon: int, state_index: int) -> None:
            values = buffers[horizon][state_index]
            if len(values[0]) < self.min_samples:
                correction_ready[horizon, state_index] = False
                return
            started = time.perf_counter_ns()
            for q_index, quantile in enumerate(self.quantiles):
                corrections[horizon, state_index, q_index] = np.quantile(
                    np.asarray(values[q_index], dtype=np.float32), quantile
                )
            update_times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
            correction_ready[horizon, state_index] = True
            dirty_updates[horizon, state_index] = 0

        for horizon in range(horizons):
            for state_index in range(self.num_states):
                refresh(horizon, state_index)

        pending: list[tuple[int, int, int, int, np.ndarray]] = []
        updates = 0
        fallback = 0
        cursor = 0
        while cursor < len(order):
            current_origin = int(origin_ns[order[cursor]])
            while pending and pending[0][0] <= current_origin:
                _, _, horizon, state_index, residual_vector = heapq.heappop(pending)
                for q_index in range(quantile_count):
                    buffers[horizon][state_index][q_index].append(float(residual_vector[q_index]))
                dirty_updates[horizon, state_index] += 1
                updates += 1
            for horizon in range(horizons):
                for state_index in range(self.num_states):
                    if dirty_updates[horizon, state_index] >= self.update_stride:
                        refresh(horizon, state_index)

            end = cursor + 1
            while end < len(order) and int(origin_ns[order[end]]) == current_origin:
                end += 1
            group = order[cursor:end]
            for sample_index in group:
                state_index = int(state[sample_index])
                for horizon in range(horizons):
                    if correction_ready[horizon, state_index]:
                        result[sample_index, horizon] += corrections[horizon, state_index]
                    elif state_index != 0 and correction_ready[horizon, 0]:
                        # Cold-start fallback uses only ordinary-feed calibration.
                        result[sample_index, horizon] += corrections[horizon, 0]
                        fallback += 1
                    else:
                        fallback += 1
                    residual_vector = target[sample_index, horizon] - predictions[sample_index, horizon]
                    heapq.heappush(
                        pending,
                        (
                            int(target_maturity_ns[sample_index, horizon]),
                            int(sample_index),
                            horizon,
                            state_index,
                            residual_vector,
                        ),
                    )
                result[sample_index] = np.sort(result[sample_index], axis=1)
            cursor = end
        return CalibrationResult(
            predictions=result,
            updates=updates,
            fallback_predictions=fallback,
            update_p50_ms=float(np.quantile(update_times_ms, 0.5)) if update_times_ms else 0.0,
            update_p95_ms=float(np.quantile(update_times_ms, 0.95)) if update_times_ms else 0.0,
        )
