"""Deterministic causal feed-corruption simulation with bounded quality channels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from railresilient.data import QUALITY_DIM, QUALITY_FIELDS, ArrayDataset

#: Saturation constants for the bounded quality transforms, in seconds and minutes.
AGE_SCALE_SECONDS = 600.0
DECLARED_DELAY_SCALE_MINUTES = 10.0


@dataclass(slots=True)
class CorruptionSpec:
    """Per-sample corruption severities. All rates are probabilities in [0, 1]."""

    loss_rate: float = 0.0
    stale_minutes: float = 0.0
    outage_minutes: float = 0.0
    outage_rate: float = 0.0
    inconsistent_rate: float = 0.0
    duplicate_rate: float = 0.0
    blackout_rate: float = 0.0

    def validate(self) -> None:
        for name in ("loss_rate", "outage_rate", "inconsistent_rate", "duplicate_rate", "blackout_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"CorruptionSpec.{name} must lie in [0, 1]")
        for name in ("stale_minutes", "outage_minutes"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"CorruptionSpec.{name} must be non-negative")


@dataclass(slots=True)
class CorruptedDataset:
    data: ArrayDataset
    quality: np.ndarray
    scenario_name: str
    manifest: dict[str, Any]


def spec_from_scenario(scenario: dict[str, Any]) -> CorruptionSpec:
    """Translate a declared evaluation scenario into explicit severities."""
    scenario_type = str(scenario["type"])
    if scenario_type == "clean":
        spec = CorruptionSpec()
    elif scenario_type == "packet_loss":
        spec = CorruptionSpec(loss_rate=float(scenario["rate"]))
    elif scenario_type == "stale":
        spec = CorruptionSpec(stale_minutes=float(scenario["minutes"]))
    elif scenario_type == "outage":
        spec = CorruptionSpec(
            outage_minutes=float(scenario["minutes"]),
            outage_rate=float(scenario.get("rate", 0.2)),
        )
    elif scenario_type == "combined":
        spec = CorruptionSpec(
            loss_rate=float(scenario.get("rate", 0.15)),
            outage_minutes=float(scenario.get("minutes", 10.0)),
            outage_rate=float(scenario.get("outage_rate", 0.2)),
            inconsistent_rate=float(scenario.get("inconsistent_rate", 0.1)),
            duplicate_rate=float(scenario.get("duplicate_rate", 0.05)),
        )
    else:
        raise ValueError(f"Unsupported corruption scenario: {scenario_type}")
    spec.validate()
    return spec


def _causal_fill(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked observations with the most recent causally available value."""
    valid = mask > 0.5
    index = np.where(valid, np.arange(values.shape[1])[None, :], -1)
    source = np.maximum.accumulate(index, axis=1)
    gathered = np.take_along_axis(values, np.maximum(source, 0), axis=1)
    return np.where(source >= 0, gathered, 0.0).astype(np.float32)


def _stale_mask(
    observed_delta: np.ndarray, minutes: np.ndarray, affected: np.ndarray
) -> np.ndarray:
    """Hide every observation newer than the per-sample feed cutoff."""
    mask = np.ones_like(observed_delta, dtype=np.float32)
    cutoff = -np.asarray(minutes, dtype=np.float32) * 60.0
    recent = observed_delta > cutoff[:, None]
    return np.where(affected[:, None] & recent, 0.0, mask)


def _bounded_quality(
    mask: np.ndarray,
    observed_delta: np.ndarray,
    declared_minutes: np.ndarray,
    inconsistent: np.ndarray,
    duplicate: np.ndarray,
) -> np.ndarray:
    """Build the bounded [0, 1] quality matrix, saturating unseen severities."""
    n_samples, context = mask.shape
    valid = mask > 0.5
    has_valid = np.any(valid, axis=1)
    missing_fraction = 1.0 - mask.mean(axis=1)
    last_valid = np.max(np.where(valid, np.arange(context)[None, :], -1), axis=1)
    age_seconds = np.zeros(n_samples, dtype=np.float32)
    rows = np.flatnonzero(has_valid)
    age_seconds[rows] = np.maximum(-observed_delta[rows, last_valid[rows]], 0.0)
    age_saturating = np.where(
        has_valid, 1.0 - np.exp(-age_seconds / AGE_SCALE_SECONDS), 1.0
    )
    declared_saturating = 1.0 - np.exp(
        -np.maximum(declared_minutes, 0.0) / DECLARED_DELAY_SCALE_MINUTES
    )
    quality = np.column_stack(
        [
            missing_fraction,
            np.maximum(missing_fraction, np.where(has_valid, age_saturating, 1.0)),
            age_saturating,
            declared_saturating,
            inconsistent,
            duplicate,
            (~has_valid).astype(np.float32),
        ]
    ).astype(np.float32)
    if quality.shape[1] != QUALITY_DIM:
        raise AssertionError("Quality matrix width does not match the declared schema")
    return np.clip(quality, 0.0, 1.0)


def apply_spec(
    dataset: ArrayDataset,
    spec: CorruptionSpec,
    seed: int,
    scenario_name: str = "custom",
    randomize_per_sample: bool = False,
) -> CorruptedDataset:
    """Apply a corruption specification causally to the observed context.

    When ``randomize_per_sample`` is set, each sample draws its own severities
    uniformly between zero and the specification's values. That yields the
    training curriculum; evaluation scenarios always use fixed severities.
    """
    spec.validate()
    rng = np.random.default_rng(seed)
    n_samples, context = dataset.observation_mask.shape

    def per_sample(value: float) -> np.ndarray:
        magnitude = float(value)
        if not randomize_per_sample or magnitude == 0.0:
            return np.full(n_samples, magnitude, dtype=np.float32)
        return rng.uniform(0.0, magnitude, size=n_samples).astype(np.float32)

    loss_rate = per_sample(spec.loss_rate)
    stale_minutes = per_sample(spec.stale_minutes)
    outage_minutes = per_sample(spec.outage_minutes)
    outage_rate = per_sample(spec.outage_rate)
    inconsistent_rate = per_sample(spec.inconsistent_rate)
    duplicate_rate = per_sample(spec.duplicate_rate)
    blackout_rate = per_sample(spec.blackout_rate)

    mask = (rng.random((n_samples, context)) >= loss_rate[:, None]).astype(np.float32)
    everyone = np.ones(n_samples, dtype=bool)
    mask *= _stale_mask(dataset.past_observed_delta, stale_minutes, everyone)
    outage_affected = rng.random(n_samples) < outage_rate
    mask *= _stale_mask(dataset.past_observed_delta, outage_minutes, outage_affected)
    blackout = rng.random(n_samples) < blackout_rate
    mask[blackout] = 0.0

    declared_minutes = np.maximum(
        np.where(outage_affected, outage_minutes, 0.0), stale_minutes
    ).astype(np.float32)
    declared_minutes = np.where(blackout, np.maximum(declared_minutes, 15.0), declared_minutes)
    inconsistent = (rng.random(n_samples) < inconsistent_rate).astype(np.float32)
    duplicate = (rng.random(n_samples) < duplicate_rate).astype(np.float32)

    delay = dataset.past_delay.copy()
    event_type = dataset.past_event_type.copy()
    station = dataset.past_station.copy()
    observed_delta = dataset.past_observed_delta.copy()

    inconsistent_rows = np.flatnonzero(inconsistent > 0.5)
    if inconsistent_rows.size:
        columns = rng.integers(0, context, size=inconsistent_rows.size)
        signs = rng.choice(np.array([-1.0, 1.0]), size=inconsistent_rows.size)
        delay[inconsistent_rows, columns] += (signs * 120.0).astype(np.float32)
    duplicate_rows = np.flatnonzero(duplicate > 0.5)
    if duplicate_rows.size and context > 1:
        for array in (delay, event_type, station, observed_delta, mask):
            array[duplicate_rows, -1] = array[duplicate_rows, -2]

    filled_delay = _causal_fill(delay, mask)
    quality = _bounded_quality(
        mask, observed_delta, declared_minutes, inconsistent, duplicate
    )
    corrupted = ArrayDataset(
        past_delay=filled_delay,
        past_planned_delta=dataset.past_planned_delta.copy(),
        past_observed_delta=observed_delta.astype(np.float32),
        past_event_type=event_type.astype(np.int64),
        past_station=station.astype(np.int64),
        observation_mask=mask.astype(np.float32),
        future_planned_delta=dataset.future_planned_delta.copy(),
        future_observed_ns=dataset.future_observed_ns.copy(),
        target_delay=dataset.target_delay.copy(),
        forecast_origin_ns=dataset.forecast_origin_ns.copy(),
        day_ordinal=dataset.day_ordinal.copy(),
        regime=dataset.regime.copy(),
        current_delay=filled_delay[:, -1].astype(np.float32),
    )
    manifest = {
        "scenario": scenario_name,
        "spec": asdict(spec),
        "randomized_per_sample": randomize_per_sample,
        "seed": seed,
        "samples": n_samples,
        "context_events": context,
        "quality_fields": list(QUALITY_FIELDS),
        "quality_bounds": "every channel is clipped to [0, 1]",
        "masked_observations": int(np.sum(mask < 0.5)),
        "empty_context_samples": int(np.sum(quality[:, 6] > 0.5)),
        "inconsistent_samples": int(np.sum(inconsistent)),
        "duplicate_samples": int(np.sum(duplicate)),
        "affected_samples": int(np.sum(quality[:, 0] > 0.0)),
        "mean_missing_fraction": float(np.mean(quality[:, 0])),
        "quality_max": quality.max(axis=0).tolist(),
        "timestamp_basis": "observed event time relative to forecast origin",
    }
    return CorruptedDataset(
        data=corrupted, quality=quality, scenario_name=scenario_name, manifest=manifest
    )


def apply_corruption(
    dataset: ArrayDataset, scenario: dict[str, Any], seed: int
) -> CorruptedDataset:
    """Apply a declared evaluation scenario at its exact fixed severity."""
    return apply_spec(
        dataset,
        spec_from_scenario(scenario),
        seed,
        scenario_name=str(scenario["name"]),
        randomize_per_sample=False,
    )


def training_spec(config: dict[str, Any]) -> CorruptionSpec:
    """Upper severity bounds for the randomized training curriculum."""
    curriculum = config["training_corruption"]
    spec = CorruptionSpec(
        loss_rate=float(curriculum["max_loss_rate"]),
        stale_minutes=float(curriculum["max_stale_minutes"]),
        outage_minutes=float(curriculum["max_outage_minutes"]),
        outage_rate=float(curriculum["max_outage_rate"]),
        inconsistent_rate=float(curriculum["max_inconsistent_rate"]),
        duplicate_rate=float(curriculum["max_duplicate_rate"]),
        blackout_rate=float(curriculum["max_blackout_rate"]),
    )
    spec.validate()
    return spec


def write_corruption_manifest(result: CorruptedDataset, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.manifest, indent=2) + "\n", encoding="utf-8")
