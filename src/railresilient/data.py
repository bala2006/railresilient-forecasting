"""RIDE data acquisition and causal event-window construction."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

RIDE_BASE_URL = "https://huggingface.co/datasets/orailix/ride-silver/resolve"
EVENT_COLUMNS = [
    "train_id",
    "service_date",
    "op_id",
    "event_type",
    "planned_ts",
    "observed_ts",
    "delay_sec",
]
EVENT_TYPE_MAP = {"A": 0, "D": 1, "P": 2}
REGIME_NAMES = {0: "normal", 1: "shock", 2: "recovery"}

#: Bounded feed-quality channels. Every channel is constrained to [0, 1] so that
#: unseen corruption severities saturate instead of extrapolating, which was the
#: dominant failure mode of the first pilot.
QUALITY_FIELDS = (
    "missing_fraction",
    "stale_fraction",
    "observation_age_saturating",
    "declared_delay_saturating",
    "inconsistent_flag",
    "duplicate_flag",
    "no_fresh_observation",
)
QUALITY_DIM = len(QUALITY_FIELDS)


@dataclass(slots=True)
class ArrayDataset:
    past_delay: np.ndarray
    past_planned_delta: np.ndarray
    past_observed_delta: np.ndarray
    past_event_type: np.ndarray
    past_station: np.ndarray
    observation_mask: np.ndarray
    future_planned_delta: np.ndarray
    future_observed_ns: np.ndarray
    target_delay: np.ndarray
    forecast_origin_ns: np.ndarray
    day_ordinal: np.ndarray
    regime: np.ndarray
    current_delay: np.ndarray

    def __len__(self) -> int:
        return int(self.target_delay.shape[0])

    def select(self, indices: np.ndarray) -> ArrayDataset:
        return ArrayDataset(
            **{name: getattr(self, name)[indices] for name in self.__dataclass_fields__}
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output, **{name: getattr(self, name) for name in self.__dataclass_fields__}
        )

    @classmethod
    def load(cls, path: str | Path) -> ArrayDataset:
        with np.load(path) as arrays:
            missing = set(cls.__dataclass_fields__).difference(arrays.files)
            if missing:
                raise ValueError(
                    f"Prepared dataset uses an obsolete schema; missing {sorted(missing)}. "
                    "Run prepare again."
                )
            return cls(**{name: arrays[name] for name in cls.__dataclass_fields__})


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def download_ride_months(
    config: dict[str, Any], data_root: str | Path = "data"
) -> dict[str, Any]:
    data_config = config["data"]
    revision = data_config["release_revision"]
    months = sorted(
        set(data_config["train_months"])
        | set(data_config["validation_months"])
        | set(data_config["test_months"])
    )
    root = Path(data_root)
    raw_root = root / "raw" / "ride-silver" / "events"
    files: list[dict[str, Any]] = []
    for month in months:
        filename = f"events_{month}.parquet"
        destination = raw_root / filename
        url = f"{RIDE_BASE_URL}/{revision}/events/{filename}"
        if not destination.exists() or destination.stat().st_size == 0:
            _download(url, destination)
        files.append(
            {
                "month": month,
                "path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "url": url,
            }
        )
    manifest = {
        "dataset": data_config["source"],
        "revision": revision,
        "license": "CC-BY-4.0",
        "attribution": ["RIDE", "Infrabel", "Open-Meteo"],
        "verification": "SHA-256 is recorded for local reproducibility; upstream trusted hashes were unavailable.",
        "files": files,
    }
    manifest_path = root / "manifests" / "ride_download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _regime(
    past_delay: np.ndarray,
    shock_threshold: float,
    recovery_threshold: float,
    severe_threshold: float,
) -> int:
    current = float(past_delay[-1])
    recent_width = min(3, past_delay.size - 1)
    trend = current - float(past_delay[-1 - recent_width])
    if current >= severe_threshold or trend >= shock_threshold:
        return 1
    if current > 60.0 and trend <= recovery_threshold:
        return 2
    return 0


def _candidate_windows(
    frame: pd.DataFrame,
    context: int,
    horizon: int,
    station_buckets: int,
    rng: np.random.Generator,
    shock_threshold: float,
    recovery_threshold: float,
    severe_threshold: float,
) -> Iterator[dict[str, np.ndarray | int | float]]:
    frame = frame.dropna(subset=["planned_ts", "observed_ts", "delay_sec"])
    frame = frame.sort_values(
        ["service_date", "train_id", "planned_ts", "observed_ts"], kind="stable"
    )
    for _, journey in frame.groupby(
        ["service_date", "train_id"], sort=False, observed=True
    ):
        if len(journey) < context + horizon:
            continue
        observed_ns = np.fromiter(
            (pd.Timestamp(value).value for value in journey["observed_ts"]),
            dtype=np.int64,
            count=len(journey),
        )
        anchors = np.arange(context - 1, len(journey) - horizon, dtype=np.int64)
        past_windows = np.lib.stride_tricks.sliding_window_view(observed_ns, context)
        future_windows = np.lib.stride_tricks.sliding_window_view(observed_ns, horizon)
        past_max = past_windows[: len(anchors)].max(axis=1)
        future_min = future_windows[anchors + 1].min(axis=1)
        valid_anchors = anchors[
            (past_max <= observed_ns[anchors])
            & (future_min > observed_ns[anchors])
        ]
        if valid_anchors.size == 0:
            continue
        anchor = int(valid_anchors[int(rng.integers(0, len(valid_anchors)))])
        origin = pd.Timestamp(journey["observed_ts"].iloc[anchor])
        past = journey.iloc[anchor - context + 1 : anchor + 1]
        future = journey.iloc[anchor + 1 : anchor + 1 + horizon]
        planned_origin = pd.Timestamp(past["planned_ts"].iloc[-1])
        past_delay = past["delay_sec"].to_numpy(dtype=np.float32)
        event_type = np.array(
            [EVENT_TYPE_MAP.get(str(value), 2) for value in past["event_type"]],
            dtype=np.int64,
        )
        stations = np.mod(
            past["op_id"].fillna(0).to_numpy(dtype=np.int64), station_buckets
        )
        past_planned = (
            (pd.to_datetime(past["planned_ts"]) - planned_origin)
            .dt.total_seconds()
            .to_numpy(dtype=np.float32)
        )
        past_observed = (
            (pd.to_datetime(past["observed_ts"]) - origin)
            .dt.total_seconds()
            .to_numpy(dtype=np.float32)
        )
        future_planned = (
            (pd.to_datetime(future["planned_ts"]) - planned_origin)
            .dt.total_seconds()
            .to_numpy(dtype=np.float32)
        )
        future_observed_ns = np.asarray(
            [pd.Timestamp(value).value for value in future["observed_ts"]],
            dtype=np.int64,
        )
        target = future["delay_sec"].to_numpy(dtype=np.float32)
        origin_ns = int(origin.value)
        if np.any(past_observed > 0.0) or np.any(future_observed_ns <= origin_ns):
            raise AssertionError("Causal event-window invariant failed")
        yield {
            "past_delay": past_delay,
            "past_planned_delta": past_planned,
            "past_observed_delta": past_observed,
            "past_event_type": event_type,
            "past_station": stations,
            "observation_mask": np.ones(context, dtype=np.float32),
            "future_planned_delta": future_planned,
            "future_observed_ns": future_observed_ns.astype(np.int64),
            "target_delay": target,
            "forecast_origin_ns": origin_ns,
            "day_ordinal": int(origin_ns // 86_400_000_000_000 + 719_163),
            "regime": _regime(
                past_delay, shock_threshold, recovery_threshold, severe_threshold
            ),
            "current_delay": float(past_delay[-1]),
        }


def _stack(records: list[dict[str, Any]]) -> ArrayDataset:
    if not records:
        raise ValueError("No valid event windows were produced")
    return ArrayDataset(
        past_delay=np.stack([record["past_delay"] for record in records]).astype(
            np.float32
        ),
        past_planned_delta=np.stack(
            [record["past_planned_delta"] for record in records]
        ).astype(np.float32),
        past_observed_delta=np.stack(
            [record["past_observed_delta"] for record in records]
        ).astype(np.float32),
        past_event_type=np.stack(
            [record["past_event_type"] for record in records]
        ).astype(np.int64),
        past_station=np.stack([record["past_station"] for record in records]).astype(
            np.int64
        ),
        observation_mask=np.stack(
            [record["observation_mask"] for record in records]
        ).astype(np.float32),
        future_planned_delta=np.stack(
            [record["future_planned_delta"] for record in records]
        ).astype(np.float32),
        future_observed_ns=np.stack(
            [record["future_observed_ns"] for record in records]
        ).astype(np.int64),
        target_delay=np.stack([record["target_delay"] for record in records]).astype(
            np.float32
        ),
        forecast_origin_ns=np.asarray(
            [record["forecast_origin_ns"] for record in records], dtype=np.int64
        ),
        day_ordinal=np.asarray(
            [record["day_ordinal"] for record in records], dtype=np.int32
        ),
        regime=np.asarray([record["regime"] for record in records], dtype=np.int64),
        current_delay=np.asarray(
            [record["current_delay"] for record in records], dtype=np.float32
        ),
    )


def _stratified_limit(
    dataset: ArrayDataset,
    limit: int,
    seed: int,
    severe_threshold: float,
    target_aware: bool,
) -> ArrayDataset:
    if limit < 1:
        raise ValueError("Sample limit must be positive")
    if len(dataset) <= limit:
        return dataset.select(np.argsort(dataset.forecast_origin_ns, kind="stable"))
    rng = np.random.default_rng(seed)
    if not target_aware:
        selected = rng.choice(len(dataset), size=limit, replace=False)
    else:
        severe = np.max(dataset.target_delay, axis=1) >= severe_threshold
        priority = np.flatnonzero(severe | (dataset.regime == 2))
        ordinary = np.flatnonzero(~(severe | (dataset.regime == 2)))
        priority_budget = min(
            len(priority), max(limit // 3, int(round(0.15 * limit)))
        )
        ordinary_budget = min(len(ordinary), limit - priority_budget)
        if priority_budget + ordinary_budget < limit:
            priority_budget = min(len(priority), limit - ordinary_budget)
        if priority_budget + ordinary_budget < limit:
            ordinary_budget = min(len(ordinary), limit - priority_budget)
        chosen_priority = (
            rng.choice(priority, size=priority_budget, replace=False)
            if priority_budget
            else np.empty(0, dtype=np.int64)
        )
        chosen_ordinary = (
            rng.choice(ordinary, size=ordinary_budget, replace=False)
            if ordinary_budget
            else np.empty(0, dtype=np.int64)
        )
        selected = np.concatenate([chosen_priority, chosen_ordinary])
        if len(selected) != limit:
            raise AssertionError("Stratified sample allocation failed")
    order = selected[np.argsort(dataset.forecast_origin_ns[selected], kind="stable")]
    return dataset.select(order)


def build_split(
    month_files: list[Path],
    config: dict[str, Any],
    limit: int,
    seed_offset: int = 0,
    target_aware_sampling: bool = False,
) -> ArrayDataset:
    data_config = config["data"]
    rng = np.random.default_rng(config["seed"] + seed_offset)
    records: list[dict[str, Any]] = []
    for path in month_files:
        frame = pd.read_parquet(path, columns=EVENT_COLUMNS)
        records.extend(
            _candidate_windows(
                frame,
                context=data_config["context_events"],
                horizon=data_config["future_events"],
                station_buckets=data_config["station_buckets"],
                rng=rng,
                shock_threshold=data_config["shock_trend_seconds"],
                recovery_threshold=data_config["recovery_trend_seconds"],
                severe_threshold=data_config["severe_delay_seconds"],
            )
        )
    return _stratified_limit(
        _stack(records),
        limit=limit,
        seed=config["seed"] + seed_offset,
        severe_threshold=data_config["severe_delay_seconds"],
        target_aware=target_aware_sampling,
    )


def prepare_datasets(
    config: dict[str, Any], data_root: str | Path = "data"
) -> dict[str, Any]:
    root = Path(data_root)
    raw_root = root / "raw" / "ride-silver" / "events"
    processed_root = root / "processed" / "ride-silver-pilot"
    processed_root.mkdir(parents=True, exist_ok=True)
    data_config = config["data"]
    split_definitions = {
        "train": (
            data_config["train_months"],
            data_config["max_train_samples"],
            11,
            True,
        ),
        "validation": (
            data_config["validation_months"],
            data_config["max_validation_samples"],
            23,
            False,
        ),
        "test": (
            data_config["test_months"],
            data_config["max_test_samples"],
            37,
            False,
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": "1.1.0",
        "context_events": data_config["context_events"],
        "future_events": data_config["future_events"],
        "causal_contract": "All context observed timestamps are <= origin; every target matures strictly after origin.",
        "sampling": {
            "candidate_unit": "one uniformly selected valid anchor per eligible (service_date, train_id) journey",
            "train": "target-aware severe/recovery enrichment when capped",
            "validation": "uniform outcome-blind sample when capped",
            "test": "uniform outcome-blind sample when capped",
        },
        "splits": {},
    }
    split_ranges: dict[str, tuple[int, int]] = {}
    for name, (months, limit, offset, target_aware) in split_definitions.items():
        files = [raw_root / f"events_{month}.parquet" for month in months]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Run download first; missing files: {missing}")
        dataset = build_split(
            files,
            config,
            limit=limit,
            seed_offset=offset,
            target_aware_sampling=target_aware,
        )
        output = processed_root / f"{name}.npz"
        dataset.save(output)
        minimum = int(dataset.forecast_origin_ns.min())
        maximum = int(dataset.forecast_origin_ns.max())
        split_ranges[name] = (minimum, maximum)
        summary["splits"][name] = {
            "samples": len(dataset),
            "path": str(output.resolve()),
            "forecast_origin_min_ns": minimum,
            "forecast_origin_max_ns": maximum,
            "severe_fraction": float(
                np.mean(
                    np.max(dataset.target_delay, axis=1)
                    >= data_config["severe_delay_seconds"]
                )
            ),
            "regime_counts": {
                REGIME_NAMES[index]: int(np.sum(dataset.regime == index))
                for index in REGIME_NAMES
            },
        }
    if not (
        split_ranges["train"][1] < split_ranges["validation"][0]
        and split_ranges["validation"][1] < split_ranges["test"][0]
    ):
        raise ValueError("Chronological split invariant failed")
    manifest_path = root / "manifests" / "prepared_dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def estimate_download_gib(config: dict[str, Any]) -> float:
    months = len(
        set(config["data"]["train_months"])
        | set(config["data"]["validation_months"])
        | set(config["data"]["test_months"])
    )
    return math.ceil(months * 0.033 * 100) / 100
