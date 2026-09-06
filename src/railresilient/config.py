"""Configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or unsafe."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def _require(section: dict[str, Any], name: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(f"Missing {name} keys: {missing}")


def validate_config(config: dict[str, Any]) -> None:
    required = {"seed", "data", "model", "calibration", "evaluation", "scenarios"}
    missing = required.difference(config)
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")
    if not isinstance(config["seed"], int):
        raise ConfigError("seed must be an integer")

    data = config["data"]
    _require(
        data,
        "data",
        (
            "source",
            "release_revision",
            "context_events",
            "future_events",
            "station_buckets",
            "train_months",
            "validation_months",
            "test_months",
            "max_train_samples",
            "max_validation_samples",
            "max_test_samples",
            "severe_delay_seconds",
            "shock_trend_seconds",
            "recovery_trend_seconds",
        ),
    )
    if data["source"] != "orailix/ride-silver":
        raise ConfigError("The implemented pilot accepts only the audited RIDE Silver release")
    if data["context_events"] < 2 or data["future_events"] < 1:
        raise ConfigError("At least two context events and one future event are required")
    if data["station_buckets"] < 2:
        raise ConfigError("station_buckets must be at least two")
    if any(data[key] < 1 for key in ("max_train_samples", "max_validation_samples", "max_test_samples")):
        raise ConfigError("All sample limits must be positive")
    split_months = [
        set(data["train_months"]),
        set(data["validation_months"]),
        set(data["test_months"]),
    ]
    if any(not months for months in split_months):
        raise ConfigError("Every chronological split must contain at least one month")
    if any(
        a.intersection(b)
        for index, a in enumerate(split_months)
        for b in split_months[index + 1 :]
    ):
        raise ConfigError("Train, validation, and test months must be disjoint")
    if not (
        max(split_months[0]) < min(split_months[1])
        and max(split_months[1]) < min(split_months[2])
    ):
        raise ConfigError("Split months must be strictly chronological")

    model = config["model"]
    _require(
        model,
        "model",
        (
            "hidden_dim",
            "station_embedding_dim",
            "event_embedding_dim",
            "expert_multiplier",
            "num_experts",
            "quantiles",
            "dropout",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "patience",
            "balance_weight",
            "huber_weight",
        ),
    )
    expected_quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    if model["quantiles"] != expected_quantiles:
        raise ConfigError(f"model.quantiles must equal {expected_quantiles}")
    if any(model[key] < 1 for key in ("hidden_dim", "num_experts", "epochs", "batch_size")):
        raise ConfigError("Model dimensions, epochs, experts, and batch size must be positive")
    if not 0.0 <= model["dropout"] < 1.0 or model["learning_rate"] <= 0.0:
        raise ConfigError("dropout or learning_rate is outside its valid range")
    if model.get("dense_trunk_width", 0) < 1:
        raise ConfigError("model.dense_trunk_width must be positive")

    calibration = config["calibration"]
    _require(calibration, "calibration", ("window", "min_samples", "online_update_stride"))
    if calibration["window"] < calibration["min_samples"] or calibration["min_samples"] < 1:
        raise ConfigError("Calibration requires window >= min_samples >= 1")
    if calibration["online_update_stride"] < 1:
        raise ConfigError("calibration.online_update_stride must be positive")

    evaluation = config["evaluation"]
    _require(
        evaluation,
        "evaluation",
        (
            "inference_batch_size",
            "warmup_batches",
            "timing_batches",
            "alert_fraction",
            "bootstrap_replicates",
            "transfer_buffer_seconds",
        ),
    )
    if any(evaluation[key] < 1 for key in ("inference_batch_size", "timing_batches", "bootstrap_replicates")):
        raise ConfigError("Inference, timing, and bootstrap counts must be positive")
    if evaluation["warmup_batches"] < 0 or not 0.0 < evaluation["alert_fraction"] < 1.0:
        raise ConfigError("Invalid warmup_batches or alert_fraction")
    if evaluation["transfer_buffer_seconds"] < 0.0:
        raise ConfigError("evaluation.transfer_buffer_seconds must be non-negative")

    scenario_names = [scenario.get("name") for scenario in config["scenarios"]]
    if len(scenario_names) != len(set(scenario_names)):
        raise ConfigError("Scenario names must be unique")
    scenario_types = [scenario.get("type") for scenario in config["scenarios"]]
    if scenario_types.count("clean") != 1:
        raise ConfigError("Exactly one clean scenario is mandatory")
    if "packet_loss" not in scenario_types or "stale" not in scenario_types:
        raise ConfigError("Training augmentation requires packet_loss and stale scenarios")
    valid_types = {"clean", "packet_loss", "stale", "outage", "combined"}
    unknown = set(scenario_types).difference(valid_types)
    if unknown:
        raise ConfigError(f"Unsupported scenario types: {sorted(unknown)}")
    for scenario in config["scenarios"]:
        if "rate" in scenario and not 0.0 <= float(scenario["rate"]) <= 1.0:
            raise ConfigError(f"Invalid scenario rate in {scenario.get('name')}")
        if "minutes" in scenario and float(scenario["minutes"]) < 0.0:
            raise ConfigError(f"Invalid scenario minutes in {scenario.get('name')}")

    curriculum = config.get("training_corruption")
    if not isinstance(curriculum, dict):
        raise ConfigError("training_corruption section is required")
    curriculum_keys = (
        "max_loss_rate",
        "max_stale_minutes",
        "max_outage_minutes",
        "max_outage_rate",
        "max_inconsistent_rate",
        "max_duplicate_rate",
        "max_blackout_rate",
    )
    _require(curriculum, "training_corruption", curriculum_keys)
    for key in curriculum_keys:
        value = float(curriculum[key])
        if value < 0.0 or ("rate" in key and value > 1.0):
            raise ConfigError(f"Invalid training_corruption.{key}")


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
