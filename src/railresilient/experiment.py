"""End-to-end experiment orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn

from railresilient.calibration import OnlineQuantileCalibrator, StaticQuantileCalibrator
from railresilient.config import save_resolved_config
from railresilient.corruption import (
    apply_corruption,
    apply_spec,
    training_spec,
    write_corruption_manifest,
)
from railresilient.data import ArrayDataset
from railresilient.metrics import (
    paired_day_bootstrap,
    point_alert_metrics,
    probabilistic_metrics,
    regime_metrics,
    routing_metrics,
    select_alert_threshold,
    weighted_interval_score_values,
)
from railresilient.models import (
    DenseForecaster,
    GRUForecaster,
    Normalization,
    QuantileRidge,
    R2SMoE,
    R3SMoE,
    balance_loss,
    count_parameters,
    quantile_loss,
    torch_batch,
)
from railresilient.risk import connection_miss_probability


def _stable_offset(name: str, modulus: int = 100_000) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def set_seed(seed: int, torch_threads: int | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    thread_count = torch_threads or min(8, os.cpu_count() or 1)
    torch.set_num_threads(max(1, int(thread_count)))
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        torch.use_deterministic_algorithms(True, warn_only=True)


def concatenate_data(parts: list[ArrayDataset]) -> ArrayDataset:
    return ArrayDataset(
        **{
            name: np.concatenate([getattr(part, name) for part in parts], axis=0)
            for name in ArrayDataset.__dataclass_fields__
        }
    )


def _train_model(
    name: str,
    model: nn.Module,
    train_data: ArrayDataset,
    train_quality: np.ndarray,
    validation_data: ArrayDataset,
    validation_quality: np.ndarray,
    normalization: Normalization,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    model_config = config["model"]
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
    )
    batch_size = model_config["batch_size"]
    best_validation = float("inf")
    best_epoch = -1
    patience_remaining = model_config["patience"]
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(config["seed"] + _stable_offset(name))

    for epoch in range(model_config["epochs"]):
        model.train()
        order = rng.permutation(len(train_data))
        training_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = torch_batch(
                train_data, train_quality, indices, normalization, device
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = quantile_loss(
                output["quantiles"],
                batch["target"],
                model_config["quantiles"],
                model_config["huber_weight"],
            )
            if "router_probabilities" in output:
                loss = loss + model_config["balance_weight"] * balance_loss(
                    output["router_probabilities"]
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            training_losses.append(float(loss.detach()))

        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            indices = np.arange(len(validation_data))
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = torch_batch(
                    validation_data,
                    validation_quality,
                    batch_indices,
                    normalization,
                    device,
                )
                output = model(batch)
                loss = quantile_loss(
                    output["quantiles"],
                    batch["target"],
                    model_config["quantiles"],
                )
                validation_losses.append(float(loss))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(training_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation - 1e-4:
            best_validation = validation_loss
            best_epoch = epoch + 1
            patience_remaining = model_config["patience"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                break
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    return {
        "name": name,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_validation_loss_normalized": best_validation,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
    }


def predict_model(
    model: nn.Module,
    data: ArrayDataset,
    quality: np.ndarray,
    normalization: Normalization,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    model.eval()
    predictions: list[np.ndarray] = []
    assignments: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    device = torch.device("cpu")
    with torch.no_grad():
        indices = np.arange(len(data))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = torch_batch(data, quality, batch_indices, normalization, device)
            output = model(batch)
            raw = output["quantiles"].cpu().numpy()
            predictions.append(
                raw * normalization.delay_std + normalization.delay_mean
            )
            if "router_assignments" in output:
                assignments.append(output["router_assignments"].cpu().numpy())
                probabilities.append(output["router_probabilities"].cpu().numpy())
    return (
        np.concatenate(predictions).astype(np.float32),
        np.concatenate(assignments) if assignments else None,
        np.concatenate(probabilities) if probabilities else None,
    )


def baseline_quantiles(
    point_prediction: np.ndarray,
    validation_point: np.ndarray,
    validation_target: np.ndarray,
    quantiles: list[float],
) -> np.ndarray:
    residual = validation_target - validation_point
    offsets = np.quantile(residual, quantiles, axis=0).T.astype(np.float32)
    return np.sort(
        point_prediction[:, :, None] + offsets[None, :, :], axis=2
    ).astype(np.float32)


def latency_metrics(
    model: nn.Module,
    data: ArrayDataset,
    quality: np.ndarray,
    normalization: Normalization,
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(data) == 0:
        raise ValueError("Cannot benchmark latency on an empty dataset")
    evaluation = config["evaluation"]
    batch_size = min(evaluation["inference_batch_size"], len(data))
    indices = np.arange(batch_size)
    batch = torch_batch(data, quality, indices, normalization, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        for _ in range(evaluation["warmup_batches"]):
            model(batch)
        times = []
        for _ in range(evaluation["timing_batches"]):
            started = time.perf_counter_ns()
            model(batch)
            times.append((time.perf_counter_ns() - started) / 1_000_000.0)
    checkpoint_size = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    process = psutil.Process()
    p50 = float(np.quantile(times, 0.5))
    p95 = float(np.quantile(times, 0.95))
    return {
        "batch_size": batch_size,
        "batch_p50_ms": p50,
        "batch_p95_ms": p95,
        "amortized_p50_ms_per_sample": p50 / batch_size,
        "amortized_p95_ms_per_sample": p95 / batch_size,
        "throughput_samples_per_second": float(
            batch_size / (np.mean(times) / 1000.0)
        ),
        "parameter_bytes": int(checkpoint_size),
        "process_rss_bytes_after_training": int(process.memory_info().rss),
        "cpu_threads": torch.get_num_threads(),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _scenario_by_type(config: dict[str, Any], scenario_type: str) -> dict[str, Any]:
    return next(
        scenario
        for scenario in config["scenarios"]
        if scenario["type"] == scenario_type
    )


def _chronological_validation_split(
    validation: ArrayDataset,
) -> tuple[ArrayDataset, ArrayDataset]:
    if len(validation) < 4:
        raise ValueError("Validation requires at least four samples")
    split = len(validation) // 2
    return validation.select(np.arange(split)), validation.select(
        np.arange(split, len(validation))
    )


def run_experiments(
    config: dict[str, Any],
    data_root: str | Path = "data",
    artifact_root: str | Path = "artifacts",
    run_name: str | None = None,
) -> Path:
    set_seed(
        config["seed"],
        config.get("experiment", {}).get("runtime", {}).get("torch_threads"),
    )
    run_name = run_name or time.strftime("pilot-%Y%m%d-%H%M%S", time.gmtime())
    run_dir = Path(artifact_root) / run_name
    if (run_dir / "results.json").exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    prediction_dir = run_dir / "predictions"
    scenario_dir = run_dir / "scenario_manifests"
    for directory in (checkpoint_dir, prediction_dir, scenario_dir):
        directory.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir / "resolved_config.json")

    processed = Path(data_root) / "processed" / "ride-silver-pilot"
    train = ArrayDataset.load(processed / "train.npz")
    validation = ArrayDataset.load(processed / "validation.npz")
    test = ArrayDataset.load(processed / "test.npz")
    validation_selection, validation_calibration = _chronological_validation_split(
        validation
    )
    normalization = Normalization.fit(train)
    normalization.save(run_dir / "normalization.json")

    clean_scenario = _scenario_by_type(config, "clean")
    curriculum = training_spec(config)
    clean_train = apply_corruption(
        train, clean_scenario, config["seed"] + _stable_offset("train-clean")
    )
    curriculum_parts = [clean_train.data]
    curriculum_quality = [clean_train.quality]
    for index in range(3):
        sample = apply_spec(
            train,
            curriculum,
            config["seed"] + _stable_offset(f"train-curriculum-{index}"),
            scenario_name=f"curriculum_{index}",
            randomize_per_sample=True,
        )
        curriculum_parts.append(sample.data)
        curriculum_quality.append(sample.quality)
    augmented_train = concatenate_data(curriculum_parts)
    augmented_quality = np.concatenate(curriculum_quality)
    clean_selection = apply_corruption(
        validation_selection,
        clean_scenario,
        config["seed"] + _stable_offset("validation-selection"),
    )
    clean_calibration = apply_corruption(
        validation_calibration,
        clean_scenario,
        config["seed"] + _stable_offset("validation-calibration"),
    )

    quantiles = config["model"]["quantiles"]
    requested_models = config.get("experiment", {}).get("models")
    if requested_models is None:
        requested_models = ["dense", "gru", "r2s_moe", "r2s_no_quality", "r3s_moe"]
    available_models: dict[str, nn.Module] = {
        "dense": DenseForecaster(config, include_quality=True),
        "gru": GRUForecaster(config),
        "r2s_moe": R2SMoE(config, include_quality=True),
        "r2s_no_quality": R2SMoE(config, include_quality=False),
        "r3s_moe": R3SMoE(config),
    }
    unknown_models = sorted(set(requested_models) - set(available_models))
    if unknown_models:
        raise ValueError(f"Unknown experiment models: {unknown_models}")
    model_objects: dict[str, nn.Module] = {
        name: available_models[name] for name in requested_models
    }
    training_summaries: dict[str, Any] = {}
    for model_name, model in model_objects.items():
        if model_name == "gru":
            model_train, model_quality = clean_train.data, clean_train.quality
        else:
            model_train, model_quality = augmented_train, augmented_quality
        training_summaries[model_name] = _train_model(
            model_name,
            model,
            model_train,
            model_quality,
            clean_selection.data,
            clean_selection.quality,
            normalization,
            config,
            checkpoint_dir / f"{model_name}.pt",
        )
        rl_config = config.get("experiment", {}).get("rl", {})
        if model_name == "r3s_moe" and rl_config.get("enabled", False):
            from railresilient.rl import train_contextual_bandit

            training_summaries[model_name]["rl_stage"] = train_contextual_bandit(
                model,
                model_train,
                model_quality,
                clean_selection.data,
                clean_selection.quality,
                normalization,
                config,
                checkpoint_dir / f"{model_name}.pt",
            )

    ridge = QuantileRidge(quantiles, include_quality=True)
    ridge.fit(
        augmented_train,
        augmented_quality,
        clean_calibration.data,
        clean_calibration.quality,
        normalization,
    )
    with (checkpoint_dir / "ridge.pkl").open("wb") as handle:
        pickle.dump(ridge, handle)

    calibration_predictions: dict[str, np.ndarray] = {}
    for model_name, model in model_objects.items():
        calibration_predictions[model_name] = predict_model(
            model,
            clean_calibration.data,
            clean_calibration.quality,
            normalization,
            config["evaluation"]["inference_batch_size"],
        )[0]
    calibration_predictions["ridge"] = ridge.predict(
        clean_calibration.data, clean_calibration.quality, normalization
    )
    calibration_persistence = np.repeat(
        clean_calibration.data.current_delay[:, None],
        config["data"]["future_events"],
        axis=1,
    )
    calibration_zero = np.zeros_like(calibration_persistence)
    baseline_calibration_predictions = {
        "zero": baseline_quantiles(
            calibration_zero,
            calibration_zero,
            clean_calibration.data.target_delay,
            quantiles,
        ),
        "persistence": baseline_quantiles(
            calibration_persistence,
            calibration_persistence,
            clean_calibration.data.target_delay,
            quantiles,
        ),
        "ridge": calibration_predictions["ridge"],
    }
    validation_persistence_point = calibration_persistence

    static_calibrators: dict[str, StaticQuantileCalibrator] = {}
    static_calibration_predictions: dict[str, np.ndarray] = {}
    for model_name in model_objects:
        calibrator = StaticQuantileCalibrator(quantiles)
        calibrator.fit(
            calibration_predictions[model_name],
            clean_calibration.data.target_delay,
        )
        static_calibrators[model_name] = calibrator
        static_calibration_predictions[model_name] = calibrator.transform(
            calibration_predictions[model_name]
        )

    severe_threshold = config["data"]["severe_delay_seconds"]
    transfer_buffer = float(config["evaluation"]["transfer_buffer_seconds"])
    alert_fraction = config["evaluation"]["alert_fraction"]
    alert_thresholds: dict[str, dict[str, float]] = {}
    for name, prediction in baseline_calibration_predictions.items():
        alert_thresholds[name] = {
            "static": select_alert_threshold(
                prediction, quantiles, severe_threshold, alert_fraction
            )
        }
    for name in model_objects:
        alert_thresholds[name] = {
            "uncalibrated": select_alert_threshold(
                calibration_predictions[name],
                quantiles,
                severe_threshold,
                alert_fraction,
            ),
            "static": select_alert_threshold(
                static_calibration_predictions[name],
                quantiles,
                severe_threshold,
                alert_fraction,
            ),
        }
        alert_thresholds[name]["online"] = alert_thresholds[name]["static"]

    calibration_artifact = {
        "validation_partition": {
            "selection_samples": len(validation_selection),
            "calibration_samples": len(validation_calibration),
            "partition": "chronological first half / second half",
        },
        "alert_fraction": alert_fraction,
        "alert_thresholds": alert_thresholds,
        "derived_transfer_buffer_seconds": config["evaluation"]["transfer_buffer_seconds"],
        "static_corrections": {
            name: calibrator.corrections.tolist()
            for name, calibrator in static_calibrators.items()
            if calibrator.corrections is not None
        },
    }
    (run_dir / "calibration.json").write_text(
        json.dumps(calibration_artifact, indent=2) + "\n", encoding="utf-8"
    )

    results: dict[str, Any] = {
        "run_name": run_name,
        "scope": "RIDE Silver selected-month CPU pilot; not official RIDE Gold and not Japanese validation",
        "training": training_summaries,
        "validation_partition": calibration_artifact["validation_partition"],
        "scenarios": {},
        "statistics": {},
        "systems": {},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
    }

    clean_test_quality: np.ndarray | None = None
    for scenario in config["scenarios"]:
        scenario_name = str(scenario["name"])
        scenario_seed = config["seed"] + _stable_offset(f"test-{scenario_name}")
        corrupted = apply_corruption(test, scenario, scenario_seed)
        if scenario["type"] == "clean":
            clean_test_quality = corrupted.quality
        write_corruption_manifest(
            corrupted, scenario_dir / f"{scenario_name}.json"
        )
        persistence_point = np.repeat(
            corrupted.data.current_delay[:, None],
            config["data"]["future_events"],
            axis=1,
        )
        persistence_prediction = baseline_quantiles(
            persistence_point,
            calibration_persistence,
            clean_calibration.data.target_delay,
            quantiles,
        )
        zero_point = np.zeros_like(persistence_point)
        zero_prediction = baseline_quantiles(
            zero_point,
            calibration_zero,
            clean_calibration.data.target_delay,
            quantiles,
        )
        baseline_predictions = {
            "zero": zero_prediction,
            "persistence": persistence_prediction,
            "ridge": ridge.predict(
                corrupted.data, corrupted.quality, normalization
            ),
        }
        persistence_mae = float(np.mean(np.abs(test.target_delay - persistence_point)))
        scenario_results: dict[str, Any] = {}
        for baseline_name, baseline_prediction in baseline_predictions.items():
            transfer_risk = connection_miss_probability(
                baseline_prediction, quantiles, transfer_buffer
            )
            point_alert = None
            if baseline_name == "persistence":
                point_alert = point_alert_metrics(
                    test.target_delay,
                    persistence_point,
                    severe_threshold,
                    validation_persistence_point,
                    clean_calibration.data.target_delay,
                    alert_fraction,
                )
            scenario_results[baseline_name] = {
                "static": probabilistic_metrics(
                    test.target_delay,
                    baseline_prediction,
                    quantiles,
                    persistence_mae,
                    severe_threshold,
                    alert_thresholds[baseline_name]["static"],
                ),
                "regime": regime_metrics(
                    test.target_delay, baseline_prediction, test.regime, quantiles
                ),
                "derived_transfer_feasibility": {
                    "assumed_buffer_seconds": transfer_buffer,
                    "mean_miss_probability_by_horizon": transfer_risk.mean(axis=0).tolist(),
                    "label_status": "derived utility; no observed passenger outcomes",
                },
                "point_alert": point_alert,
            }
            np.savez_compressed(
                prediction_dir / f"{scenario_name}_{baseline_name}.npz",
                static=baseline_prediction,
                connection_miss_probability=transfer_risk,
                target=test.target_delay,
                origin_ns=test.forecast_origin_ns,
                regime=test.regime,
            )

        online_predictions: dict[str, np.ndarray] = {}
        for model_name, model in model_objects.items():
            prediction, assignments, probabilities = predict_model(
                model,
                corrupted.data,
                corrupted.quality,
                normalization,
                config["evaluation"]["inference_batch_size"],
            )
            static_prediction = static_calibrators[model_name].transform(prediction)
            online = OnlineQuantileCalibrator(
                quantiles,
                window=config["calibration"]["window"],
                min_samples=config["calibration"]["min_samples"],
                update_stride=config["calibration"]["online_update_stride"],
            ).transform(
                prediction,
                test.target_delay,
                test.forecast_origin_ns,
                test.future_observed_ns,
                state_key=feed_state_key(corrupted.quality),
                initial_predictions=calibration_predictions[model_name],
                initial_target=clean_calibration.data.target_delay,
                initial_state_key=feed_state_key(clean_calibration.quality),
            )
            online_predictions[model_name] = online.predictions
            transfer_risk = connection_miss_probability(
                online.predictions, quantiles, transfer_buffer
            )
            scenario_results[model_name] = {
                "uncalibrated": probabilistic_metrics(
                    test.target_delay,
                    prediction,
                    quantiles,
                    persistence_mae,
                    severe_threshold,
                    alert_thresholds[model_name]["uncalibrated"],
                ),
                "static": probabilistic_metrics(
                    test.target_delay,
                    static_prediction,
                    quantiles,
                    persistence_mae,
                    severe_threshold,
                    alert_thresholds[model_name]["static"],
                ),
                "online": probabilistic_metrics(
                    test.target_delay,
                    online.predictions,
                    quantiles,
                    persistence_mae,
                    severe_threshold,
                    alert_thresholds[model_name]["online"],
                ),
                "regime": regime_metrics(
                    test.target_delay, online.predictions, test.regime, quantiles
                ),
                "online_updates": online.updates,
                "online_fallback_sample_horizons": online.fallback_predictions,
                "online_update_p50_ms": online.update_p50_ms,
                "online_update_p95_ms": online.update_p95_ms,
                "derived_transfer_feasibility": {
                    "assumed_buffer_seconds": transfer_buffer,
                    "mean_miss_probability_by_horizon": transfer_risk.mean(axis=0).tolist(),
                    "label_status": "derived utility; no observed passenger outcomes",
                },
            }
            if assignments is not None:
                scenario_results[model_name]["routing"] = routing_metrics(
                    assignments, test.regime, config["model"]["num_experts"]
                )
            np.savez_compressed(
                prediction_dir / f"{scenario_name}_{model_name}.npz",
                uncalibrated=prediction,
                static=static_prediction,
                online=online.predictions,
                connection_miss_probability=transfer_risk,
                target=test.target_delay,
                origin_ns=test.forecast_origin_ns,
                regime=test.regime,
                router_assignments=(
                    assignments
                    if assignments is not None
                    else np.array([], dtype=np.int64)
                ),
                router_probabilities=(
                    probabilities
                    if probabilities is not None
                    else np.array([], dtype=np.float32)
                ),
            )
        point_alert_result = scenario_results["persistence"]["point_alert"]
        scenario_results["h3_alert_comparison"] = {
            "point_persistence": point_alert_result,
            "probabilistic_brier": {
                model_name: scenario_results[model_name]["online"]["severe_brier"]
                for model_name in model_objects
            },
            "criterion": "lower Brier than point-only persistence at validation-frozen alert budget",
        }
        results["scenarios"][scenario_name] = scenario_results

        scenario_statistics: dict[str, Any] = {}
        if "r2s_moe" in online_predictions and "r2s_no_quality" in online_predictions:
            r2s_loss = weighted_interval_score_values(
                test.target_delay, online_predictions["r2s_moe"], quantiles
            ).mean(axis=1)
            no_quality_loss = weighted_interval_score_values(
                test.target_delay, online_predictions["r2s_no_quality"], quantiles
            ).mean(axis=1)
            scenario_statistics["r2s_moe_vs_no_quality_wis"] = paired_day_bootstrap(
                test.day_ordinal,
                no_quality_loss,
                r2s_loss,
                config["evaluation"]["bootstrap_replicates"],
                config["seed"] + _stable_offset(f"bootstrap-{scenario_name}-quality"),
            )
        if scenario["type"] == "clean" and "dense" in online_predictions and "r2s_moe" in online_predictions:
            dense_loss = weighted_interval_score_values(
                test.target_delay, online_predictions["dense"], quantiles
            ).mean(axis=1)
            r2s_loss = weighted_interval_score_values(
                test.target_delay, online_predictions["r2s_moe"], quantiles
            ).mean(axis=1)
            scenario_statistics["r2s_moe_vs_dense_wis"] = paired_day_bootstrap(
                test.day_ordinal,
                dense_loss,
                r2s_loss,
                config["evaluation"]["bootstrap_replicates"],
                config["seed"] + _stable_offset("bootstrap-clean-dense"),
            )
        results["statistics"][scenario_name] = scenario_statistics

    if clean_test_quality is None:
        raise AssertionError("Clean test scenario was not evaluated")
    for model_name, model in model_objects.items():
        latency = latency_metrics(
            model,
            test,
            clean_test_quality,
            normalization,
            config,
        )
        update_p50 = [
            results["scenarios"][name][model_name].get("online_update_p50_ms", 0.0)
            for name in results["scenarios"]
            if model_name in results["scenarios"][name]
        ]
        update_p95 = [
            results["scenarios"][name][model_name].get("online_update_p95_ms", 0.0)
            for name in results["scenarios"]
            if model_name in results["scenarios"][name]
        ]
        latency["online_update_p50_ms_max"] = float(max(update_p50, default=0.0))
        latency["online_update_p95_ms_max"] = float(max(update_p95, default=0.0))
        results["systems"][model_name] = latency
    results_path = run_dir / "results.json"
    results_path.write_text(
        json.dumps(_jsonable(results), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_dir



def feed_state_key(quality: np.ndarray) -> np.ndarray:
    """Map bounded observable feed quality into three calibration strata."""
    if quality.ndim != 2 or quality.shape[1] < 7:
        raise ValueError("v2 quality schema requires seven channels")
    return np.where(
        quality[:, 6] > 0.5,
        2,
        np.where((quality[:, 0] > 0.0) | (quality[:, 2] > 0.25), 1, 0),
    ).astype(np.int64)


def point_from_quantiles(prediction: np.ndarray, quantiles: list[float]) -> np.ndarray:
    return prediction[:, :, quantiles.index(0.5)]
