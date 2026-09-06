"""Offline contextual-bandit fine-tuning for R4S-MoE routing.

The repository has historical forecasting rows, not an operational transition
simulator. This module therefore implements a one-step, offline contextual
bandit: the router chooses an expert pair for a context and receives a reward
from the already observed future horizon. It is intentionally not described as
sequential railway-control RL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from railresilient.data import ArrayDataset
from railresilient.models import (
    Normalization,
    count_parameters,
    quantile_loss,
    torch_batch,
)

PAIR_INDICES = torch.tensor(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long
)


def _pair_logits(probabilities: torch.Tensor) -> torch.Tensor:
    pairs = PAIR_INDICES.to(probabilities.device)
    log_probabilities = probabilities.clamp_min(1e-8).log()
    return log_probabilities[:, pairs[:, 0]] + log_probabilities[:, pairs[:, 1]]


def _wis_per_row(
    target: torch.Tensor, prediction: torch.Tensor, quantiles: list[float]
) -> torch.Tensor:
    indices = {round(value, 4): index for index, value in enumerate(quantiles)}
    median = prediction[:, :, indices[0.5]]
    components = 0.5 * torch.abs(target - median)
    for alpha, lower_q, upper_q in ((0.5, 0.25, 0.75), (0.2, 0.1, 0.9), (0.1, 0.05, 0.95)):
        lower = prediction[:, :, indices[lower_q]]
        upper = prediction[:, :, indices[upper_q]]
        interval = upper - lower
        interval = interval + (2.0 / alpha) * torch.relu(lower - target)
        interval = interval + (2.0 / alpha) * torch.relu(target - upper)
        components = components + (alpha / 2.0) * interval
    return (components / (len(indices) - 3.5)).mean(dim=1)


def _validation_wis(
    model: nn.Module,
    data: ArrayDataset,
    quality: np.ndarray,
    normalization: Normalization,
    quantiles: list[float],
    batch_size: int,
) -> float:
    model.eval()
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            indices = np.arange(start, min(start + batch_size, len(data)))
            batch = torch_batch(data, quality, indices, normalization, torch.device("cpu"))
            output = model(batch)
            values.append(_wis_per_row(batch["target"], output["quantiles"], quantiles))
    return float(torch.cat(values).mean())


def train_contextual_bandit(
    model: nn.Module,
    train_data: ArrayDataset,
    train_quality: np.ndarray,
    validation_data: ArrayDataset,
    validation_quality: np.ndarray,
    normalization: Normalization,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Fine-tune an already supervised R4S checkpoint with one-step REINFORCE."""
    experiment = config.get("experiment", {})
    rl_config = experiment.get("rl", {})
    if rl_config.get("method") != "reinforce_contextual_bandit":
        raise ValueError("Unsupported RL method; expected reinforce_contextual_bandit")

    router_only = bool(rl_config.get("router_only", False))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (not router_only) or name.startswith("router.")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("RL stage has no trainable parameters")

    learning_rate = float(rl_config.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=float(rl_config.get("weight_decay", 0.0)),
    )
    epochs = int(rl_config.get("epochs", 2))
    batch_size = int(rl_config.get("batch_size", config["model"]["batch_size"]))
    temperature = float(rl_config.get("temperature", 1.0))
    entropy_coefficient = float(rl_config.get("entropy_coefficient", 0.01))
    anchor_weight = float(rl_config.get("supervised_anchor_weight", 0.25))
    quantiles = config["model"]["quantiles"]
    rng = np.random.default_rng(config["seed"] + 910_001)
    best_validation_wis = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []
    running_baseline = 0.0
    baseline_initialized = False

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_data))
        policy_losses: list[float] = []
        rewards: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = torch_batch(train_data, train_quality, indices, normalization, torch.device("cpu"))
            optimizer.zero_grad(set_to_none=True)
            policy_output = model(batch)
            distribution = Categorical(logits=_pair_logits(policy_output["router_probabilities"]) / temperature)
            actions = distribution.sample()
            route_indices = PAIR_INDICES.to(actions.device)[actions]
            action_batch = dict(batch)
            action_output = model(action_batch, route_indices=route_indices)
            reward = -_wis_per_row(batch["target"], action_output["quantiles"], quantiles)
            batch_reward = float(reward.mean().detach())
            if not baseline_initialized:
                running_baseline = batch_reward
                baseline_initialized = True
            else:
                running_baseline = 0.95 * running_baseline + 0.05 * batch_reward
            advantage = reward.detach() - running_baseline
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
            policy_loss = -(advantage * distribution.log_prob(actions)).mean()
            entropy = distribution.entropy().mean()
            anchor = quantile_loss(
                action_output["quantiles"],
                batch["target"],
                quantiles,
                config["model"]["huber_weight"],
            )
            loss = policy_loss - entropy_coefficient * entropy + anchor_weight * anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach()))
            rewards.append(batch_reward)

        validation_wis = _validation_wis(
            model,
            validation_data,
            validation_quality,
            normalization,
            quantiles,
            batch_size,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "policy_loss": float(np.mean(policy_losses)),
                "mean_reward": float(np.mean(rewards)),
                "validation_wis_normalized": validation_wis,
            }
        )
        if validation_wis < best_validation_wis:
            best_validation_wis = validation_wis
            best_epoch = epoch + 1
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad = True
    return {
        "method": "offline one-step REINFORCE contextual bandit",
        "router_only": router_only,
        "action_space": "six unordered pairs from four experts",
        "reward": "negative four-horizon normalized WIS with supervised quantile anchor",
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_validation_wis_normalized": best_validation_wis,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
    }
