"""Baseline, dense, GRU, and R2S-MoE models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F

from railresilient.data import QUALITY_DIM, ArrayDataset


@dataclass(slots=True)
class Normalization:
    delay_mean: float
    delay_std: float
    planned_scale: float = 3600.0

    @classmethod
    def fit(cls, data: ArrayDataset) -> Normalization:
        observed = data.past_delay[data.observation_mask > 0.5]
        mean = float(np.mean(observed))
        std = max(float(np.std(observed)), 30.0)
        return cls(delay_mean=mean, delay_std=std)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def numpy_features(
    data: ArrayDataset, quality: np.ndarray, normalization: Normalization, include_quality: bool = True
) -> np.ndarray:
    delay = (data.past_delay - normalization.delay_mean) / normalization.delay_std
    planned = data.past_planned_delta / normalization.planned_scale
    future = data.future_planned_delta / normalization.planned_scale
    features = [delay, planned, data.observation_mask, future]
    if include_quality:
        features.append(quality)
    return np.concatenate(features, axis=1).astype(np.float32)


def torch_batch(
    data: ArrayDataset,
    quality: np.ndarray,
    indices: np.ndarray,
    normalization: Normalization,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    index = np.asarray(indices)
    return {
        "past_delay": torch.as_tensor(
            (data.past_delay[index] - normalization.delay_mean) / normalization.delay_std,
            dtype=torch.float32,
            device=device,
        ),
        "past_planned": torch.as_tensor(
            data.past_planned_delta[index] / normalization.planned_scale,
            dtype=torch.float32,
            device=device,
        ),
        "event_type": torch.as_tensor(data.past_event_type[index], dtype=torch.long, device=device),
        "station": torch.as_tensor(data.past_station[index], dtype=torch.long, device=device),
        "mask": torch.as_tensor(data.observation_mask[index], dtype=torch.float32, device=device),
        "future_planned": torch.as_tensor(
            data.future_planned_delta[index] / normalization.planned_scale,
            dtype=torch.float32,
            device=device,
        ),
        "quality": torch.as_tensor(quality[index], dtype=torch.float32, device=device),
        "target": torch.as_tensor(
            (data.target_delay[index] - normalization.delay_mean) / normalization.delay_std,
            dtype=torch.float32,
            device=device,
        ),
    }


class QuantileRidge:
    """A fast autoregressive baseline with empirical residual quantiles."""

    def __init__(self, quantiles: list[float], alpha: float = 10.0, include_quality: bool = True) -> None:
        self.quantiles = np.asarray(quantiles, dtype=np.float32)
        self.alpha = alpha
        self.include_quality = include_quality
        self.models: list[Ridge] = []
        self.residual_quantiles: np.ndarray | None = None

    def fit(
        self,
        train: ArrayDataset,
        train_quality: np.ndarray,
        validation: ArrayDataset,
        validation_quality: np.ndarray,
        normalization: Normalization,
    ) -> None:
        x_train = numpy_features(train, train_quality, normalization, self.include_quality)
        x_validation = numpy_features(validation, validation_quality, normalization, self.include_quality)
        self.models = []
        medians = []
        for horizon in range(train.target_delay.shape[1]):
            model = Ridge(alpha=self.alpha)
            model.fit(x_train, train.target_delay[:, horizon])
            self.models.append(model)
            medians.append(model.predict(x_validation))
        median_prediction = np.stack(medians, axis=1)
        residual = validation.target_delay - median_prediction
        self.residual_quantiles = np.quantile(residual, self.quantiles, axis=0).T.astype(np.float32)

    def predict(
        self, data: ArrayDataset, quality: np.ndarray, normalization: Normalization
    ) -> np.ndarray:
        if not self.models or self.residual_quantiles is None:
            raise RuntimeError("QuantileRidge must be fitted before prediction")
        features = numpy_features(data, quality, normalization, self.include_quality)
        median = np.stack([model.predict(features) for model in self.models], axis=1)
        prediction = median[:, :, None] + self.residual_quantiles[None, :, :]
        return np.sort(prediction.astype(np.float32), axis=2)


class EventEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        station_buckets: int,
        station_embedding_dim: int,
        event_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_buckets, station_embedding_dim)
        self.event_embedding = nn.Embedding(3, event_embedding_dim)
        input_dim = 3 + station_embedding_dim + event_embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=2, groups=hidden_dim
        )
        self.z_projection = nn.Linear(hidden_dim, hidden_dim)
        self.a_projection = nn.Linear(hidden_dim + QUALITY_DIM, hidden_dim)
        self.q_projection = nn.Linear(hidden_dim + QUALITY_DIM, hidden_dim)
        self.residual_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["mask"]
        continuous = torch.stack([batch["past_delay"], batch["past_planned"], mask], dim=-1)
        embedded = torch.cat(
            [
                continuous,
                self.event_embedding(batch["event_type"]),
                self.station_embedding(batch["station"]),
            ],
            dim=-1,
        )
        projected = F.silu(self.input_projection(embedded))
        # Masked positions carry only imputed values, so they are zeroed before the
        # depthwise mixing to stop imputations leaking into neighbouring events.
        gate = mask[:, :, None]
        projected = projected * gate
        convolved = self.depthwise(projected.transpose(1, 2))[:, :, : projected.shape[1]].transpose(1, 2)
        projected = (projected + F.silu(convolved)) * gate
        state = torch.zeros_like(projected[:, 0])
        quality = batch["quality"]
        for step in range(projected.shape[1]):
            current = projected[:, step]
            combined = torch.cat([current, quality], dim=-1)
            update = torch.sigmoid(self.a_projection(combined))
            reliability = torch.sigmoid(self.q_projection(combined))
            candidate = F.silu(self.z_projection(current))
            proposed = reliability * candidate + (1.0 - reliability) * state
            proposed = update * state + (1.0 - update) * proposed
            # An unobserved event must not advance the recurrent state at all.
            step_gate = gate[:, step]
            state = step_gate * proposed + (1.0 - step_gate) * state
        last_valid = self._last_valid(projected, mask)
        return self.norm(state + self.residual_projection(last_valid))

    @staticmethod
    def _last_valid(projected: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Most recent observed event representation, or zeros for an empty context."""
        positions = torch.arange(mask.shape[1], device=mask.device, dtype=mask.dtype)
        index = torch.where(mask > 0.5, positions[None, :], torch.full_like(mask, -1.0))
        newest = index.max(dim=1).values
        has_valid = (newest >= 0).to(projected.dtype)[:, None]
        gather_index = newest.clamp(min=0).long()[:, None, None].expand(
            -1, 1, projected.shape[-1]
        )
        gathered = torch.gather(projected, 1, gather_index).squeeze(1)
        return gathered * has_valid


class MonotonicQuantileHead(nn.Module):
    def __init__(self, hidden_dim: int, horizons: int, quantiles: list[float]) -> None:
        super().__init__()
        if quantiles != [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
            raise ValueError("The monotonic head currently requires seven symmetric quantiles")
        self.horizons = horizons
        self.future_projection = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU())
        self.output = nn.Linear(hidden_dim * 2, 7)

    def forward(self, hidden: torch.Tensor, future_planned: torch.Tensor) -> torch.Tensor:
        repeated = hidden[:, None, :].expand(-1, self.horizons, -1)
        future = self.future_projection(future_planned[:, :, None])
        raw = self.output(torch.cat([repeated, future], dim=-1))
        median = raw[:, :, 3:4]
        lower_steps = F.softplus(raw[:, :, :3])
        upper_steps = F.softplus(raw[:, :, 4:])
        q25 = median - lower_steps[:, :, 2:3]
        q10 = q25 - lower_steps[:, :, 1:2]
        q05 = q10 - lower_steps[:, :, 0:1]
        q75 = median + upper_steps[:, :, 0:1]
        q90 = q75 + upper_steps[:, :, 1:2]
        q95 = q90 + upper_steps[:, :, 2:3]
        return torch.cat([q05, q10, q25, median, q75, q90, q95], dim=-1)


class DenseForecaster(nn.Module):
    def __init__(self, config: dict[str, Any], include_quality: bool = True) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        hidden = model["hidden_dim"]
        self.include_quality = include_quality
        self.encoder = EventEncoder(
            hidden,
            data["station_buckets"],
            model["station_embedding_dim"],
            model["event_embedding_dim"],
        )
        trunk_width = int(model.get("dense_trunk_width", hidden * 2))
        self.dense = nn.Sequential(
            nn.Linear(hidden + (QUALITY_DIM if include_quality else 0), trunk_width),
            nn.SiLU(),
            nn.Dropout(model["dropout"]),
            nn.Linear(trunk_width, hidden),
            nn.SiLU(),
        )
        self.head = MonotonicQuantileHead(hidden, data["future_events"], model["quantiles"])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.include_quality:
            batch = dict(batch)
            batch["quality"] = torch.zeros_like(batch["quality"])
        encoded = self.encoder(batch)
        dense_input = torch.cat([encoded, batch["quality"]], dim=-1) if self.include_quality else encoded
        hidden = self.dense(dense_input)
        return {"quantiles": self.head(hidden, batch["future_planned"])}


class GRUForecaster(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        hidden = model["hidden_dim"]
        self.event_embedding = nn.Embedding(3, model["event_embedding_dim"])
        self.station_embedding = nn.Embedding(data["station_buckets"], model["station_embedding_dim"])
        input_dim = 3 + model["event_embedding_dim"] + model["station_embedding_dim"]
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = MonotonicQuantileHead(hidden, data["future_events"], model["quantiles"])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = torch.cat(
            [
                torch.stack([batch["past_delay"], batch["past_planned"], batch["mask"]], dim=-1),
                self.event_embedding(batch["event_type"]),
                self.station_embedding(batch["station"]),
            ],
            dim=-1,
        )
        _, hidden = self.gru(inputs)
        return {"quantiles": self.head(hidden[-1], batch["future_planned"])}


class Expert(nn.Module):
    def __init__(self, hidden_dim: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * multiplier),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * multiplier, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)


class R2SMoE(nn.Module):
    def __init__(self, config: dict[str, Any], include_quality: bool = True) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        hidden = model["hidden_dim"]
        self.include_quality = include_quality
        self.num_experts = model["num_experts"]
        self.encoder = EventEncoder(
            hidden,
            data["station_buckets"],
            model["station_embedding_dim"],
            model["event_embedding_dim"],
        )
        self.shared = Expert(hidden, model["expert_multiplier"], model["dropout"])
        self.experts = nn.ModuleList(
            [Expert(hidden, model["expert_multiplier"], model["dropout"]) for _ in range(self.num_experts)]
        )
        router_input = hidden + 3 + (QUALITY_DIM if include_quality else 0)
        self.router = nn.Sequential(
            nn.Linear(router_input, max(8, hidden // 4)),
            nn.SiLU(),
            nn.Linear(max(8, hidden // 4), self.num_experts),
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.head = MonotonicQuantileHead(hidden, data["future_events"], model["quantiles"])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        working = batch
        if not self.include_quality:
            working = dict(batch)
            working["quality"] = torch.zeros_like(batch["quality"])
        encoded = self.encoder(working)
        current = working["past_delay"][:, -1]
        trend = current - working["past_delay"][:, -min(4, working["past_delay"].shape[1])]
        shock = ((current > 1.0) | (trend > 0.5)).float()
        recovery = ((current > 0.2) & (trend < -0.3)).float()
        regime_features = torch.stack([current, trend, shock - recovery], dim=-1)
        router_parts = [encoded, regime_features]
        if self.include_quality:
            router_parts.append(batch["quality"])
        logits = self.router(torch.cat(router_parts, dim=-1))
        probabilities = torch.softmax(logits, dim=-1)
        assignments = torch.argmax(probabilities, dim=-1)
        if self.training:
            hard = F.one_hot(assignments, self.num_experts).to(probabilities.dtype)
            weights = hard + probabilities - probabilities.detach()
            expert_outputs = torch.stack(
                [expert(encoded) for expert in self.experts], dim=1
            )
            routed = torch.sum(weights[:, :, None] * expert_outputs, dim=1)
        else:
            # Genuine top-1 sparse dispatch: each expert only sees its own rows.
            routed = torch.zeros_like(encoded)
            for index, expert in enumerate(self.experts):
                selected = torch.nonzero(assignments == index, as_tuple=True)[0]
                if selected.numel():
                    routed.index_copy_(
                        0, selected, expert(encoded.index_select(0, selected))
                    )
        hidden = self.output_norm(encoded + self.shared(encoded) + routed)
        return {
            "quantiles": self.head(hidden, batch["future_planned"]),
            "router_probabilities": probabilities,
            "router_assignments": torch.argmax(probabilities, dim=-1),
        }


def quantile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float],
    huber_weight: float = 0.0,
) -> torch.Tensor:
    q = torch.as_tensor(quantiles, device=prediction.device, dtype=prediction.dtype)
    error = target[:, :, None] - prediction
    pinball = torch.maximum(q * error, (q - 1.0) * error).mean()
    if huber_weight:
        median_index = quantiles.index(0.5)
        pinball = pinball + huber_weight * F.huber_loss(prediction[:, :, median_index], target)
    return pinball


def balance_loss(probabilities: torch.Tensor) -> torch.Tensor:
    mean_usage = probabilities.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
    return torch.sum((mean_usage - target) ** 2)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)



class SelectiveStatePath(nn.Module):
    """Small quality-conditioned diagonal state scan for short event histories."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        input_dim = hidden_dim + QUALITY_DIM
        self.forget = nn.Linear(input_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        quality: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.zeros_like(tokens[:, 0])
        outputs: list[torch.Tensor] = []
        expanded_quality = quality[:, None, :].expand(-1, tokens.shape[1], -1)
        for step in range(tokens.shape[1]):
            token = tokens[:, step]
            combined = torch.cat([token, expanded_quality[:, step]], dim=-1)
            forget = torch.sigmoid(self.forget(combined))
            input_gate = torch.sigmoid(self.input_gate(combined))
            candidate = torch.tanh(self.candidate(token))
            proposed = forget * state + input_gate * candidate
            step_mask = mask[:, step, None]
            state = step_mask * proposed + (1.0 - step_mask) * state
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class R3SEncoder(nn.Module):
    """Reliability-aware local-mixing plus bidirectional-past state encoder."""

    def __init__(
        self,
        hidden_dim: int,
        station_buckets: int,
        station_embedding_dim: int,
        event_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_buckets, station_embedding_dim)
        self.event_embedding = nn.Embedding(3, event_embedding_dim)
        input_dim = 3 + station_embedding_dim + event_embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=2, groups=hidden_dim
        )
        self.forward_path = SelectiveStatePath(hidden_dim)
        self.reverse_path = SelectiveStatePath(hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 3 + QUALITY_DIM, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["mask"]
        gate = mask[:, :, None]
        continuous = torch.stack(
            [batch["past_delay"], batch["past_planned"], mask], dim=-1
        )
        embedded = torch.cat(
            [
                continuous,
                self.event_embedding(batch["event_type"]),
                self.station_embedding(batch["station"]),
            ],
            dim=-1,
        )
        tokens = F.silu(self.input_projection(embedded)) * gate
        local = self.depthwise(tokens.transpose(1, 2))[:, :, : tokens.shape[1]]
        local = (tokens + F.silu(local.transpose(1, 2))) * gate
        forward_states = self.forward_path(local, mask, batch["quality"])
        reverse_local = torch.flip(local, dims=[1])
        reverse_mask = torch.flip(mask, dims=[1])
        reverse_states = torch.flip(
            self.reverse_path(reverse_local, reverse_mask, batch["quality"]),
            dims=[1],
        )
        forward_last = self._last_valid(forward_states, mask)
        reverse_last = self._last_valid(reverse_states, mask)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (local * gate).sum(dim=1) / denominator
        fused = torch.cat([forward_last, reverse_last, pooled, batch["quality"]], dim=-1)
        return self.norm(F.silu(self.fusion(fused)))

    @staticmethod
    def _last_valid(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(mask.shape[1], device=mask.device, dtype=mask.dtype)
        index = torch.where(mask > 0.5, positions[None, :], torch.full_like(mask, -1.0))
        newest = index.max(dim=1).values
        has_valid = (newest >= 0).to(states.dtype)[:, None]
        gather_index = newest.clamp(min=0).long()[:, None, None].expand(
            -1, 1, states.shape[-1]
        )
        return torch.gather(states, 1, gather_index).squeeze(1) * has_valid

    @staticmethod
    def last_valid_value(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(mask.shape[1], device=mask.device, dtype=mask.dtype)
        index = torch.where(mask > 0.5, positions[None, :], torch.full_like(mask, -1.0))
        newest = index.max(dim=1).values
        has_valid = (newest >= 0).to(values.dtype)
        gather_index = newest.clamp(min=0).long()[:, None]
        return values.gather(1, gather_index).squeeze(1) * has_valid


class LowRankAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(F.silu(self.down(hidden))))


class PersistenceResidualQuantileHead(nn.Module):
    """Monotonic quantiles around the last-valid delay with horizon conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        horizons: int,
        quantiles: list[float],
        horizon_embedding_dim: int,
    ) -> None:
        super().__init__()
        if quantiles != [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
            raise ValueError("The residual head currently requires seven symmetric quantiles")
        self.horizons = horizons
        self.horizon_embedding = nn.Embedding(horizons, horizon_embedding_dim)
        self.future_projection = nn.Linear(1, horizon_embedding_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + 2 * horizon_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim, 7)

    def forward(
        self,
        hidden: torch.Tensor,
        future_planned: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        horizon_ids = torch.arange(
            self.horizons, device=hidden.device, dtype=torch.long
        )[None, :].expand(hidden.shape[0], -1)
        horizon = self.horizon_embedding(horizon_ids)
        future = self.future_projection(future_planned[:, :, None])
        repeated = hidden[:, None, :].expand(-1, self.horizons, -1)
        raw = self.output(self.trunk(torch.cat([repeated, horizon, future], dim=-1)))
        median = raw[:, :, 3:4]
        lower_steps = F.softplus(raw[:, :, :3])
        upper_steps = F.softplus(raw[:, :, 4:])
        q25 = median - lower_steps[:, :, 2:3]
        q10 = q25 - lower_steps[:, :, 1:2]
        q05 = q10 - lower_steps[:, :, 0:1]
        q75 = median + upper_steps[:, :, 0:1]
        q90 = q75 + upper_steps[:, :, 1:2]
        q95 = q90 + upper_steps[:, :, 2:3]
        residual = torch.cat([q05, q10, q25, median, q75, q90, q95], dim=-1)
        return baseline[:, :, None] + residual


class R3SMoE(nn.Module):
    """Reliability-gated residual selective-state MoE forecaster."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        hidden = model["hidden_dim"]
        self.num_experts = model["num_experts"]
        self.encoder = R3SEncoder(
            hidden,
            data["station_buckets"],
            model["station_embedding_dim"],
            model["event_embedding_dim"],
        )
        self.shared = Expert(hidden, model["expert_multiplier"], model["dropout"])
        adapter_rank = int(model.get("r3s_adapter_rank", max(4, hidden // 6)))
        self.adapters = nn.ModuleList(
            [LowRankAdapter(hidden, adapter_rank, model["dropout"]) for _ in range(self.num_experts)]
        )
        router_input = hidden + 3 + QUALITY_DIM
        router_width = max(8, hidden // 3)
        self.router = nn.Sequential(
            nn.Linear(router_input, router_width),
            nn.SiLU(),
            nn.Linear(router_width, self.num_experts),
        )
        self.reliability_gate = nn.Linear(hidden + QUALITY_DIM, hidden)
        self.output_norm = nn.LayerNorm(hidden)
        horizon_embedding_dim = int(model.get("r3s_horizon_embedding_dim", max(4, hidden // 4)))
        self.head = PersistenceResidualQuantileHead(
            hidden,
            data["future_events"],
            model["quantiles"],
            horizon_embedding_dim,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        quality = batch["quality"]
        encoded = self.encoder(batch)
        current = R3SEncoder.last_valid_value(batch["past_delay"], batch["mask"])
        first = batch["past_delay"][:, :1].squeeze(1)
        trend = current - first
        shock = ((current > 1.0) | (trend > 0.5)).float()
        recovery = ((current > 0.2) & (trend < -0.3)).float()
        regime_features = torch.stack([current, trend, shock - recovery], dim=-1)
        logits = self.router(torch.cat([encoded, regime_features, quality], dim=-1))
        probabilities = torch.softmax(logits, dim=-1)
        assignments = torch.argmax(probabilities, dim=-1)
        if self.training:
            hard = F.one_hot(assignments, self.num_experts).to(probabilities.dtype)
            weights = hard + probabilities - probabilities.detach()
            adapter_outputs = torch.stack(
                [adapter(encoded) for adapter in self.adapters], dim=1
            )
            routed = torch.sum(weights[:, :, None] * adapter_outputs, dim=1)
        else:
            routed = torch.zeros_like(encoded)
            for index, adapter in enumerate(self.adapters):
                selected = torch.nonzero(assignments == index, as_tuple=True)[0]
                if selected.numel():
                    routed.index_copy_(
                        0, selected, adapter(encoded.index_select(0, selected))
                    )
        reliability = torch.sigmoid(
            self.reliability_gate(torch.cat([encoded, quality], dim=-1))
        )
        hidden = self.output_norm(encoded + self.shared(encoded) + reliability * routed)
        return {
            "quantiles": self.head(
                hidden,
                batch["future_planned"],
                current[:, None].expand(-1, batch["future_planned"].shape[1]),
            ),
            "router_probabilities": probabilities,
            "router_assignments": assignments,
        }
