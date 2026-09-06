# RailResilient overview

## The problem in one sentence

RailResilient asks whether a small CPU-first model can forecast the next four railway delay events when observations are missing, delayed, stale, duplicated, or inconsistent.

## What goes in and out

```text
8 observed train events
+ 7 bounded feed-quality signals
+ station/event context
            ↓
      forecasting model
            ↓
4 future horizons × 7 quantiles
+ calibration, alert, and routing diagnostics
```

The output is a probability range, not a guaranteed arrival time. Lower delay error, lower weighted interval score (WIS), and well-calibrated coverage are preferred.

## Dataset and protocol

- Dataset: selected months from RIDE Silver, Belgian railway operations.
- Train: January–April 2023, capped at 80,000 samples.
- Validation: January–February 2024, capped at 20,000 samples.
- Test: January–February 2025, capped at 30,000 samples.
- Context: eight past events; forecast horizon: four future events.
- Quantiles: 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95.
- Stress scenarios: clean, 15% packet loss, five-minute staleness, 15-minute outage, and combined corruption.

Validation is chronological: the first validation portion selects the checkpoint, the later portion fits calibration and alert thresholds, and the test period is evaluated afterward. Synthetic feed corruption is applied causally to context observations; it is not a claim that measured GTFS-RT receipt histories are available.

## Model history

| Stage | Architecture | Main idea | Evidence status | Decision |
|---|---|---|---|---|
| v1 | Persistence, Ridge, Dense, GRU | Establish simple point and probabilistic baselines. | Historical reference. | Keep as baselines. |
| v2 | R2S-MoE | Bounded quality channels, mask-aware state, quality-conditioned residual routing. | Two-seed aggregate reference. | Reliability conditioning helped stress robustness, but clean performance was mixed. |
| v3 | R3S-MoE | Dual encoder, shared expert, 3 low-rank adapters, top-1 routing, persistence-anchored quantiles. | One completed seed. Clean WIS `28.72`. | Strong historical control; not a proof of superiority. |
| v4 | R4S-MoE | Four adapters, shared expert, top-2 routing, same reliability-aware backbone. | Five one-seed variants; exploratory. | End-to-end offline contextual-bandit row is the current candidate, pending replication. |

See the [architecture page](model-architectures.md) for diagrams and the [benchmark summary](benchmark-summary.md) for numbers.

## Claim boundaries

This repository demonstrates a research prototype and decision-support workflow. It does not establish train-control suitability, signaling use, dispatch automation, safety certification, Japanese validation, passenger outcomes, or absolute novelty. The public Cloudflare preview intentionally uses a deterministic fallback because private checkpoints are not hosted at the edge.

## Practical interpretation

The model is most useful as a human-facing reliability-aware forecast:

- high-quality feeds should produce tighter, calibrated ranges;
- stale or missing feeds should widen uncertainty and trigger advisory warnings;
- outages should be treated as degraded evidence, not hidden as confident predictions;
- the model should be compared against persistence, not only against another neural model.

## Reproduce

The root [README](../README.md) contains environment and command examples. The detailed [experiment protocol](experiment-protocol.md) defines the split, metrics, calibration, and future evaluation requirements.
