# RailResilient documentation

This is the reader-friendly map of the project. Start with the short overview, then choose the path that matches your goal.

## Status at a glance

| Label | Meaning |
|---|---|
| **Historical** | Completed v1/v2/v3 evidence retained for comparison. |
| **Current candidate** | R4S-MoE: four experts, shared expert, top-2 routing. |
| **Exploratory** | One-seed v4 variants, including offline contextual-bandit RL. |
| **Not claimed** | Train control, signaling, safety certification, Japanese validation, passenger outcomes, or autonomous dispatch. |

## Start here

1. [Project overview](overview.md) — what the system does, what the data means, and what the results do and do not prove.
2. [Model architectures](model-architectures.md) — simple diagrams and explanations for the baselines, R2S-MoE v2, R3S-MoE v3, and R4S-MoE v4.
3. [Benchmark summary](benchmark-summary.md) — the main tables and current graphs.
4. [Findings and decisions](findings.md) — what improved, what did not, and what should happen next.
5. [Experiment protocol](experiment-protocol.md) — chronological splits, calibration, corruption, metrics, and research roadmap.

## Choose a reading path

### For a quick review

Read [overview](overview.md), then [benchmark summary](benchmark-summary.md) and [findings](findings.md).

### For model and ML details

Read [model architectures](model-architectures.md), [v3 design research](v3-design-research.md), [R4S top-2 design](r4s-top2-design.md), and the [five-variant benchmark](r4s-variant-benchmark.md).

### For reproducibility

Read [experiment protocol](experiment-protocol.md), inspect the configs in `configs/`, and follow the commands in the root [README](../README.md). Checkpoint and prediction artifacts are intentionally local and ignored.

### For the interactive demo

See [`demo/README.md`](../demo/README.md) or open the [public simulation preview](https://railresilient-simulation.cultivate-earl.workers.dev). The public edge preview uses an honest deterministic fallback; it does not expose the private PyTorch checkpoint.

## Visual index

- [Model architecture family](architecture-family.svg)
- [Experiment and evaluation flow](experiment-flow.svg)
- [v4 clean benchmark graph](v4-clean-benchmark.svg)
- [v4 stress benchmark graph](v4-stress-benchmark.svg)
- [Figure notes and sources](figures/README.md)

## Historical documents

The following documents remain useful but are now organized under the overview and benchmark pages:

- [v2 redesign comparison](v2-redesign-comparison.md)
- [v3 comparison](v3-comparison.md)
- [v3 design research](v3-design-research.md)
- [R4S-MoE top-2 design](r4s-top2-design.md)
- [R4S-MoE five-variant benchmark](r4s-variant-benchmark.md)
- [Research protocol and roadmap](experiment-protocol.md)
- [Literature matrix](literature-matrix.md)

## Evidence rule

The current best-looking v4 row is the end-to-end offline contextual-bandit run: clean WIS `28.705`, MAE `45.191 s`. This is **one-seed exploratory evidence**, not proof that RL or R4S-MoE is superior. The result must be repeated across seeds and, for sequential RL claims, evaluated with a validated simulator or logged-action off-policy protocol.
