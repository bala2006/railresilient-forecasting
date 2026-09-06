# Benchmark summary

This page is the main results view. Numbers are kept separate by version and evidence scope so a one-seed result is not mistaken for a multi-seed result.

## How to read the tables

- Lower MAE, WIS, severe-delay Brier, and latency are better.
- 90% coverage should be close to 90%, not maximized without regard to interval width.
- `online` means the learned model uses the repository's causal online calibration path.
- v2 values are two-seed aggregates where stated; v3 and v4 values are one seed.
- The v4 RL rows are offline contextual-bandit experiments, not sequential operational RL.

## Version timeline

| Version | Model | Routing | Seed scope | Clean WIS | Main conclusion |
|---|---|---|---:|---:|---|
| v1 | Dense / baseline family | None | 1 | 29.40 best v1 reference | Established baselines and exposed reliability failure modes. |
| v2 | R2S-MoE | 3 experts, top-1 | 2 seeds | 30.76 ± 0.42 | Quality-aware design improved stress behavior, but clean performance was mixed. |
| v3 | R3S-MoE | 3 experts, top-1 | 1 seed | **28.72** | Strong historical control with 47,582 parameters. |
| v4 control | R4S-MoE | 4 experts, top-2 | 1 seed | 28.87 | Competitive but slightly worse/slower than v3. |
| v4 lead candidate | R4S-MoE + offline end-to-end RL | 4 experts, top-2 | 1 seed | **28.705** | Best current exploratory row; requires replication. |

## Historical model comparison

| Model | Version / role | Clean MAE (s) | Clean WIS | Parameters | Scope |
|---|---|---:|---:|---:|---|
| Persistence | Operational baseline | 48.62 | 31.82 | — | Static, all runs |
| GRU | Compact temporal baseline | 49.12 | 30.98 | 12,195 | v3 seed 20260908 |
| Dense | Neural reference | 47.19 | 30.09 | 52,291 | v3 seed 20260908 |
| R2S-MoE | Reliability/routing reference | 48.30 | 30.41 | 52,254 | v3 seed 20260908 |
| **R3S-MoE** | **Historical v3 candidate** | **45.17** | **28.72** | **47,582** | **One seed** |
| R4S-MoE control | v4 top-2 | 45.440 | 28.874 | 48,367 | One seed |
| **R4S-MoE end-to-end RL** | **v4 exploratory lead** | **45.191** | **28.705** | **48,367** | **One seed** |

![Historical and v4 benchmark context](v4-clean-benchmark.svg)

## Fresh v4 variant comparison

| Variant | Routing | Clean MAE | Clean WIS | Brier | 90% coverage | CPU p95 |
|---|---|---:|---:|---:|---:|---:|
| Supervised top-1 | 4 / top-1 | 45.192 | 28.774 | 0.02275 | 89.86% | 10.36 ms |
| RL router-only | 4 / top-2 | 45.337 | 28.824 | 0.02277 | 89.81% | 7.50 ms |
| **RL end-to-end** | **4 / top-2** | **45.191** | **28.705** | **0.02269** | 89.80% | 7.05 ms |
| Supervised, no balance | 4 / top-2 | 45.364 | 28.849 | 0.02293 | 89.81% | 8.41 ms |
| Supervised, strong balance | 4 / top-2 | 45.267 | 28.763 | 0.02283 | 89.82% | 9.63 ms |

CPU p95 was measured on a shared machine while runs were launched concurrently. Do not interpret the lower RL timings as a confirmed speedup without repeated isolated timing runs.

## Stress comparison: WIS

![Fresh v4 stress WIS](v4-stress-benchmark.svg)

| Variant | Packet loss | Stale feed | Outage | Combined |
|---|---:|---:|---:|---:|
| Supervised top-1 | 28.965 | 29.909 | 33.729 | 31.277 |
| RL router-only | 29.045 | 30.039 | 33.653 | 31.247 |
| **RL end-to-end** | **28.881** | **29.363** | **33.254** | **30.901** |
| Supervised, no balance | 29.063 | 30.002 | 33.641 | 31.209 |
| Supervised, strong balance | 28.967 | 30.000 | 33.549 | 31.155 |

## Statistical caution

Paired day-bootstrap WIS differences versus supervised top-1 favored end-to-end RL in this test run: clean `-0.098`, packet loss `-0.119`, stale `-0.560`, outage `-0.496`, and combined `-0.376`. These are descriptive one-seed intervals, do not correct for trying multiple variants, and do not establish generalization.

See [the full v4 benchmark](r4s-variant-benchmark.md), [the v3 comparison](v3-comparison.md), and [the source tables](results/README.md).
