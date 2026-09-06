# R3S-MoE v3 comparison

## Short answer

On the completed **single-seed v3 test run**, R3S-MoE was the best clean probabilistic model in this experiment: **28.72 online WIS** and **45.17 s online MAE**. It outperformed the v3 dense reference and v2 R2S-MoE reference on this run, while remaining below the declared resource budget. This is promising, not a proof of superiority: v3 has one seed, while the v2 reference numbers below are two-seed aggregates.

## Easy model comparison

All learned-model values use the **online-calibrated** mode; zero, persistence, and ridge are static-only. These are v3 seed `20260908` results on the chronological 30,000-sample test split.

| Model | Role | Parameters | Clean MAE (s) | Clean WIS | Clean coverage error | Clean severe-delay Brier |
|---|---|---:|---:|---:|---:|---:|
| Persistence | Strong operational baseline | — | 48.62 | 31.82 | 0.0330 | 0.0419 |
| Ridge | Linear baseline | — | 55.99 | 35.67 | 0.0274 | — |
| GRU | Temporal baseline | 12,195 | 49.12 | 30.98 | 0.0010 | — |
| Dense | v3 dense reference | 52,291 | 47.19 | 30.09 | 0.0018 | — |
| R2S-MoE | v2 reliability/routing model | 52,254 | 48.30 | 30.41 | 0.0016 | 0.0238 |
| R2S no-quality | v2 ablation | 52,170 | 48.24 | 30.62 | 0.0019 | — |
| **R3S-MoE** | **v3 candidate** | **47,582** | **45.17** | **28.72** | **0.0016** | **0.0227** |

At the configured 10% alert budget, R3S-MoE’s Brier score was lower than the point-only persistence comparator: **0.0227 vs 0.0419**.

## v1, v2, and v3 headline comparison

| Metric | v1 pilot | v2 aggregate | v3 R3S-MoE |
|---|---:|---:|---:|
| Learned-model seeds | 1 | 2 | 1 |
| Best clean model | Dense, WIS 29.40 | Dense, WIS 30.06 ± 0.33 | **R3S, WIS 28.72** |
| R2S clean online WIS | 30.42 | 30.76 ± 0.42 | 30.41 |
| Best candidate clean MAE | 48.12 s | 48.49 ± 0.97 s for R2S | **45.17 s** |
| Reliability wins across four stress scenarios | 2/4 | 4/4 in both seeds | R3S degradation below v2 R2S reference in 4/4 scenarios; single-seed comparison |
| Outage quality-aware degradation | Catastrophic empty-context failure; WIS 1128.2 in diagnosed subset | 18.99 ± 0.08% | **16.96%** |
| Combined quality-aware degradation | Catastrophic empty-context failure; WIS 1343.8 in diagnosed subset | 10.48 ± 1.79% | **8.63%** |
| CPU forward p95 | 5.88 ms | 5.84 ± 0.04 ms | **9.98 ms** |
| Calibration refresh p95 | Not adjudicated | 0.563 ± 0.005 ms | **0.560 ms** |

The v1 outage/combined values are the diagnosed empty-context subset, not full-scenario WIS values. The v2 values are two-seed mean ± standard deviation; v3 is one seed, so the table is directional rather than a significance test.

## v3 stress behavior

Relative degradation is measured from v3 R3S-MoE’s own clean online WIS of 28.718.

| Test scenario | R3S online WIS | Relative degradation | Plain-English result |
|---|---:|---:|---|
| Clean | 28.72 | 0.00% | Best v3 reference point |
| 15% packet loss | 28.96 | 0.84% | Very small degradation |
| 5-minute staleness | 29.86 | 3.98% | Small degradation |
| 15-minute outage | 33.59 | 16.96% | Moderate degradation |
| Combined corruption | 31.20 | 8.63% | Moderate degradation |

For context, the completed v2 R2S aggregate degraded by 0.45%, 6.68%, 18.99%, and 10.48% in those same scenarios. R3S is lower in this v3 seed for all four scenarios, but this is not yet a controlled multi-seed statistical claim.

## Resource and protocol checks

- Train split: 80,000 samples from January–April 2023.
- Validation split: 20,000 samples from January–February 2024; first half selected the checkpoint and second half fitted calibration/alert thresholds.
- Test split: 30,000 samples from January–February 2025; targets were not used for model selection.
- Parameter count: 47,582, below the 100,000 limit and 8.9% below v2 R2S-MoE.
- Forward p95: 9.98 ms for a 512-sample CPU batch, below the 1-second limit.
- Calibration refresh p95: 0.560 ms, below the 100 ms limit.
- Causal and chronological data checks: passed.
- R3S top-1 routing: used during inference; training evaluates all adapters for differentiable routing.

## Honest conclusion

R3S-MoE is the strongest candidate observed in the current runs, and its design hypotheses are supported by this one v3 seed: persistence anchoring and horizon conditioning coincided with lower clean error, while the dual-path/reliability-gated residual design remained robust under the four tested corruptions. The result still does **not** establish architecture superiority, publication-grade evidence, Japanese validation, passenger outcomes, safety suitability, or absolute novelty. The required next check is at least two more substantive v3 seeds, ideally with official RIDE Gold Lite and additional corruption severities.
