# RailResilient-JP v2 redesign and comparison

## Executive result

The v2 redesign fixed the specific v1 failure mechanism. Across two full substantive seeds, quality-aware R2S-MoE no longer catastrophically failed under empty-context, outage, or combined corruption. Its relative WIS degradation was lower than the no-quality ablation in all four stressed scenarios for both seeds.

The redesign did **not** make R2S-MoE the clean probabilistic winner and did not meet the online-calibration WIS threshold. The result supports a stronger engineering design and a more defensible robustness hypothesis, not a publication-ready architecture-superiority claim.

## v1 failure diagnosis

Saved v1 predictions showed:

- 1,986 empty-context outage samples: quality-aware WIS 1,128.2 versus 97.2 for no-quality.
- 442 empty-context combined samples: quality-aware WIS 1,343.8 versus 101.5.
- `delayed_minutes/10` reached 1.5 in outage while training reached only 0.5.
- Inconsistent and duplicate flags were zero throughout v1 training augmentation but active in combined test.
- Catastrophic residuals entered shared online buffers and degraded otherwise ordinary feed states.

Thus the failure was not evidence that reliability information is intrinsically harmful. It was a combination of unbounded/unseen feature extrapolation, unsafe empty-context encoding, and calibration contamination.

## v2 changes

1. Seven bounded quality channels in `[0, 1]`, including explicit `no_fresh_observation`.
2. Saturating age and declared-delay transforms instead of linear extrapolation.
3. Mask-gated convolution/recurrent updates and last-valid residual representation.
4. Randomized per-sample training curriculum covering packet loss, staleness, outages, blackouts, inconsistent values, and duplicates at bounded severities below the test maximum.
5. State-stratified, per-horizon online residual buffers: ordinary, partially missing, and empty/no-fresh states are separated.
6. Genuine top-1 sparse expert dispatch at inference.
7. Dense comparator width tuned to within 1% of total R2S-MoE parameters.
8. Point-only persistence alert comparator and calibration-refresh latency measurements.

## Full v2 results

Values below are mean ± sample standard deviation across seeds `20260906` and `20260907`.

| Metric | v2 result |
|---|---:|
| Dense clean online WIS | **30.06 ± 0.33** |
| R2S-MoE clean online WIS | **30.76 ± 0.42** |
| No-quality R2S clean online WIS | **31.00 ± 0.22** |
| Dense clean online MAE | **46.98 ± 0.15 s** |
| R2S-MoE clean online MAE | **48.49 ± 0.97 s** |
| Online R2S WIS gain over static | **0.37 ± 0.01%** |
| Online R2S coverage-error reduction | **0.20 ± 0.05 percentage points** |
| R2S versus point-persistence clean Brier | **0.0238 vs 0.0419** |
| R2S forward p95 | **5.84 ± 0.04 ms** per 512-sample batch |
| Calibration refresh p95 | **0.563 ± 0.005 ms** |

### Relative stressed WIS degradation from each model's own clean WIS

| Scenario | Quality-aware R2S | No-quality R2S | Interpretation |
|---|---:|---:|---|
| 15% packet loss | **0.45 ± 0.19%** | 2.61 ± 0.46% | v2 quality wins |
| 5-minute staleness | **6.68 ± 4.36%** | 33.21 ± 0.59% | v2 quality wins |
| 15-minute outage | **18.99 ± 0.08%** | 23.94 ± 1.64% | v2 quality wins, smaller margin |
| Combined corruption | **10.48 ± 1.79%** | 18.66 ± 0.21% | v2 quality wins |

The v1 corresponding quality-aware relative degradation was 1.70%, 18.66%, 292.12%, and 78.88% for packet loss, staleness, outage, and combined corruption. The v2 outage and combined failures were therefore reduced from catastrophic to moderate degradation.

## Decision criteria

- **Online calibration:** FAIL. The 0.37% WIS gain and 0.20-point coverage-error reduction are below the 5% / 2-point thresholds.
- **Reliability conditioning:** PASS for this two-seed pilot. Relative stressed degradation was lower in 4/4 scenarios for both seeds.
- **Matched dense clean MAE:** PASS. Dense and R2S-MoE differ by less than 1% in parameter count; R2S clean MAE remains within 5% of dense for both seeds.
- **Efficiency:** PASS for forward and calibration-refresh latency. Peak-RSS and isolated serialized-size characterization remain incomplete.
- **Leakage:** PASS on the prepared arrays: all context observations are at or before origin, targets mature after origin, and splits are chronological.
- **H3 alert utility:** PASS in both seeds: R2S probabilistic Brier is lower than point-only persistence at the validation-frozen alert budget.

## Honest conclusion

V2 is a meaningful improvement over v1. It repairs the observed quality-feature OOD collapse and makes the reliability-conditioning hypothesis reproducibly positive on the selected RIDE pilot. However, the dense comparator remains slightly better on clean WIS, online calibration adds only a small incremental benefit, the experiments still use synthetic corruption and one benchmark family, and the study is only two seeds rather than publication-grade evidence.

The next research step should be three or more seeds, official RIDE Gold Lite comparison, held-out corruption severities, isolated memory/checkpoint measurements, and Japanese ODPT validation only if access and terms permit. No Japanese deployment, passenger-outcome, safety, or absolute novelty claim is established.
