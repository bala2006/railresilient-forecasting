# R4S-MoE v4 variant benchmark

This benchmark compares five fresh R4S-MoE v4 runs using the same RIDE Silver selected-month protocol, seed (`20260908`), chronological train/validation/test split, corruption curriculum, online calibration, and CPU evaluation procedure. The five runs were launched concurrently with isolated artifact directories.

## Important RL scope

The repository does not contain a railway transition simulator or logged dispatch actions. The two RL variants therefore use an explicitly labeled **offline one-step REINFORCE contextual-bandit** stage: the router selects one of six unordered expert pairs for a context and receives a reward from the already observed four-event forecast horizon. This is not sequential railway-control RL and does not establish a claim about operational policy improvement.

The RL stage starts from a supervised checkpoint. The router-only version freezes the encoder, shared expert, adapters, reliability gate, and quantile head. The end-to-end version fine-tunes all parameters with a retained supervised quantile anchor.

## Clean online results

Lower is better for MAE, WIS, Brier, and latency. Coverage should be close to the nominal 90% target.

| Variant | Experts / routing | Training | MAE (s) | Online WIS | Severe Brier | 90% coverage | CPU p95 (ms) |
|---|---|---|---:|---:|---:|---:|---:|
| Supervised top-1 | 4 / top-1 | Quantile + balance | 45.192 | 28.774 | 0.02275 | 89.86% | 10.36 |
| RL router top-2 | 4 / top-2 | Supervised + router-only offline REINFORCE | 45.337 | 28.824 | 0.02277 | 89.81% | 7.50 |
| RL end-to-end top-2 | 4 / top-2 | Supervised + end-to-end offline REINFORCE | **45.191** | **28.705** | **0.02269** | 89.80% | 7.05 |
| Supervised top-2, no balance | 4 / top-2 | Quantile, balance weight 0 | 45.364 | 28.849 | 0.02293 | 89.81% | 8.41 |
| Supervised top-2, strong balance | 4 / top-2 | Quantile, balance weight 0.05 | 45.267 | 28.763 | 0.02283 | 89.82% | 9.63 |
| Earlier v4 control | 4 / top-2 | Quantile, balance weight 0.01 | 45.440 | 28.874 | 0.02289 | 89.84% | 12.11 |

The earlier v4 control is the previously trained single-seed artifact `r4s-top2-seed20260908`; the five rows above are the new concurrent runs. CPU timings are noisy on a shared machine, so the lower RL p95 values should not be treated as a demonstrated speedup without repeated isolated timing runs.

## Stress scenarios

Online MAE / WIS for the five fresh variants:

| Variant | 15% packet loss | 5-minute stale feed | 15-minute outage | Combined corruption |
|---|---:|---:|---:|---:|
| Supervised top-1 | 45.43 / 28.96 | 46.83 / 29.91 | 52.02 / 33.73 | 48.88 / 31.28 |
| RL router top-2 | 45.64 / 29.05 | 47.32 / 30.04 | 52.06 / 33.65 | 48.91 / 31.25 |
| RL end-to-end top-2 | **45.45 / 28.88** | **46.02 / 29.36** | **51.38 / 33.25** | **48.35 / 30.90** |
| Supervised top-2, no balance | 45.69 / 29.06 | 47.28 / 30.00 | 52.01 / 33.64 | 48.86 / 31.21 |
| Supervised top-2, strong balance | 45.54 / 28.97 | 47.13 / 30.00 | 51.86 / 33.55 | 48.77 / 31.15 |

## Paired comparison against supervised top-1

The following are paired day-bootstrap differences in WIS: candidate minus supervised top-1. Negative is better. These intervals are within this one seed and are not a substitute for multi-seed replication.

| Variant | Clean | Packet loss | Stale feed | Outage | Combined |
|---|---:|---:|---:|---:|---:|
| RL router top-2 | +0.029 [-0.025, +0.079] | +0.052 [-0.008, +0.106] | +0.114 [+0.026, +0.198] | -0.100 [-0.210, -0.004] | -0.015 [-0.087, +0.057] |
| RL end-to-end top-2 | **-0.098 [-0.151, -0.049]** | **-0.119 [-0.174, -0.070]** | **-0.560 [-0.652, -0.476]** | **-0.496 [-0.618, -0.392]** | **-0.376 [-0.442, -0.306]** |
| Supervised top-2, no balance | +0.045 [-0.008, +0.092] | +0.067 [+0.005, +0.122] | +0.071 [-0.020, +0.155] | -0.119 [-0.230, -0.029] | -0.070 [-0.135, -0.008] |
| Supervised top-2, strong balance | -0.045 [-0.099, +0.004] | -0.030 [-0.083, +0.021] | +0.065 [-0.020, +0.145] | -0.217 [-0.315, -0.133] | -0.138 [-0.209, -0.066] |

These intervals are descriptive comparisons on the same test rows. They do not correct for trying several variants, and the end-to-end RL result still requires additional seeds and a stronger offline-policy evaluation design.

## Artifact locations

- `artifacts/r4s-sup-top1-seed20260908`
- `artifacts/r4s-rl-router-top2-seed20260908`
- `artifacts/r4s-rl-e2e-top2-seed20260908`
- `artifacts/r4s-sup-top2-nobalance-seed20260908`
- `artifacts/r4s-sup-top2-strongbalance-seed20260908`

Checkpoints and prediction bundles remain local and are not committed to the repository.

## Conclusion

In this one-seed experiment, end-to-end offline contextual-bandit fine-tuning produced the best aggregate forecast metrics among the five new runs, especially under stale, outage, and combined corruption. The router-only RL stage did not improve the supervised top-1 control. The result is encouraging but exploratory: it may reflect the reward formulation, the supervised anchor, or seed-specific behavior. It should not be called proof that reinforcement learning improves R4S-MoE until the experiment is repeated across seeds and evaluated with a validated sequential simulator or logged-action off-policy protocol.
