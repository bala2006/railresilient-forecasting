# R4S-MoE: four experts with top-2 routing

R4S-MoE is the proposed v4 architecture for RailResilient. It is an implementation candidate, not a completed benchmark result.

## Why change top-1?

The v3 R3S-MoE design used one shared expert plus three low-rank routed adapters and selected one adapter per event. Top-1 routing is attractive for CPU inference, but it forces mixed regimes into a single specialist. A railway feed can be simultaneously stale, incomplete, and in recovery. Selecting two specialists lets the model combine compatible signals while retaining a sparse path.

This is a hypothesis, not an assumed improvement. The new design must be compared with the historical v3 top-1 result under the same chronological split, corruption scenarios, calibration procedure, and latency measurement.

## Revised design

- **Encoder:** unchanged dual local-mixing and quality-conditioned selective-state encoder.
- **Shared expert:** always active and responsible for common delay dynamics.
- **Routed adapters:** four small low-rank residual specialists.
- **Router:** consumes encoded history, shock/trend/recovery features, and all seven bounded quality channels.
- **Sparse choice:** top two router probabilities are selected and renormalized to sum to one.
- **Reliability gate:** scales the combined routed residual before it is added to the shared path.
- **Forecast head:** unchanged persistence-anchored monotonic seven-quantile head.

The forward path is therefore:

```text
encoded history
      ├── shared expert ─────────────────────┐
      └── top-2 router → two of four adapters ─┼→ reliability gate → quantiles
```

During training, all adapter outputs are materialized for simple, stable gradient flow, but only the selected two contribute to each row. During evaluation, each adapter receives only the rows that selected it, and weighted outputs are accumulated. This preserves the sparse-inference intent.

## Why four experts?

Four specialists create enough capacity to separate the main operational regimes without turning the small CPU model into a large mixture:

1. normal flow;
2. shock or severe delay;
3. stale or incomplete feed;
4. recovery and residual correction.

The router is not forced to learn these exact labels. They are useful design interpretations; the learned assignments still require inspection and evaluation.

## Evaluation gate

No R4S-MoE superiority claim should be made until it has:

1. a newly trained four-expert/top-2 checkpoint;
2. at least the current seed plus additional seeds;
3. clean, packet-loss, staleness, outage, and combined-corruption results;
4. MAE, WIS, severe-delay Brier score, calibration coverage, routing utilization, and CPU p95 latency;
5. comparison with persistence, dense, and historical v3 top-1 R3S-MoE;
6. a check that top-2 routing does not overreact when quality signals are poor.

The public website intentionally continues to use a deterministic fallback unless a matching checkpoint is supplied. It demonstrates the reliability behavior and UI, but it does not fabricate R4S-MoE benchmark results.


## First v4 run

The first matching run used the same selected RIDE Silver months and one seed (`20260908`). It is useful evidence, not a final conclusion:

| Scenario | MAE | Online WIS | Severe-delay Brier | 90% coverage |
|---|---:|---:|---:|---:|
| Clean | 45.44 s | 28.87 | 0.0229 | 89.84% |
| 15% packet loss | 45.72 s | 29.07 | 0.0230 | 89.73% |
| 5 min stale feed | 47.22 s | 30.00 | 0.0237 | 89.80% |
| 15 min outage | 52.01 s | 33.64 | 0.0280 | 89.77% |
| Combined corruption | 48.87 s | 31.22 | 0.0252 | 89.73% |

The candidate has **48,367 parameters** and a measured CPU batch p95 of **12.11 ms**. The historical v3 top-1 result was 45.17 s MAE, 28.72 WIS, 47,582 parameters, and 9.98 ms CPU p95. Therefore, this first top-2 run is competitive but slightly worse and slower than v3 in clean single-seed comparison. It should not replace v3 as the default until additional seeds or routing regularization demonstrate a consistent benefit.

The first run also showed that primary-expert utilization was uneven in clean data, while stress scenarios activated different experts more often. Because those existing report values describe only the primary expert, future runs should report the combined top-2 selected mass as well as primary assignment utilization.
