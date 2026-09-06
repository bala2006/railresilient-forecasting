# RailResilient-JP v3 design research

## Decision

The next candidate is **R3S-MoE: Reliability-gated Residual Selective-State Mixture of Experts**. It is a deliberately small, CPU-feasible extension rather than an attempt to copy an LLM at railway scale. The model keeps the existing seven quantiles, eight-event causal context, corruption protocol, validation partition, calibration replay, and test contract unchanged.

### Research findings transferred

| Source finding | Transferable lesson | v3 use | Deliberately excluded |
|---|---|---|---|
| [Bi-Mamba4TS](https://arxiv.org/html/2404.15772v1) reports that selective state-space sequence processing and local patching can complement one another in time-series forecasting. | A short operational history can benefit from separate local mixing and gated state memory. | A two-path local depthwise convolution plus a small input-dependent diagonal state scan. The scan is causal over observed history and uses the observation mask. | Full Mamba kernels, long-context patching, and bidirectional future access. The railway context is only eight events. |
| [DeepSeekMoE](https://arxiv.org/html/2401.06066v1) isolates shared experts and separates routed specialists to reduce redundancy. | A common dynamics path should remain active while specialists model residual regimes. | One always-active shared MLP plus low-rank routed residual adapters. | LLM-scale expert counts and token-level language specialization. |
| [DeepSeek-V3](https://arxiv.org/html/2412.19437v2) emphasizes efficient sparse routing and load balancing, while its training scale and infrastructure are not transferable. | Routing must be measured separately from total parameter count and should not create an overloaded expert. | Straight-through top-1 routing during training, genuine top-1 dispatch at inference, and the existing utilization penalty. | MLA, FP8, auxiliary-loss-free distributed balancing, multi-token language objectives, and RL. |
| [Time-MoE](https://arxiv.org/html/2409.16040v2) shows that sparse experts can be useful for time-series forecasting when capacity and compute are scaled together. | MoE value is an empirical capacity/compute question, not a novelty claim. | The v3 run records parameters, latency, routing, and clean/stressed metrics against dense and v2 artifacts. | Foundation-model pretraining and claims that a small railway model follows scaling laws. |
| [Mamba-ProbTSF](https://arxiv.org/html/2503.10873v1) argues that uncertainty must be modeled alongside state dynamics rather than judged only from point error. | State dynamics and predictive spread need separate, auditable treatment. | A persistence-anchored monotonic residual-quantile head with horizon embeddings; uncertainty remains directly evaluated by WIS and coverage. | Gaussian variance assumptions; the project retains quantile forecasts because they already support WIS, alerts, and calibration. |
| [PatchTST](https://arxiv.org/html/2211.14730v2) demonstrates that simple tokenization choices can be competitive in long-horizon forecasting. | Architectural complexity is not automatically beneficial. | No patching is added: eight event tokens do not provide a meaningful long-patch regime. This is a negative transfer decision. |

Content was rephrased for compliance with licensing restrictions.

## Architecture

1. **Reliability-aware dual path:** event/station embeddings and normalized delay/schedule/mask inputs pass through a mask-gated local depthwise convolution and two small selective diagonal scans. The scan updates are input-dependent and quality-conditioned; unobserved events cannot advance state.
2. **Context fusion:** the model fuses masked pooled local/state features, most-recent-valid state, and the seven quality channels.
3. **Shared plus residual experts:** one always-active MLP models common delay persistence. Three low-rank adapters are routed sparsely and represent residual dynamics, keeping routed capacity cheaper than duplicating full experts.
4. **Reliability gate:** a learned sigmoid gate scales the routed residual using the encoded context and observable feed quality. This is intended to reduce specialist overreaction when the feed is stale or empty.
5. **Persistence-anchored probabilistic head:** each horizon receives a learned embedding and predicts monotonic residual quantiles around the last valid observed delay. This builds in a strong railway forecasting prior while allowing learned corrections and uncertainty widening.

## Falsifiable hypotheses

- **V3-H1:** persistence-anchored horizon conditioning lowers clean online WIS and/or MAE versus v2 R2S-MoE without exceeding the declared CPU budget.
- **V3-H2:** the dual path and reliability gate lower relative WIS degradation in at least three of four corruption scenarios versus the v2 R2S-MoE reference.
- **V3-H3:** the v3 residual head improves severe-delay Brier score over point-only persistence and does not materially worsen calibration.
- **V3-H4:** routed residual adapters provide useful specialization without severe expert collapse; this is measured, not assumed.

A failed hypothesis is a valid result. The v3 run is one new substantive seed unless a second run is explicitly completed; v2 aggregate comparisons use the existing two-seed mean ± standard deviation and are not treated as paired repeated trials.

## Evaluation protocol

- Training: the existing 80,000-sample chronological training split plus the existing randomized corruption curriculum.
- Model selection: first chronological half of the 20,000-sample validation split.
- Calibration and alert threshold: second chronological validation half only.
- Test: one chronological 30,000-sample test split, evaluated after model selection and calibration are frozen.
- Scenarios: clean, 15% loss, five-minute staleness, 15-minute outage, combined corruption.
- Outputs: seven monotonic quantiles for four horizons, static and causal online calibration, WIS, MAE, coverage, alert Brier, routing, and CPU latency.

## Guardrails

The design does not claim novelty because it combines established state-space, residual, quality-gating, and MoE ideas. It does not claim Japanese validity, passenger outcomes, safety suitability, or superiority before the held-out test results are available. All results must be compared with persistence, the v2 dense model, v2 R2S-MoE, and the v2 no-quality ablation under clearly labeled seeds and modes.


## Declared resource budget

Before training, v3 is constrained to **≤100,000 trainable parameters**, **≤2× the v2 R2S-MoE parameter count**, and **<1 second p95 CPU forward latency** for the existing 512-sample benchmark. The existing **<100 ms p95 online calibration refresh** requirement remains unchanged. These are engineering gates, not evidence of accuracy; if v3 exceeds them, the comparison will report the excess rather than silently changing the budget.
