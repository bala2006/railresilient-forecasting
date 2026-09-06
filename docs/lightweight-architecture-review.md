# Lightweight architecture review and proposed model

## Non-negotiable correction: “100% novelty” is not a scientific claim

No researcher can guarantee 100% novelty from web searches. The components requested here already have close precedents:

- shared and routed experts: DeepSeekMoE, DeepSeek-V3, Time-MoE, Moirai-MoE, TimeExpert/TMOE, and MoHE;
- auxiliary-loss-free routing bias: DeepSeek-V3;
- structure-guided expert routing: AME-TS and traffic TESTAM+;
- online/adaptive expert assignment: TFMoE;
- linear/state-space memory: Mamba-family models and Kimi Linear/KDA;
- sparse attention: DeepSeek-V3.2/DSA and Kimi Linear’s discussion of sparse attention.

The defensible claim is therefore not “every component is new.” It is:

> We propose and evaluate a lightweight railway-specific combination of reliability-conditioned routing, an always-on operational-dynamics expert, regime-specialized residual experts, online predictive calibration, and feed-corruption stress testing. The novelty claim is a task/method combination and empirical result, not a guarantee of absolute architectural priority.

A final systematic search across Scopus, Web of Science, IEEE Xplore, ACM Digital Library, TRID, and Google Scholar is required before submission.

## What the recent model families actually contribute

| Family | Relevant mechanism | What transfers to this project | What does not transfer |
|---|---|---|---|
| [DeepSeek-V2](https://arxiv.org/html/2405.04434v3) | Multi-head Latent Attention compresses KV state; DeepSeekMoE uses fine-grained routed experts and shared experts | Shared baseline path, sparse routing, compressed state as design inspiration | 236B-scale model and distributed deployment are irrelevant to a student CPU model |
| [DeepSeek-V3](https://arxiv.org/html/2412.19437v2) | 1 shared expert plus top-k routed experts; sigmoid affinity; routing-only expert bias updated from batch load; small sequence balance loss; multi-token prediction | Shared operational expert, top-1/2 routing, routing-only load-balance bias, multi-horizon loss | FP8, node-limited routing, redundant expert placement, and MTP infrastructure are unnecessary |
| [DeepSeek-V3.2](https://arxiv.org/html/2512.02556) | DeepSeek Sparse Attention: lightweight indexer followed by top-k token selection under MLA | Selective history retrieval could be an optional ablation if a long context is needed | It still requires KV memory and is designed for very long LLM contexts; railway windows are short |
| [DeepSeekMoE](https://arxiv.org/html/2401.06066v1) | Fine-grained expert segmentation and isolated shared experts | Motivates one always-on common railway-dynamics path plus small routed specialists | The language-model scale and token semantics do not apply directly |
| [GLM-4.5](https://arxiv.org/html/2508.06471) | MoE with loss-free balance routing, sigmoid gates, narrow/deep design, GQA/partial RoPE, QK-Norm, MoE MTP | Narrow/deep lightweight backbone, sigmoid router, shared expert, multi-horizon head | Reasoning modes, RL, GQA, and large attention head counts are not needed for delay regression |
| GLM-5/5.2 public model information | Larger MoE and sparse-attention direction; model-card reports integrate DSA-like attention in later GLM releases | Confirms sparse indexing is a major efficiency direction | Use official technical details only when an authoritative report is available; do not copy a model-card claim as a paper result |
| [Kimi K2](https://arxiv.org/html/2507.20534v1) | 1T-total/32B-active MoE and MuonClip optimizer | Confirms active-parameter thinking and optimizer stability are separate design issues | MuonClip is aimed at giant language-model pretraining and should not be a core contribution here |
| [Kimi Linear](https://arxiv.org/html/2510.26692) | Kimi Delta Attention: fine-grained gated delta-rule state; 3 KDA layers to 1 full MLA layer; shared/routed MoE channel mixer | A gated diagonal recurrent memory is a good lightweight online-state baseline; periodic global mixing can be an ablation | Exact KDA, its kernels, 1M-context results, and 3:1 ratio would be borrowed rather than novel |
| [Time-MoE](https://arxiv.org/html/2409.16040v2) | Sparse decoder-only time-series Transformer, shared expert, multi-resolution heads, 50M activated CPU-oriented model | Establishes that time-series MoE and shared experts already exist; use multi-horizon supervision | A railway-specific paper cannot claim “first time-series MoE” |
| [Moirai-MoE](https://arxiv.org/html/2410.10469v1) | Token-level time-series expert specialization and centroid-guided routing | Supports data-driven routing and routing-geometry analysis | Generic token routing is already established |
| [TFMoE](https://arxiv.org/html/2406.03140v1) | Clustered experts for continual traffic forecasting and reconstruction-based gate | Supports online adaptation and expert specialization for evolving transport data | It is road traffic, and its clustering/replay design overlaps with possible railway regime routing |
| [AME-TS](https://arxiv.org/html/2605.25166v1) | Series-level interpretable structural descriptors create a soft prior for token-level router; training-only alignment | Use railway descriptors such as disruption phase, headway pressure, delay trend, and feed reliability as routing prior | Generic structure-guided routing is now recent prior art, so our descriptors must be railway-operational and our evaluation must be different |
| [TimeExpert/TMOE](https://arxiv.org/html/2509.23145v1) | Local temporal experts plus a shared global expert inside attention | Supports local/global memory decomposition and anomaly filtering | Shared-global plus local experts alone is not novel |
| [MoHETS](https://arxiv.org/html/2601.21866v1) | Shared depthwise-convolution continuity expert and routed Fourier experts | Heterogeneous expert functions are a strong lightweight pattern | Shared conv plus Fourier routed experts is already close; do not reproduce it unchanged |
| [TESTAM+](https://arxiv.org/html/2510.07426v1) | Top-1 context routing among identity, adaptive, attention, and topology-aware traffic experts | Motivates top-1 strategic expert selection and latency reporting | Topology-aware traffic MoE is already published/preprinted; railway topology must be evaluated separately |
| [STAMImputer](https://arxiv.org/abs/2506.08054) | Spatiotemporal attention MoE for block-missing traffic data | Supports treating missingness as a first-class transport signal | A railway delay model must not claim generic missing-data MoE novelty |

## Proposed architecture: R2S-MoE

Working name: **Reliability- and Regime-conditioned Shared-Routed Mixture of Experts** (**R2S-MoE**).

The name is provisional and can change after a literature/database search. The architecture is intentionally small and designed for train-event forecasting rather than language generation.

### Design principle

Use one always-active expert for common railway dynamics and route each event through only one small specialist. The router sees both operational regime and feed reliability. A separate online calibrator updates uncertainty from newly observed residuals.

### Data representation

For each active train-stop event at forecast origin `t`, construct an event vector containing:

- current and lagged delay, dwell, headway, and schedule slack;
- route/stop/direction/through-service embeddings;
- sparse predecessor/successor and transfer-edge summaries;
- disruption/status indicators and time-of-day/calendar features;
- observation age, missingness mask, staleness, duplicate, and consistency flags;
- an availability timestamp for every feature.

No passenger identity or individual tracking is used.

### Block 1: railway event encoder

A normalized feature projection maps the event vector to `d = 48` or `64`. Use a small causal depthwise convolution with kernel 3 or 5 over the recent event window, followed by a gated diagonal state update:

```text
z_t = SiLU(W_z x_t)
q_t = sigmoid(W_q [x_t, quality_t])
a_t = sigmoid(W_a [x_t, quality_t])
s_t = a_t ⊙ s_{t-1} + (1 - a_t) ⊙ z_t
h_t = LayerNorm(s_t + W_r x_t)
```

This is a compact recurrent memory, not a claim to invent Mamba or KDA. It allows online state updates without replaying a long history.

### Block 2: shared operational-dynamics expert

The shared expert is always active and captures common dynamics:

```text
h_shared = h + W_2 SiLU(W_1 [h, graph_message, schedule_context])
```

`graph_message` is one sparse, causal aggregation over railway relations: previous/next event on the train, same-platform/headway relation where available, and published transfer relation. Keep the graph operation one-hop and fixed for the first version.

This shared path represents ordinary delay persistence and local propagation. It is the railway analogue of the shared expert idea in DeepSeekMoE/Time-MoE, but its role is explicitly operational and its benefit must be ablated.

### Block 3: three tiny routed specialists

A router selects exactly one of three specialists, while the shared expert remains active:

- **E-normal:** stable timetable/persistence residuals;
- **E-shock:** incident onset, abrupt headway conflict, and severe-delay tails;
- **E-recovery:** post-disruption decay, delay reduction, and asymmetric uncertainty.

Each specialist is a bottleneck residual MLP with hidden width `2d` or `3d`; no specialist contains a full Transformer or GNN. The routed output is:

```text
h_route = E_k(h, regime_features)
h_out = h_shared + g_k h_route
```

The model should also be tested with a fourth **E-quality** specialist only if the three-expert version fails under feed outages. Do not add experts merely to increase capacity.

### Block 4: reliability- and regime-conditioned router

The router input is:

```text
r_t = [pool(h_recent), delay_trend, headway_pressure,
       disruption_phase, observation_age, missingness_rate,
       stale_fraction, source_consistency]
```

Use a low-rank sigmoid router:

```text
logits = W_up SiLU(W_down r_t) + b_route
k = argmax(logits)
```

The bias `b_route` is updated outside backpropagation from recent expert utilization, following the routing-only bias idea in DeepSeek-V3. Compare it against a standard auxiliary balance loss. The router is not allowed to use future incident labels.

The key railway-specific hypothesis is that **the correct expert depends jointly on the physical/operational regime and the freshness of the feed**. This joint conditioning—not MoE by itself—is the proposed mechanism to test.

### Block 5: probabilistic and passenger-risk heads

From `h_out`, use low-rank heads to predict monotonic quantiles `q10`, `q50`, and `q90` for each horizon. Enforce non-crossing through positive increments:

```text
q50 = raw50
q10 = q50 - softplus(raw_low)
q90 = q50 + softplus(raw_high)
```

Optional heads:

- recovery-time hazard or discrete recovery bins;
- transfer-miss probability derived from the feeder arrival distribution and published transfer buffer;
- crowding only when legally usable labels exist.

### Block 6: online calibration state

Maintain a small residual buffer indexed by `(operator/line, forecast horizon, routed regime)` and update it only when a future observation becomes available. Use rolling conformalized quantile residual correction or adaptive residual scale. This is the uncertainty contribution; it is separate from the neural router and must be evaluated independently.

## Preliminary size and compute target

Suggested first configuration:

- hidden width: 64;
- encoder/state layers: 2;
- routed experts: 3;
- top-k routed experts: 1;
- shared expert: 1;
- quantile heads: 3 × 4 horizons;
- no full self-attention in the core model;
- one sparse graph aggregation per time step;
- target: approximately 0.2–1.5 million trainable parameters, to be measured after implementation;
- target: p95 inference under 1 second per network snapshot on CPU, excluding network download;
- target: online update under 100 ms for a snapshot, to be measured rather than assumed.

The active compute is shared expert plus one routed expert. Total parameter count may exceed active parameters, but memory remains important; report both.

## Training objective

```text
L = L_pinball(q10, q50, q90)
  + λ_huber L_huber(delay)
  + λ_rec L_recovery
  + λ_transfer L_transfer
  + λ_cal L_calibration
```

Do not include an auxiliary balance loss in the primary R2S-MoE claim. Instead compare:

1. standard softmax/top-1 routing;
2. auxiliary-loss routing;
3. DeepSeek-style routing-only bias update;
4. oracle regime routing only as a diagnostic upper bound.

If the loss-free update is unstable at small batch sizes, use the standard balance loss and report that result honestly. A technique that works at 671B scale is not automatically best for a small railway dataset.

## Required ablations

- shared expert removed;
- regime features removed;
- reliability features removed;
- online calibration removed;
- fixed expert assignments versus learned router;
- top-1 versus top-2 routing;
- diagonal state versus GRU/TCN baseline;
- graph message removed;
- auxiliary-loss versus routing-only bias;
- clean feed versus delayed/missing feed;
- three experts versus one dense model with equal active parameters.

## What would count as a real contribution

R2S-MoE is worth a paper only if it demonstrates all of the following on held-out incidents:

1. calibration or transfer-alert utility improves over a parameter-matched dense model and persistence baseline;
2. the improvement is concentrated in disruption/recovery or feed-failure regimes, not only normal periods;
3. the router specializes reproducibly, measured by expert utilization, regime-conditioned confusion, and routing stability;
4. the model remains within the latency/memory target;
5. the effect survives incident-level and chronological splits;
6. a Japanese ODPT case study is included when legally and statistically feasible, with RIDE as the open benchmark fallback.

If these conditions fail, R2S-MoE should be reported as a useful engineering prototype, not a novel research architecture.
