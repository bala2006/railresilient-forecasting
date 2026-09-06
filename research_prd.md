# RailResilient-JP Research Product Requirements Document

**Document status:** Implementation baseline
**Version:** 1.0.0
**Frozen:** 2026-09-06
**Primary implementation target:** CPU-feasible reproducible pilot on RIDE Silver
**Proposed model:** R2S-MoE (Reliability- and Regime-conditioned Shared-Routed Mixture of Experts)

## 1. Executive summary

RailResilient-JP is an offline research and decision-support prototype for probabilistic railway delay/recovery forecasting under unreliable observations. It predicts future event delays and calibrated uncertainty, measures degradation under controlled feed loss/delay/staleness, and evaluates whether a compact shared-routed mixture of experts provides value over persistence and parameter-matched dense models.

This implementation is not a train-control, signaling, dispatching, or safety-certification system. RIDE supplies open Belgian operational records, not Japanese live operations. Japan is the target application context; an ODPT adapter is included as a token-gated collection path, but no Japanese empirical claim is permitted until legal access and sufficient historical data are obtained.

## 2. Research objective

Build and evaluate a causal, reproducible pipeline that answers:

> Can a compact probabilistic railway forecaster with explicit feed-quality conditioning and online residual calibration remain more useful than static alternatives when operational observations are delayed, missing, or stale?

### 2.1 Primary hypotheses

- **H1 — calibration:** causal online residual calibration reduces weighted interval score (WIS) and absolute coverage error versus the identical static model.
- **H2 — reliability conditioning:** quality-aware training and routing reduce relative performance degradation under corrupted observations versus an otherwise matched model without quality inputs.
- **H3 — operational value:** probabilistic severe-delay alerts improve Brier score or alert utility over a point-only persistence rule at a validation-frozen alert budget.
- **H4 — regime concentration:** any R2S-MoE advantage is larger in inferred shock/recovery or feed-failure strata than in normal operation.
- **H5 — efficiency:** p95 CPU inference is below one second for a declared batch/snapshot and online calibration update is below 100 ms.

### 2.2 Predeclared practical decision rules

A positive pilot finding requires all of:

1. online calibration improves mean WIS by at least 5% **or** reduces mean absolute interval-coverage error by at least 2 percentage points versus the same static predictions;
2. quality-aware R2S-MoE has lower relative stressed WIS degradation than its no-quality ablation;
3. clean median MAE is no more than 5% worse than the parameter-matched dense model;
4. CPU p95 inference is below one second for the frozen evaluation batch;
5. results contain no future-value or split leakage.

If these criteria are not met, the model is an engineering prototype and the report must state that the research hypothesis was not supported.

## 3. Users and use cases

### 3.1 Intended users

- railway/transport AI researchers;
- passenger-information system researchers;
- human railway operations analysts;
- prospective Japanese university supervisors evaluating a MEXT research plan.

### 3.2 Supported use cases

- next-event probabilistic delay forecasting;
- inferred disruption/recovery risk monitoring;
- severe-delay alert ranking;
- reproducible robustness evaluation under feed corruption;
- CPU latency and memory characterization;
- derived connection-feasibility calculation when a transfer buffer is supplied.

### 3.3 Explicitly unsupported

- autonomous dispatching or train control;
- signaling or safety-critical decisions;
- passenger identity, tracking, biometric analysis, or crowd surveillance;
- claims about observed passenger missed connections without suitable labels;
- claims of Japanese deployment without ODPT validation;
- “first,” “100% novel,” or guaranteed publication claims.

## 4. Scope and staged deliverables

### 4.1 Implemented core

1. RIDE Silver downloader with release manifest and SHA-256 checksums.
2. Causal event-window dataset builder with chronological train/validation/test periods.
3. Deterministic corruption simulator.
4. Persistence, autoregressive ridge, dense probabilistic network, and R2S-MoE models.
5. Monotonic multi-quantile outputs for q05/q10/q25/q50/q75/q90/q95.
6. Rolling causal quantile residual calibration.
7. Point, probabilistic, alert, routing, and systems metrics.
8. Derived connection-feasibility utility.
9. ODPT endpoint/config adapter that refuses collection without a token.
10. Reproducible experiment CLI and JSON/CSV/Markdown artifacts.

### 4.2 Pilot experiment scope

The substantive v2 pilot uses selected public RIDE Silver monthly event files:

- training: January–April 2023;
- validation: January–February 2024;
- test: January–February 2025.

The builder creates causal windows within each `(train_id, service_date)` journey. It uses a fixed past context and predicts the next four event delays. For CPU feasibility and to prevent long journeys from dominating, it selects exactly one uniformly random valid anchor per eligible journey before applying split caps. The resulting estimand is journey-representative, not uniform over every possible event window. Training caps may enrich severe/recovery cases; validation and test caps remain outcome-blind. This is a **RIDE Silver pilot**, not a reproduction of the official RIDE Gold benchmark.

### 4.3 Conditional extensions

- official RIDE Gold Lite reproduction;
- graph/event model using full RIDE topology;
- full required corruption grid and three-seed final experiment;
- ODPT Japanese static/live case study;
- transfer-feasibility labels based on legally usable transfer tables;
- external GPU/full-scale training.

## 5. Data requirements

### 5.1 Source and license

- Source release: [RIDE Silver](https://huggingface.co/datasets/orailix/ride-silver).
- Code reference: [orailix/ride](https://github.com/orailix/ride).
- RIDE dataset license: CC BY 4.0.
- Required attribution: RIDE, Infrabel, and Open-Meteo.
- Project outputs must not imply endorsement.

### 5.2 Event fields

Required RIDE fields:

- `train_id`, `service_date`, `op_id`, `event_type`;
- `planned_ts`, `observed_ts`, `delay_sec`;
- optional `arr_line_id`, `dep_line_id`.

### 5.3 Internal sample contract

Each sample contains:

- `forecast_origin`: observed timestamp of the last available context event;
- `past_delay`: past delay sequence in seconds;
- `past_planned_delta`: schedule offsets relative to origin;
- `past_event_type`: categorical event-type sequence;
- `past_station_bucket`: deterministic station hash bucket;
- `quality`: seven bounded `[0, 1]` channels: missing fraction, stale fraction, saturating observation age, saturating declared delay, inconsistency flag, duplicate flag, and no-fresh-observation indicator;
- `regime`: normal, shock, or recovery, derived only from available context;
- `future_planned_delta`: known schedule offsets for future events;
- `target_delay`: next four observed event delays;
- `target_mask`: valid future event indicators.

### 5.4 Leakage rules

1. Context uses only events at or before the forecast origin.
2. Regime uses only past/current delay trend.
3. Corruption is applied to context before quality features and model input are constructed.
4. Targets never enter input or router features.
5. Journeys do not cross chronological splits.
6. Validation selects thresholds/configuration; test is scored once per frozen run.
7. Online calibrator updates only after the corresponding target is considered matured in chronological order.
8. Random seeds for train augmentation and test corruption are distinct.

### 5.5 Missing and unreliable observation semantics

Loss, delay, staleness, duplicates, and timestamp inconsistency operate on the context sequence. A removed or stale observation is represented only by causally available retained values, an observation mask, and bounded quality channels; an original hidden value is never retained as a feature. Observation age and declared delay use saturating transforms, and `no_fresh_observation` explicitly identifies an empty/blackout context. The encoder gates convolutional and recurrent state updates by the observation mask, and online calibration buffers are separated by feed-state class and forecast horizon.

### 5.6 ODPT gate

The ODPT collector requires `ODPT_ACCESS_TOKEN`. It must:

- refuse unauthenticated requests;
- store receipt time and content hash;
- preserve raw snapshots privately;
- respect polling, retention, attribution, and redistribution terms;
- never add credentials to files or logs.

## 6. Prediction tasks

### 6.1 Primary task

Predict absolute delay in seconds at the next four operational events.

### 6.2 Quantiles

Required quantiles:

`[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]`

This supports 50%, 80%, and 90% central intervals. Quantiles must be monotonic by construction or sorting.

### 6.3 Inferred operational regime

Based only on past context:

- `normal`: small absolute delay and small trend;
- `shock`: increasing trend or severe current delay;
- `recovery`: positive delay with a decreasing trend.

Thresholds are configuration constants, not test-tuned values.

### 6.4 Severe-delay alert

Binary label: target delay exceeds a validation-frozen threshold (default 300 seconds). Probability is estimated from predictive quantiles by piecewise-linear CDF interpolation. Alert thresholds are chosen on validation at a fixed alert fraction.

### 6.5 Connection feasibility

When a connection departure and transfer buffer are supplied:

`P(miss) = P(feeder_arrival_delay > connection_buffer_seconds)`

In the RIDE pilot this is a derived score only; no passenger behavior claim is allowed.

## 7. Corruption protocol

The simulator supports:

- packet loss: 5%, 15%, 30%;
- contiguous outage equivalents: 5, 15, 30 minutes;
- delayed/stale updates: 1, 3, 5, 10 minutes;
- station/source outage;
- duplicate indicator;
- timestamp inconsistency indicator;
- combined outage plus disruption.

Every scenario writes a manifest with name, parameters, seed, sample count, and affected-context count. The substantive v2 pilot runs clean, packet-loss-15%, stale-5-minutes, outage-15-minutes, and combined scenarios. Training uses randomized per-sample corruption across bounded loss, staleness, outage, blackout, inconsistency, and duplicate severities; the training curriculum is distinct from the held-out test scenarios. Quality channels are bounded to `[0, 1]`, with saturating age/delay transforms and an explicit no-fresh-observation state.

## 8. Models

### 8.1 B0 zero schedule

Predict zero seconds for all horizons.

### 8.2 B1 persistence

Predict the last causally available delay for all horizons. Under corruption, use only the last retained observation.

### 8.3 B2 autoregressive ridge

Flatten past delays, masks, known future schedule offsets, and compact context features. Fit one regularized regression per horizon. Estimate empirical residual quantiles on validation.

### 8.4 B3 parameter-matched dense probabilistic model

Use the same event encoder and multi-quantile heads as R2S-MoE, replacing experts/router with a dense residual MLP of comparable active parameters.

### 8.5 B4 compact GRU

Encode the past event sequence with a small GRU and predict all horizon quantiles. It provides a standard temporal neural baseline.

### 8.6 B5 R2S-MoE

#### Encoder

- continuous projection plus event/station embeddings;
- causal depthwise convolution;
- quality-aware gated diagonal recurrent state;
- hidden width 48 or 64.

#### Shared expert

An always-active residual MLP captures common persistence and operational dynamics.

#### Specialists

Three bottleneck residual experts represent data-driven normal/shock/recovery specializations. Names are hypotheses; specialization must be measured.

#### Router

Input combines encoded context, available delay trend/regime indicators, and quality features. Training uses soft mixture routing with temperature; inference uses straight-through/top-1 routing. An auxiliary utilization penalty is configurable. No future regime label is used.

#### Heads

Seven monotonic quantiles for each of four future events. Optional losses are disabled unless labels exist.

#### Online calibrator

Separate rolling residual state keyed by horizon with fallback to a global buffer. The pilot avoids sparse line/regime keys unless support is sufficient.

## 9. Training

### 9.1 Loss

Mean pinball loss across valid quantiles/horizons plus:

- optional median Huber loss;
- small expert-balance penalty;
- optional quantile-spacing regularizer.

Recovery, transfer, and crowding losses default to disabled.

### 9.2 Optimization

- AdamW;
- validation-based early stopping;
- gradient clipping;
- deterministic seeds where supported;
- CPU thread count recorded;
- model checkpoints contain configuration and feature schema.

### 9.3 Sampling

Sampling is deterministic and stratified to preserve severe-delay and recovery examples. Limits are declared in the experiment config. Test examples remain chronologically ordered for calibration.

## 10. Evaluation

### 10.1 Point metrics

- MAE;
- RMSE;
- MASE using persistence MAE as denominator;
- per-horizon MAE;
- normal/shock/recovery MAE.

### 10.2 Probabilistic metrics

- mean pinball loss;
- empirical 50%, 80%, and 90% interval coverage;
- interval widths;
- weighted interval score;
- quantile-CRPS approximation, explicitly labeled approximate;
- absolute coverage error.

### 10.3 Alert metrics

- Brier score;
- log loss;
- precision, recall, F1, and AUPRC;
- precision at fixed alert budget.

### 10.4 Routing metrics

- expert utilization;
- utilization by inferred regime;
- router entropy;
- routing stability across corruption scenarios where sample identity is preserved.

### 10.5 Systems metrics

- parameter count and serialized model size;
- p50/p95 inference latency after warm-up;
- calibration update p50/p95 latency;
- throughput;
- peak process RSS where available;
- CPU count, thread count, batch size, and sample count.

### 10.6 Statistical reporting

Pilot reports paired day-block bootstrap confidence intervals where enough days exist. The completed v2 evidence contains two substantive learned-model seeds (`20260906`, `20260907`) and reports their mean ± standard deviation in `docs/v2-redesign-comparison.md`; each per-seed artifact remains a single-seed run. Publication-grade results require at least three seeds.

## 11. Experiment matrix

Minimum pilot matrix:

| Model | Clean | Loss 15% | Stale 5m | Outage 15m | Combined | Online calibration |
|---|---:|---:|---:|---:|---:|---:|
| Zero | yes | yes | yes | yes | yes | no |
| Persistence | yes | yes | yes | yes | yes | optional residual intervals |
| Ridge | yes | yes | yes | yes | yes | static |
| Dense | yes | yes | yes | yes | yes | static + online |
| GRU | clean minimum | optional | optional | optional | optional | static |
| R2S-MoE | yes | yes | yes | yes | yes | static + online |
| R2S without quality | yes | yes | yes | yes | yes | static + online |

Publication-grade expansion adds all corruption severities, three seeds, topology/graph baseline, shared-expert/router ablations, and official Gold evaluation.

## 12. CLI and artifacts

Required commands:

```bash
python -m railresilient.cli download
python -m railresilient.cli prepare
python -m railresilient.cli train
python -m railresilient.cli evaluate
python -m railresilient.cli run-all
python -m railresilient.cli odpt-check
```

Each run writes:

- resolved configuration;
- environment and data manifest;
- source checksums;
- model checkpoints;
- prediction arrays;
- scenario manifests;
- CSV/JSON metric tables;
- latency and routing summaries;
- `final_findings.md`;
- `data_card.md` and `model_card.md`.

## 13. Non-functional requirements

- Python 3.11;
- CPU-first execution;
- deterministic fixed seeds;
- typed public interfaces;
- streaming/memory-mapped data where needed;
- no secrets committed or printed;
- no raw ODPT redistribution;
- no safety-critical deployment language;
- complete run must fail clearly on missing data or invalid configuration.

## 14. Acceptance criteria

### 14.1 Engineering completion

- [x] Fresh environment installs from lock/configuration.
- [x] Public RIDE files download and verify.
- [x] Dataset builder produces documented arrays without future leakage.
- [x] All baseline and R2S-MoE models train and infer.
- [x] Corruption scenarios are deterministic.
- [x] Static and online probabilistic metrics are produced.
- [x] CPU latency and parameter-count model-size reports are produced; isolated serialized-size/RSS benchmarking remains incomplete.
- [x] Findings distinguish measured results from unavailable claims.

### 14.2 Research completion

- [x] Persistence remains in every comparison.
- [x] Dense model parameter count is within 1% of R2S-MoE; training and inference compute are not otherwise identical.
- [x] Quality-feature and online-calibration ablations are evaluated.
- [x] Results are stratified by horizon and inferred regime.
- [x] Negative results are retained.
- [x] Japan, transfer, and novelty limitations are explicit.

## 15. Falsification and stopping conditions

Stop or narrow the claim if:

- public data cannot be reproduced;
- online calibration does not improve WIS or coverage utility;
- quality conditioning does not reduce stress degradation;
- persistence equals or beats learned models without compensating probabilistic value;
- R2S-MoE gains disappear against a matched dense model;
- latency misses the target by an order of magnitude;
- inferred regimes are too unstable to support regime claims;
- transfer outputs lack defensible labels;
- ODPT access/history remains unavailable.

## 16. Publication and novelty wording

Until a final database search and supervisor review, use:

> Our review did not identify a study evaluating this complete combination of compact reliability-conditioned railway expert routing, causal online uncertainty calibration, and controlled delayed/missing-feed stress testing. We position the contribution as a reproducible task, system, and evaluation protocol; its individual architectural components have substantial prior art.

## 17. Completion definition

The current implementation is complete when the CPU-feasible RIDE Silver pilot can be reproduced end-to-end and produces auditable artifacts and honest findings. It is not publication-final until the official RIDE benchmark, full scenario grid, three seeds, incident/day confidence intervals, and—if legally feasible—an ODPT Japanese case study are completed.


## V2 redesign amendment

The first pilot exposed three implementation-level failure modes: unbounded quality-feature extrapolation, unsafe empty-context recurrent encoding, and cross-state online-calibration contamination. The v2 implementation therefore uses seven bounded quality channels, explicit no-fresh-observation state, mask-gated state transitions, randomized per-sample corruption curriculum, state-stratified per-horizon residual buffers, sparse top-1 inference dispatch, and a dense comparator within 1% of R2S-MoE parameter count. A point-only persistence alert comparator and calibration-refresh latency measurement were also added.

The v2 decision remains conservative: the reliability-conditioning criterion is supported in a two-seed RIDE Silver pilot, the H3 Brier comparison is supported, and the clean matched-MAE/latency checks pass. Online calibration does not meet the predeclared WIS or coverage-error improvement threshold. V2 does not establish Japanese validation, passenger outcomes, safety suitability, publication-grade multi-seed evidence, or absolute architectural novelty.



## V3 architecture experiment amendment

A v3 candidate, R3S-MoE (Reliability-gated Residual Selective-State Mixture of Experts), was implemented after the v1/v2 evidence. It combines a mask-gated local convolution with quality-conditioned selective state scans, a shared MLP plus low-rank top-1 routed residual adapters, a reliability gate, and persistence-anchored horizon-conditioned monotonic quantiles. The design is documented with literature-transfer decisions in `docs/v3-design-research.md`.

The completed v3 seed `20260908` used the same causal RIDE Silver train/validation/test protocol. R3S-MoE achieved clean online WIS 28.72 and MAE 45.17 seconds, with 47,582 trainable parameters, 9.98 ms p95 CPU forward latency per 512-sample batch, and 0.560 ms p95 calibration refresh. Its relative WIS degradation was 0.84%, 3.98%, 16.96%, and 8.63% for packet loss, staleness, outage, and combined corruption. These are promising single-seed results; they do not establish architecture superiority, publication-grade evidence, Japanese validation, passenger outcomes, safety suitability, or absolute novelty. At least two additional substantive v3 seeds and official benchmark validation remain required.
