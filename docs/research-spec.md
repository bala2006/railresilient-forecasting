# Research specification

## Working title

**RailResilient-JP: Online uncertainty-calibrated forecasting of railway disruption recovery and passenger transfer risk under delayed or missing real-time feeds**

Short title: **Online calibrated railway recovery forecasting under feed uncertainty**.

## Problem statement

Urban railway operations expose two related but different risks:

- operators need to know how an observed delay will evolve over the next several stops or time horizons; and
- passengers need to know whether a planned connection remains feasible.

Most published prediction systems optimize a point error on a clean historical feed. That is insufficient for a real-time decision-support tool because a live feed can arrive late, omit a vehicle, report an inconsistent timestamp, or change as a disruption recovers. A useful system must provide a calibrated range or distribution, update online, and expose when its input quality is poor.

## Primary research question

**RQ1.** Can a compact online-updated probabilistic forecaster provide more reliable disruption-recovery and transfer-risk alerts than static or point-forecast baselines when railway observations are delayed, missing, or inconsistent?

## Secondary research questions

- **RQ2:** Does online recalibration improve post-disruption calibration and recovery-time accuracy compared with a model trained once and left static?
- **RQ3:** Do explicit observation-age, missingness, and feed-quality features reduce performance degradation under realistic feed failures?
- **RQ4:** Can a forecast distribution be converted into a useful transfer-risk alert with controllable false-alarm rate and measurable lead time?
- **RQ5:** Can the end-to-end inference and update loop meet a practical CPU latency budget for a small urban network?

## Exact prediction targets

At each observation time `t`, for each active train/event and selected downstream station or time horizon, predict:

1. **Delay change** at horizons of 15, 30, 45, and 60 minutes, or the next 1–4 operational events where the source supports event-level targets.
2. **Delay distribution**, represented by quantiles (at minimum 0.1, 0.5, 0.9) and a calibrated interval; a sampled distribution may be added for CRPS.
3. **Recovery time**, defined before evaluation as the first future time at which the upper forecast threshold and observed delay remain below the operational threshold for a sustained window. The threshold and window must be fixed without test-set tuning.
4. **Transfer-miss probability** for a scheduled connection: the probability that predicted feeder arrival plus the published transfer buffer exceeds the connecting departure time. This is a derived passenger-risk output, not a claim to observe every passenger journey.
5. **Alert state:** no alert, watch, or high-risk alert, selected from forecast probabilities and evaluated at a fixed alert budget.

Crowding is a **stretch target only**. If a legally usable, time-aligned Japanese crowding label is acquired, add a multi-task crowding head and evaluate it separately. Do not make the core paper depend on SURCONFORT data access.

## Proposed contribution

The contribution is a reproducible task and evaluation protocol, implemented as a compact reference system:

1. **Online probabilistic update layer.** A base forecaster produces point and quantile predictions. A rolling calibration/update component incorporates newly observed residuals during the day or disruption episode without retraining the whole model.
2. **Feed-quality representation.** Every live observation carries an age-of-observation value, missingness mask, source timestamp, receipt timestamp, consistency flags, and last-valid-observation age. The model and the alert layer can widen or suppress alerts when information quality is poor.
3. **Passenger-risk translation.** Predictive arrival distributions are propagated through published transfer rules to produce missed-connection probabilities and lead-time/false-alarm trade-offs.
4. **Robust real-time evaluation.** The same models are evaluated on clean data and controlled, reproducible corruptions: delayed packets, missing trains, missing stations, burst outages, stale updates, and inconsistent timestamps.
5. **Japan-oriented validation path.** The target deployment setting uses Japanese public GTFS/GTFS-Realtime and service-alert feeds where terms permit, with RIDE as the mandatory open benchmark fallback.

The paper should not claim that a new GNN, Transformer, or attention mechanism is novel. Any graph model is an implementation baseline or ablation unless experiments demonstrate a separate methodological contribution.

## Candidate system design

The proposed lightweight architecture is **R2S-MoE: Reliability- and Regime-conditioned Shared-Routed Mixture of Experts**. It is a research hypothesis and working name, not a guaranteed-new architecture. The detailed design and novelty audit are in [`lightweight-architecture-review.md`](lightweight-architecture-review.md).

R2S-MoE uses a compact gated diagonal recurrent state, one always-active shared operational-dynamics expert, and one top-1 routed specialist selected from normal, shock, and recovery experts. The router conditions on both railway regime features and feed-quality features. A separate rolling conformal/adaptive residual calibrator updates uncertainty online. The core model deliberately avoids a large Transformer and targets approximately 0.2–1.5M trainable parameters, subject to measurement.

The architecture claim is not that MoE, shared experts, state-space memory, or sparse routing are individually new. Recent DeepSeek, GLM, Kimi, Time-MoE, Moirai-MoE, AME-TS, TimeExpert, MoHETS, TFMoE, and TESTAM+ work already provides close precedents. The potentially novel unit is the tested railway-specific combination of reliability-conditioned routing, operational shared dynamics, disruption-phase specialists, online calibration, and feed-corruption evaluation.

### Inputs

- static schedule: stops, stop times, trips, routes, calendars, transfers, service dates;
- observed vehicle/trip updates: event time, delay, stop, trip, route, direction, status;
- service alerts/disruption metadata when available;
- topology: consecutive-stop and transfer relationships;
- temporal context: time of day, weekday, holiday, peak/off-peak;
- quality metadata: receipt delay, missingness, staleness, duplicate/inconsistent record flags;
- optional weather only after an ablation justifies it; weather must not be assumed useful.

### Reference model ladder

- **B0:** schedule-only and zero-delay baseline.
- **B1:** translation/persistence: last valid delay propagated forward.
- **B2:** Markov or autoregressive statistical model.
- **B3:** XGBoost/LightGBM quantile models with lag, schedule, topology, and quality features.
- **B4:** compact temporal/graph model, such as an event graph encoder with recurrent or temporal convolutional heads.
- **B5:** R2S-MoE with shared operational expert, top-1 regime specialist, reliability-conditioned routing, and online calibration.
- **B6:** optional KDA/Mamba-style or small attention model as an efficiency ablation, not the main novelty claim.

The main comparison is B5 versus B1–B4. At least one lightweight statistical model must remain in the final paper because prior work repeatedly shows that persistence can beat complex models on common short-horizon cases.

### Research-use decision on recent LLM architectures

- **Adopt:** the DeepSeek/Time-MoE shared-expert pattern, DeepSeek-V3-style routing-only balancing as an ablation, GLM’s narrow/deep efficiency principle, and Kimi’s gated recurrent-memory idea as a baseline inspiration.
- **Do not adopt as the main model:** full MLA, DeepSeek Sparse Attention, Kimi K2’s MuonClip, GLM hybrid reasoning/RL, FP8 distributed training, or LLM-scale MTP infrastructure. These solve different scale and sequence problems and would add complexity without a justified railway benefit.
- **MTP adaptation:** predict several railway horizons jointly through multi-horizon quantile heads. Call this multi-horizon supervision, not a novel MTP objective, unless a separate causal training benefit is demonstrated.

### Online update candidates

Implement one primary method and one ablation, selected after the access audit:

- rolling-window quantile recalibration or conformalized quantile residual correction;
- adaptive residual scale/offset update using only observations available by the forecast origin;
- optional small learning-rate online fine-tuning only if it can be proven leakage-safe and latency-safe.

The first paper should favor transparent calibration over continual full-model retraining.

## Hypotheses

- **H1:** Online updating reduces post-disruption MAE/CRPS and improves interval coverage compared with an identical static model.
- **H2:** Explicit feed-age and missingness features reduce degradation under delayed/missing-feed stress tests.
- **H3:** The probabilistic system provides better-calibrated severe-delay and transfer-miss alerts than point forecasts at the same alert budget.
- **H4:** The largest gains occur in disruption and recovery windows, not in ordinary low-delay periods; this must be reported by regime.
- **H5:** The reference pipeline meets a predeclared p95 inference latency target of 1 second per network snapshot on CPU for a small urban network, excluding network download time.
- **H6, optional:** A crowding head improves passenger-risk ranking only when labels are sufficiently dense and quality-controlled; otherwise it must be reported as inconclusive.

## Scope boundaries

Included: forecasting, uncertainty calibration, data-quality robustness, transfer-risk decision support, reproducibility, and CPU/modest-GPU deployment measurements.

Excluded: autonomous train control, signaling intervention, timetable optimization as the primary task, reinforcement learning for real trains, facial/biometric sensing, passenger identity tracking, and claims of safety certification.

## Data leakage rules

- Features must be timestamped by **availability at the forecast origin**, not by the event’s eventual corrected timestamp.
- Future service alerts, corrected historical feed records, and final delay labels may only enter after the corresponding forecast origin.
- Splits must be chronological and incident-aware. Snapshots from one disruption cannot be split across train/event rows in a way that leaks the incident signature.
- Online calibration windows must contain only observations that would have arrived before the prediction being scored.
- Static topology may be built from data published before the evaluation period; future schedule revisions must be versioned.

## Falsification criteria

The project should be considered unsuccessful as a research contribution if any of the following holds:

- the online layer does not improve calibration or operational utility over the static model across held-out incidents;
- the result is only a tiny average MAE improvement caused by leakage or a single easy operator;
- no public/legal data path is available and the study cannot be reproduced from released data;
- the proposed system misses the latency target by an order of magnitude without a clear accuracy/utility justification;
- transfer-risk labels are too synthetic to support a passenger claim. In that case, report them as derived feasibility scores, not observed passenger outcomes.
