# Experiment protocol, roadmap, and application positioning

## Evaluation design

### Splits

1. **Chronological split:** train on earlier dates, validate on later dates, test on the final period.
2. **Incident-level split:** all snapshots belonging to one disruption remain in one split.
3. **Operator/line holdout where possible:** test transfer to an unseen route, line, or operator only after the within-network result is established.
4. **Japanese live collection:** use forward-chaining evaluation; never backfill a forecast with later-corrected feed records.

Report clean, normal, disruption, recovery, severe-delay, and feed-failure regimes separately. A single aggregate MAE is not sufficient.

### Baselines and ablations

Required baselines:

- schedule/zero-delay;
- translation/persistence;
- Markov/autoregressive statistical baseline;
- quantile XGBoost or LightGBM;
- LSTM/TCN or another compact temporal baseline;
- graph/event model;
- static version of the selected online model.

Required ablations:

- remove online update;
- remove uncertainty calibration;
- remove observation-age and missingness features;
- remove graph/topology features;
- remove disruption/status features where available;
- weather on/off;
- transfer-risk head on/off;
- corruption stress on/off.

### Metrics

#### Delay/recovery point metrics

- MAE, RMSE, and MASE against a naive seasonal/translation reference;
- error by horizon, delay magnitude, operator/line, and regime;
- recovery-time absolute error and bias;
- propagation error by run/dwell/headway relation if the data support event edges.

#### Probabilistic metrics

- CRPS for full predictive samples/distributions;
- pinball loss for quantiles;
- empirical coverage of 50%, 80%, and 90% intervals;
- interval width and weighted interval score;
- calibration plots and expected calibration error for risk bins;
- tail calibration for severe-delay events.

#### Passenger-risk and alert metrics

- Brier score and log loss for transfer-miss probability;
- precision, recall, F1, and area under the precision-recall curve for severe-delay/transfer-risk alerts;
- precision at a fixed alert budget;
- false alerts per operating hour;
- median and 90th-percentile lead time before a missed connection or threshold crossing;
- expected waiting-time proxy only if its assumptions are explicitly stated.

A transfer-risk output computed from schedule and predicted arrival is a feasibility-risk estimate, not an observed passenger outcome. State this distinction in the paper.

#### Systems metrics

- p50/p95 inference latency per network snapshot;
- online update latency;
- throughput in snapshots per minute;
- peak RAM and model size;
- CPU-only and optional modest-GPU measurements;
- failure behavior when no fresh feed is available.

### Statistical reporting

- Use fixed seeds and record environment/package lock files.
- Report confidence intervals by incident or day block, not only by row bootstrap.
- Compare paired forecasts at the same forecast origins.
- Avoid tuning alert thresholds on the test set.
- Run at least three seeds for learned models where compute permits; use deterministic baselines.
- Include a model card and a data card.

## Implementation milestones

### Phase 0 — access and specification (1–2 weeks)

- create the repository and environment lock;
- download or obtain RIDE release and verify license/checksum;
- request/verify ODPT access and terms;
- freeze the schema, forecast-origin semantics, target definitions, and leakage rules;
- produce `data_card.md` and an access decision log.

**Exit:** a legal, reproducible source for the core task and a tiny sample pipeline.

### Phase 1 — data pipeline and audit (2–3 weeks)

- parse static GTFS/GTFS-JP and GTFS-RT;
- normalize timestamps and identifiers;
- construct event, station, trip, transfer, and alert tables;
- implement incident/recovery labeling;
- implement corruption simulator and unit-level data validation;
- visualize missingness and feed age.

**Exit:** one leakage-tested dataset builder producing versioned parquet outputs.

### Phase 2 — baselines (2–3 weeks)

- implement schedule, persistence, Markov, and quantile boosting baselines;
- add chronological/incident-level evaluation;
- establish clean benchmark scores and latency measurements.

**Exit:** baseline table that a future model must beat or explain.

### Phase 3 — online probabilistic layer (3–4 weeks)

- implement quantile/distributional base model;
- implement rolling calibration/update;
- add feed-quality features and outage handling;
- test calibration before any neural graph model.

**Exit:** reproducible answer to H1–H3 on the offline benchmark.

### Phase 4 — compact graph/temporal comparison (3–4 weeks)

- add one graph/event model only if it addresses a measured baseline weakness;
- compare accuracy, calibration, robustness, and latency—not architecture novelty;
- perform all planned ablations.

**Exit:** final model selection based on operational utility, not leaderboard MAE alone.

### Phase 5 — Japan case study (4–8 weeks, access-dependent)

- collect or replay permitted ODPT snapshots;
- evaluate Tokyo Metro/Toei/JR East topology and through-service behavior where available;
- report sample count, number of disruptions, and confidence intervals;
- if insufficient, publish a transparent qualitative case study and retain RIDE as the quantitative benchmark.

**Exit:** evidence-based Japanese relevance statement.

### Phase 6 — paper and application package (3–4 weeks)

- freeze experiments and release code/configuration;
- write limitations and data-access statement;
- submit a workshop/transportation-AI paper or use the technical report for supervisor outreach;
- tailor a two-page MEXT research plan to specific laboratories.

## Publication positioning

Possible paper framing:

> **A robust real-time evaluation protocol for online probabilistic railway disruption-recovery forecasting under unreliable operational feeds.**

The paper’s claim is about task design, online calibration, feed-quality robustness, and passenger-risk utility. It is not “we applied a newer neural architecture to predict train delay.”

Minimum publishable package:

- open benchmark reproduction or clear data release/collection protocol;
- persistence and strong ML baselines;
- chronological plus incident-level evaluation;
- probabilistic calibration;
- feed-latency/missingness stress tests;
- latency and failure-mode measurements;
- honest Japanese case study or a documented reason it could not be completed.

## MEXT/Japan positioning

The topic aligns with Japanese railway priorities without promising unsafe autonomy:

- RTRI’s current direction emphasizes sustainable railway systems, resilience, labor constraints, digital operations, and passenger-flow support; see [RTRI Research 2030](https://www.rtri.or.jp/assets/edga9q000000092c-att/RESEARCH2030RTRI4.pdf).
- RTRI has published Japanese work on delay prediction and congestion prediction during disruptions, which supports relevance while ruling out generic “Japanese railway delay LSTM” novelty.
- ODPT makes a public-data pathway plausible through Tokyo Metro and other operator feeds, subject to access and terms.
- The work complements labor-saving and resilient operations by giving humans calibrated early warnings; it does not claim autonomous dispatch or control.

For the MEXT research plan, state:

1. the societal problem: disruption uncertainty and passenger connection loss;
2. the Japanese fit: dense urban rail, through-service, resilience, labor constraints, and public-data ecosystem;
3. the research gap: online calibration and feed-reliability evaluation are under-tested together;
4. the method: leakage-safe probabilistic forecasting plus alert policy;
5. the outcome: reproducible software, benchmark protocol, and operator-oriented evidence;
6. the host fit: select supervisors in railway systems, transportation engineering, AI/ML, or operations research whose lab work matches the data and evaluation plan.

The official [MEXT 2027 research-student page](https://studyinjapan.go.jp/en/smap-stopj-applications-research.html) says embassy-recommendation recruitment for April or September/October 2027 generally occurs in April–May of the preceding year and that country-specific schedules must be checked with the relevant Japanese embassy. Confirm the India-specific deadline independently; do not rely on third-party application guides.

## Final novelty wording

Use this wording in drafts until a final database search:

> “Our review did not identify an existing study that evaluates continuously updated, uncertainty-calibrated railway disruption-recovery forecasts and passenger transfer-risk alerts under controlled delayed/missing public real-time feeds in a Japanese urban-rail setting. Existing work covers important subsets of this problem, including adaptive event-time prediction, probabilistic post-disruption modeling, graph-based delay propagation, and Japanese congestion forecasting. We therefore position the contribution as a reproducible evaluation and decision-support protocol rather than as a novel neural architecture.”

This wording is defensible, specific, and does not guarantee priority.
