# Data-access audit and fallback plan

## Decision rule

No model development should begin until each dataset is assigned a source, license/terms, historical coverage, timestamp semantics, and reproducible download or collection procedure. A web page listing a dataset is not proof that historical records or redistribution rights exist.

## Data sources

### 1. RIDE — mandatory offline benchmark fallback

Source: [RIDE arXiv page](https://arxiv.org/html/2606.05070).

Expected role: standardized external benchmark for train-delay forecasting and a reproducibility anchor.

Known characteristics from the paper/search record: Belgian nationwide network, 94.5M train events, 3.6M journeys, 35.7M weather records from 2023–2025, chronological benchmark with 2023–2024 training and 2025 test, and lite/standard subsets.

Audit items before use:

- confirm the released data URL, checksum, version, and data license;
- record the exact benchmark release and commit of any code;
- identify whether incident/disruption labels are available or whether disruption/recovery windows must be derived;
- preserve the official RIDE target definition and evaluation split;
- add the proposed feed-corruption stress tests without changing the clean benchmark comparison.

RIDE can establish delay/recovery robustness, but it is not Japanese validation.

### 2. ODPT — target Japanese operational setting

Sources: [ODPT overview](https://www.odpt.org/overview/), [Tokyo Metro catalog](https://ckan.odpt.org/organization/tokyometro), and [ODPT Challenge open-data page](https://challenge2025.odpt.org/ja/opendata.html).

The Tokyo Metro catalog currently lists GTFS/GTFS-JP static railway information, GTFS-Realtime railway information, train status information, timetables, station/route information, and passenger survey information. The challenge page identifies JR East GTFS and GTFS-RT train-location/operation data among the participating railway sources.

Target use:

- static topology and transfers;
- live/replayed trip updates and vehicle positions;
- service alerts and operation status;
- optional aggregate passenger survey features, not individual tracking.

Access gates:

1. register for the applicable ODPT/challenge account or API key;
2. read the specific dataset terms, rate limits, attribution requirements, and redistribution rules;
3. test whether historical snapshots are downloadable or whether a new collection period is required;
4. verify stable identifiers across static GTFS and GTFS-RT;
5. collect only permitted fields and retain raw snapshots with retrieval timestamps;
6. document whether the resulting derivative features can be released with the paper.

If only current feeds are available, begin a compliant collector and use a rolling evaluation after enough disruptions have accumulated. Do not call a short live collection a historical benchmark.

### 3. SURCONFORT / passenger reports — optional crowding extension

Source: [SURCONFORT paper](https://arxiv.org/html/2410.17510v1).

The paper demonstrates Japanese railway congestion forecasting using sparse passenger reports and a railway graph. It explicitly discusses subjective labels, sparse reports, and missing multi-line/platform information. Treat the paper as a methodological and motivation reference. Do not assume raw reports are publicly downloadable or redistributable.

The crowding extension is admitted only if one of these is secured:

- a permitted public release with clear labels and terms;
- written permission from the data owner;
- a new collection protocol with ethics/privacy review and an appropriate consent basis.

Otherwise, use transfer-risk as the passenger-facing output and leave crowding as future work.

## Canonical internal schema

Every normalized observation should retain provenance and availability timestamps:

- `source_operator`, `source_dataset`, `retrieval_timestamp_utc`;
- `service_date`, `trip_id`, `route_id`, `stop_id`, `direction_id`;
- `scheduled_arrival`, `scheduled_departure`;
- `observed_arrival`, `observed_departure`, `observation_timestamp`;
- `feed_timestamp`, `receipt_timestamp`, `observation_age_seconds`;
- `delay_seconds`, `vehicle_status`, `alert_id`;
- `is_missing`, `is_stale`, `is_duplicate`, `is_inconsistent`;
- `static_feed_version`, `raw_snapshot_hash`;
- derived `forecast_origin`, `horizon`, `incident_id`, and `regime`.

Never overwrite raw snapshots. Normalize into versioned parquet tables, and retain a data card describing fields that are unavailable for each operator.

## Incident and recovery labeling

Preferred order:

1. use explicit service-alert intervals and operator status feeds;
2. link alerts to affected routes/stations using documented rules;
3. if alerts are absent, derive candidate events from abrupt delay/headway changes, but label them as inferred and run sensitivity analysis;
4. define recovery using an operational threshold and sustained period fixed before test evaluation.

Do not use the post hoc “resolved” timestamp as an input at the forecast origin.

## Feed corruption simulator

The corruption layer must operate after the clean timestamped dataset is built and before feature construction, so it cannot accidentally reveal future values. Required scenarios:

- independent packet loss: 5%, 15%, 30%;
- contiguous outage: 5, 15, and 30 minutes;
- delayed updates: 1, 3, 5, and 10 minutes;
- stale vehicle record held until timeout;
- station-specific or operator-specific outage;
- contradictory timestamps and duplicate records;
- combined outage plus disruption.

Each corrupted stream receives a reproducible random seed and a corruption manifest.

## Fallback ladder

- **Fallback 0:** RIDE lite for pipeline development and clean/stress benchmark.
- **Fallback 1:** RIDE standard for final offline paper experiments.
- **Fallback 2:** Japanese static GTFS plus a permitted collection of ODPT GTFS-RT snapshots for topology and live-feed case studies.
- **Fallback 3:** Japanese data only for qualitative replay/demo if sample size is too small for statistical claims.
- **Do not use:** scraped services with unclear terms, private operational logs, synthetic delays presented as real observations, or passenger labels without permission.
