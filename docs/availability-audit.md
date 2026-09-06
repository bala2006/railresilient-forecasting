# Project availability audit

**Audit date:** 2026-09-06
**Project:** RailResilient / R2S-MoE
**Audit scope:** workspace, data, licenses, APIs, literature, runtimes, compute, storage, and implementation readiness.

## Executive decision

| Area | Status | Decision |
|---|---|---|
| Offline research prototype | **Available** | Start with RIDE Gold Lite. |
| Public benchmark and code | **Available** | RIDE code is public; Gold Lite, Gold Standard, and Silver releases are public. |
| Japanese static railway data | **Available conditionally** | Tokyo Metro catalog exposes GTFS/GTFS-JP, subject to account/token and terms. |
| Japanese real-time data | **Available conditionally** | Tokyo Metro catalog exposes GTFS-RT and service-status resources, but the API requires an access token. |
| Historical Japanese replay archive | **Not verified** | Do not assume one exists; begin a compliant collector if access is granted. |
| Transfer-risk ground truth | **Not available in RIDE** | Report derived connection-feasibility risk unless an actual passenger/transfer label source is obtained. |
| Real feed-age/missingness ground truth | **Not available in RIDE** | Use controlled corruption simulation offline; use ODPT collection for real feed-quality analysis. |
| Crowding labels | **Not available for this project** | Keep SURCONFORT/crowding optional and non-blocking. |
| CPU execution | **Available** | Small models and RIDE Lite are feasible; measure all latency claims. |
| GPU execution | **Not available locally** | Use CPU or external GPU only if needed. |
| Current implementation | **Not available** | The workspace contains specifications only; no pipeline or model code exists yet. |

## 1. Workspace audit

The workspace currently contains `README.md` and five planning documents under `docs/`:

- `research-spec.md`
- `literature-matrix.md`
- `data-access-audit.md`
- `experiment-protocol.md`
- `lightweight-architecture-review.md`

There is no source package, environment lock, dataset directory, downloader, test suite, model implementation, data card, or collection script. The project is therefore at **specification-complete / implementation-not-started** status.

The workspace is not currently a Git repository. Changes can be prepared locally, but there is no commit or branch workflow available until a repository is initialized or a remote project is supplied.

## 2. RIDE benchmark: verified availability

The RIDE source repository is public at [github.com/orailix/ride](https://github.com/orailix/ride). Its repository metadata reports a public, non-archived repository with an MIT code license. The repository contains download scripts, pipeline code, manifests, configuration, model/evaluation code, and documentation.

The releases are public on Hugging Face:

- [RIDE Gold Lite](https://huggingface.co/datasets/orailix/ride-gold-lite)
- [RIDE Gold Standard](https://huggingface.co/datasets/orailix/ride-gold-standard)
- [RIDE Silver](https://huggingface.co/datasets/orailix/ride-silver)

The repository's `DATA_LICENSE.md` assigns **CC BY 4.0** to the released RIDE datasets and asks users to attribute RIDE, Infrabel, and Open-Meteo. The source code is MIT. The RIDE paper reports Infrabel source data as CC0 and Open-Meteo data as CC BY 4.0. Verify the license files again at the time of publication in case the release changes.

### Verified release sizes

The following totals were computed from the public Hugging Face dataset-tree metadata on the audit date:

| Release | Approximate stored size | Practical use |
|---|---:|---|
| Gold Lite | **26.48 GB** | Recommended first benchmark and development tier |
| Gold Standard | **68.37 GB** | Final offline benchmark if storage permits |
| Silver | **1.12 GB** | Relational/event-level data for custom features and labels |

The Gold Lite release contains 50 files and about 623k core rows. Gold Standard contains 50 files and about 2.09M core rows. Silver contains 94.5M event rows across monthly Parquet files plus journeys, topology, and weather tables.

The public release is genuinely downloadable: Hugging Face resolves the release files to public CDN URLs, and the GitHub repository publishes the downloader and component-level targets. The project should download only `gold_lite_core`, `gold_lite_tabular`, or `gold_lite_sequential` initially, not every component.

### What RIDE does and does not provide

RIDE provides historical observed/scheduled train events, journeys, infrastructure, and weather. It is suitable for delay prediction, chronological benchmarking, and reproducible synthetic feed-corruption experiments.

RIDE does **not** provide the following as first-class observed data:

- feed receipt timestamps or observation-age metadata;
- a live GTFS-RT replay archive;
- authoritative disruption-alert intervals;
- observed passenger transfer outcomes;
- individual passenger data;
- a transfer-label table;
- a Japanese railway network.

Therefore:

- feed delays, packet loss, stale records, duplicates, and inconsistent timestamps must be simulated for the RIDE benchmark;
- disruption/recovery intervals must be derived from event trajectories or another source and sensitivity-tested;
- transfer risk derived from predicted arrival distributions must be called **connection-feasibility risk**, not observed passenger missed-transfer probability;
- Japanese relevance requires a separate ODPT case study.

The RIDE paper and release are public, but the audit did not verify peer-reviewed publication status independently. Cite it as an arXiv/preprint resource unless an official proceedings record is confirmed before submission.

## 3. ODPT Japanese data: verified availability and access gate

The [Tokyo Metro ODPT catalog](https://ckan.odpt.org/organization/tokyometro) currently lists:

- [Tokyo Metro GTFS/GTFS-JP railway information](https://ckan.odpt.org/en/dataset/train-tokyometro);
- [Tokyo Metro GTFS-Realtime railway information](https://ckan.odpt.org/en/dataset/r_train_gtfs_rt-odpt_train-tokyometro);
- [Tokyo Metro service-status JSON](https://ckan.odpt.org/en/dataset/r_train_status-tokyometro);
- station, route, timetable, train timetable, passenger survey, and other related datasets.

The [ODPT Challenge 2026 open-data page](https://challenge2026.odpt.org/ja/opendata.html) also states that JR East GTFS and GTFS-RT train-location/operation data, along with other railway data, are being provided through the challenge ecosystem. Challenge-only data have separate terms and must not be treated as ordinary ODPT data without checking the applicable license.

The catalog exposes actual resource URL templates. For example, Tokyo Metro static GTFS uses an endpoint of the form:

```text
https://api.odpt.org/api/v4/files/TokyoMetro/data/TokyoMetro-Train-GTFS.zip?acl:consumerKey=YOUR_ACCESS_TOKEN
```

The GTFS-RT resource is exposed as a Protobuf endpoint, and the service-status resource is exposed as JSON. Direct requests made during this audit without a token returned **HTTP 403**, confirming that catalog visibility does not equal unauthenticated data access.

The [official ODPT rules/guidelines page](https://developer.odpt.org/terms) confirms that the applicable data license requires account information registration before use, and that provider-specific terms may add restrictions. The basic license permits public or commercial deliverables subject to the license and guidelines, but it generally prohibits redistributing raw data, copies, or derivatives from which most of the original data can be restored without prior written approval. It also requires accurate representation, attribution/usage compliance, and user responsibility for the resulting deliverable.

### ODPT gates still open

1. Create/register the required developer account.
2. Obtain an access token.
3. Test static GTFS, GTFS-RT, and status endpoints.
4. Record rate limits, polling rules, attribution, retention, and publication restrictions.
5. Determine whether historical files or only current feeds are available.
6. Start a compliant raw-snapshot collector if historical replay is not offered.
7. Preserve retrieval timestamps and raw snapshot hashes.
8. Do not publish raw ODPT files or mostly-reconstructable derivatives without written permission.

The ODPT Japanese validation path is therefore **realistic but not yet unlocked** in this cloud session because no user access token or account credentials are available.

## 4. SURCONFORT and crowding extension

The [SURCONFORT paper](https://arxiv.org/html/2410.17510v1) is publicly available and is a published ACM SIGSPATIAL 2024 work. Its experiments use six months of passenger reports collected through the LY Corporation transit application, with sparse congestion labels on JR Yamanote.

The paper does not provide a public downloadable raw-report release or a redistribution license for those reports. The paper also documents subjectivity and sparsity limitations. Consequently, SURCONFORT is useful evidence that Japanese passenger congestion is important, but it is **not a usable project dependency** unless written permission or a permitted public release is obtained.

The core project remains feasible without crowding. Do not collect passenger identity data, individual trajectories, facial data, or private app logs.

## 5. Local compute and storage

The sandbox audit found:

- 8 logical CPU cores;
- approximately 30 GiB RAM;
- approximately 89 GiB free space on `/projects/sandbox`;
- approximately 16 GiB temporary space on `/tmp`;
- no visible NVIDIA GPU or `nvidia-smi` executable;
- Python 3.9 is the system `python3`, but Python 3.11.15 is installed through pyenv;
- `uv` 0.12.1 is available;
- Node.js 22.23.2 is available but not required.

The scientific Python/ML packages are **not installed** in the system Python environment. A dry-run against the RIDE Python 3.11 requirements resolved successfully to 111 packages, including the requested versions of NumPy, pandas, PyArrow, SciPy, scikit-learn, XGBoost, PyTorch, PyTorch Geometric, Optuna, and geospatial libraries.

The default PyPI resolution selected CUDA/NVIDIA packages even though no local GPU is present. A CPU-only PyTorch 2.10.0 resolution is available from the official PyTorch CPU index. The eventual environment should use Python 3.11 and an explicit CPU PyTorch installation, then install the remaining requirements. Do not install the default CUDA bundle into this 89 GiB workspace unless an actual GPU becomes available.

### Compute decision

- RIDE Gold Lite, statistical baselines, calibration, corruption simulation, and the proposed small R2S-MoE are feasible on CPU.
- Full RIDE Gold Standard GNN experiments are storage- and time-intensive locally and should be staged or run on external compute.
- The target of 0.2–1.5M trainable parameters is compatible with CPU development, but the p95 under-one-second target must be measured rather than assumed.
- Downloading Gold Lite plus a full Python environment and experiment outputs is feasible but should be managed carefully.
- Downloading Gold Standard, Silver, caches, environments, and many checkpoints together risks exhausting the workspace.

## 6. Literature and novelty availability

The literature is available for the main comparison and gap analysis. The closest verified subsets include:

- [Probabilistic modeling of delays for train journeys with transfers](https://arxiv.org/html/2504.17479v1): probabilistic delay plus transfer reliability;
- [Hierarchical multi-leg arrival and transfer-feasibility prediction](https://link.springer.com/article/10.1186/s12544-026-00814-4): transfer feasibility and aggregated journey prediction;
- [Bayesian post-disruption travel-time prediction](https://arxiv.org/html/2602.19952): calibrated probabilistic recovery-related prediction;
- [Nguyen and Li cascading railway delay forecasting](https://arxiv.org/html/2510.09350v1): live multi-step graph-based delay propagation;
- [RIDE](https://arxiv.org/html/2606.05070v1): open railway benchmark and strong baseline reference;
- [Time-MoE](https://arxiv.org/html/2409.16040v2), [Moirai-MoE](https://arxiv.org/html/2410.10469v1), and [TFMoE](https://arxiv.org/html/2406.03140v1): existing time-series/transport MoE precedents;
- [RTRI disruption congestion research](https://www.rtri.or.jp/eng/publish/rtrirep/2025/olr0hr0000000ech-att/rep202507s-4.pdf): direct Japanese relevance precedent.

The current web-level review still finds no exact match for the full combination of reliability-conditioned railway expert routing, online uncertainty calibration, transfer/connection-risk alerts, and controlled delayed/missing public-feed evaluation. This remains a **conditional literature gap**, not a priority guarantee. A final search through Scopus, Web of Science, IEEE Xplore, ACM Digital Library, TRID, and Google Scholar is still required before submission.

## 7. Final readiness assessment

### Ready now

- Freeze the RIDE-based offline schema.
- Install a Python 3.11 CPU environment.
- Download RIDE Gold Lite core/tabular or sequential artifacts.
- Implement schedule, persistence, Markov/autoregressive, and quantile boosting baselines.
- Build leakage-safe corruption simulation.
- Implement quantile metrics and rolling/conformal calibration.
- Measure CPU latency and memory.

### Blocked or conditional

- Japanese live/historical validation: requires ODPT account/token and data-terms review.
- Real feed-age study: requires collected ODPT snapshots with retrieval timestamps.
- Empirical passenger missed-transfer validation: requires a permitted transfer/passenger label source.
- Crowding head: not available without a new legal/ethical data path.
- Full standard-tier experiments: local storage/compute is constrained.
- Final novelty claim: requires database-level search and supervisor review.

### Recommended go/no-go

**GO for the offline core project.** RIDE is publicly available, licensed, reproducible, and large enough to support the delay/recovery robustness paper.
**Conditional GO for the Japanese case study.** Begin ODPT registration immediately, but do not promise historical Japanese validation until access and archival coverage are verified.
**NO-GO for making crowding or observed passenger-transfer labels a core dependency.** Keep both as optional extensions.

## Sources checked

- [RIDE paper](https://arxiv.org/html/2606.05070v1)
- [RIDE code repository](https://github.com/orailix/ride)
- [RIDE Gold Lite](https://huggingface.co/datasets/orailix/ride-gold-lite)
- [RIDE Gold Standard](https://huggingface.co/datasets/orailix/ride-gold-standard)
- [RIDE Silver](https://huggingface.co/datasets/orailix/ride-silver)
- [Tokyo Metro ODPT catalog](https://ckan.odpt.org/organization/tokyometro)
- [ODPT Tokyo Metro GTFS](https://ckan.odpt.org/en/dataset/train-tokyometro)
- [ODPT Tokyo Metro GTFS-RT](https://ckan.odpt.org/en/dataset/r_train_gtfs_rt-odpt_train-tokyometro)
- [ODPT rules and licenses](https://developer.odpt.org/terms)
- [ODPT Challenge 2026 open data](https://challenge2026.odpt.org/ja/opendata.html)
- [SURCONFORT](https://arxiv.org/html/2410.17510v1)
