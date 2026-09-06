# Figure guide

The committed figures are lightweight, accessible SVGs with embedded titles, descriptions, source notes, and explicit evidence caveats.

| Figure | Purpose | Source |
|---|---|---|
| [`../architecture-family.svg`](../architecture-family.svg) | Baselines and R2S/R3S/R4S architecture evolution. | Architecture descriptions in `docs/model-architectures.md` and `src/railresilient/models.py`. |
| [`../experiment-flow.svg`](../experiment-flow.svg) | Chronological training, calibration, test, and stress path. | `docs/experiment-protocol.md` and the experiment runner. |
| [`../v4-clean-benchmark.svg`](../v4-clean-benchmark.svg) | Clean MAE and WIS for the fresh v4 variants. | [`../results/v4-variants.csv`](../results/v4-variants.csv). |
| [`../v4-stress-benchmark.svg`](../v4-stress-benchmark.svg) | WIS under four feed-corruption scenarios. | [`../results/v4-variants.csv`](../results/v4-variants.csv). |
| [`../architecture-r4s.svg`](../architecture-r4s.svg) | Detailed R4S-MoE top-2 architecture. | `docs/r4s-top2-design.md`. |
| [`../clean-wis.svg`](../clean-wis.svg) | Historical v3 clean WIS figure. | Historical v3 result summary. |
| [`../stress-robustness.svg`](../stress-robustness.svg) | Historical v2/v3 stress degradation figure. | Historical v2/v3 result summary. |

The new figures are hand-authored SVG summaries backed by compact CSV source tables. They are not substitutes for raw prediction bundles or multi-seed confidence analysis.
