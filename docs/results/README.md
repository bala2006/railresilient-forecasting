# Documentation result sources

These compact CSV files are the source tables for the reader-facing benchmark pages and SVG figures. They contain summary metrics only; raw data, checkpoints, and prediction bundles remain local and ignored.

- [`model-history.csv`](model-history.csv) — one-row-per-version history and evidence scope.
- [`v4-variants.csv`](v4-variants.csv) — five fresh v4 variants plus the earlier top-2 control.
- [`../benchmarks.csv`](../benchmarks.csv) — legacy compact benchmark table retained for compatibility.

All v3 and v4 learned-model rows use the selected RIDE Silver protocol and seed `20260908` unless the `seed_scope` column says otherwise. Values are exploratory where the scope is one seed.
