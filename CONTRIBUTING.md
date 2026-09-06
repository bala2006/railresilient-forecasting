# Contributing

Thank you for helping improve RailResilient.

## Good contributions

- Reproducible baselines and ablations
- Better causal leakage checks
- Clear documentation and visualizations
- CPU-efficiency improvements
- Replication results on public railway datasets

## Before opening a pull request

1. Explain the research or engineering motivation.
2. Keep train/validation/test chronology intact.
3. Do not tune on the test set.
4. Do not add credentials, raw private data, checkpoints, or large prediction bundles.
5. Run:

```bash
ruff check src
pyright --pythonpath .venv/bin/python src
python -m compileall -q src
```

Please label results as exploratory when they use a new seed, dataset, or evaluation protocol. Avoid claims of Japanese deployment, passenger outcomes, safety certification, or absolute novelty unless they are directly supported by new evidence.
